# CivicLens KG - Member 1 deliverable (Knowledge Graph)

Everything the other modules need to know what a complaint is connected to.
**All data is illustrative sample data** (Mumbai-area coordinates, invented complaints).

```
CivicLensKG/
├── complaints.csv          # shared dataset: 80 complaints (the agreed schema)
├── knowledge_graph.py      # the module: build graph, query relationships, explain
├── prepare_images.py       # links pothole / garbage / streetlight photos to complaints (see below)
├── images/                 # one photo per complaint (80) + image_manifest.csv + SOURCES.md
├── demo/                   # c101_graph.png, c101_related_graph.png, graph.json
├── tests/test_knowledge_graph.py
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
python -m pytest -q            # 17 tests
python knowledge_graph.py      # demo: summary + C101 reasoning paths
```

## The dataset (`complaints.csv`)

Columns (do not rename): `complaint_id, issue_type, description, latitude, longitude, area,
road_name, road_importance, department, status, image_path, created_date`

| Field | Values in the sample data |
|---|---|
| `issue_type` | Pothole, Damaged Road (Roads) · Garbage (Sanitation) · Streetlight (Streetlight) |
| `department` | Roads, Sanitation, Streetlight - these are the three separate indexes for Member 5 |
| `status` | Unresolved, In Progress, Resolved |
| `road_importance` | Major, Minor (one value per road name) |
| `area` | Andheri, Jogeshwari, Vile Parle, Santacruz, Bandra, Kurla, Powai, Goregaon, Malad |
| `image_path` | `images/<ID>.jpg`. Filled for all 80 complaints. **Every complaint has one (C101-C180).** |

Built-in cases for demos: C101 / C102 are near-duplicates (same road, ~30 m apart), C113 / C114
likewise; C122 / C131 are duplicate garbage reports; C104 is a pothole on a *minor* road near
C101, so C101 should outrank it; several complaints are Resolved or In Progress.

## The graph

| Relationship | Meaning |
|---|---|
| `Complaint -[LOCATED_AT]-> Location` | area of the complaint |
| `Complaint -[AFFECTS_ROAD]-> Road` | road (has `importance`) |
| `Complaint -[ASSIGNED_TO]-> Department` | owning department |
| `Complaint -[HAS_EVIDENCE]-> EvidenceImage` | photo (has `path`, `exists`) |
| `Road -[PASSES_THROUGH]-> Location` | derived: roads serve every area they have complaints in |

Sample graph: 188 nodes (80 Complaint, 9 Location, 16 Road, 3 Department, 80 EvidenceImage), 344 edges.

## API cheat sheet

```python
from knowledge_graph import (load_graph, get_relationships, reasoning_paths,
    graph_explanation, get_road_importance, related_complaints, ...)

G = load_graph("complaints.csv")          # validates the CSV, raises ValueError if it is bad

get_relationships(G, "C101")              # dict: attributes, location, road, department, evidence, edges, paths
reasoning_paths(G, "C101")                # ['C101 -[LOCATED_AT]-> Andheri', 'C101 -[AFFECTS_ROAD]-> Main Road (Major)', ...]
graph_explanation(G, "C101")              # one sentence for the result card
get_road_importance(G, "C101")            # 'Major'  (use for the R term in S = 0.5T + 0.3G + 0.2R)
get_department(G, "C101")                 # 'Roads'
related_complaints(G, "C101", same_issue_only=True)   # same road/area reports + the graph path (duplicate review)
complaints_on_road(G, "Main Road")        # ['C101', 'C102', ...]
complaints_at_location(G, "Andheri")
complaints_for_department(G, "Roads")
roads_in_location(G, "Andheri")
graph_summary(G)                          # counts per node type / relation
draw_complaint_subgraph(G, "C101", "c101.png", show_related=True)
export_graph_json(G, "graph.json")
```

Who uses what:
- **Member 2:** `get_road_importance` for ranking, `graph_explanation` / `reasoning_paths` for the
  "why retrieved" text, `related_complaints` as extra evidence for duplicate detection.
- **Member 3:** `Location` nodes carry centroid `latitude` / `longitude`; `roads_in_location`.
- **Member 4:** `get_evidence(G, id)` gives the photo path and whether the file exists; garbage and
  streetlight photos are added with `prepare_images.py` (see "Photos" below).
- **Member 5:** each complaint's `department` is its index; call `load_graph` once and pass `G`
  around (e.g. `@st.cache_resource` in Streamlit).

`G.graph["validation_warnings"]` lists non-fatal problems (e.g. an `image_path` whose file is missing).

## Retrieval, Ranking, and Duplicate Detection (`retrieval.py` - Member 2)

Hybrid Information Retrieval combining lexical matching (TF-IDF) and dense semantic search
(`sentence-transformers/all-MiniLM-L6-v2`) with explainability and duplicate detection.

```bash
python retrieval.py                         # demo hybrid search
python retrieval.py --duplicates            # detect potential duplicate complaint pairs
python retrieval.py --eval                  # run 8-query evaluation benchmark (Precision@5, Recall@5, latency)
```

```python
from retrieval import ComplaintRetriever, search_complaints, find_duplicate_complaints

# Quick search:
results = search_complaints(
    "unresolved pothole near bus stop",
    complaints="complaints.csv",
    top_k=5,
    status="Unresolved",
    alpha=0.4,           # 0.4 lexical (TF-IDF) + 0.6 dense (Sentence Transformers)
    rank_by="composite", # S = 0.5*T + 0.3*G + 0.2*R
    G=G,                 # optional NetworkX graph from Member 1
)
for r in results:
    print(r.complaint_id, r.ranking_score, r.hybrid_score, r.explanation)

# Detect candidate duplicates:
dupes = find_duplicate_complaints("complaints.csv", threshold=0.85, G=G)
for d in dupes:
    print(d.complaint_id_1, d.complaint_id_2, d.similarity, d.explanation)
```


## Photos (potholes, garbage, streetlights)

**All 80 complaints have a photo in `images/`** (`images/<complaint id>.jpg`, max 800 px, about 5.6 MB
in total), so teammates do not need to download anything. Photo sources, with per-photo credits in
`images/SOURCES.md`:

| Category (`issue_type`) | Complaints | Source | Licence |
|---|---|---|---|
| pothole (Pothole, Damaged Road) | 61 | Kaggle [andrewmvd/pothole-detection](https://www.kaggle.com/datasets/andrewmvd/pothole-detection) (Make ML's Potholes Dataset) | DbCL v1.0 |
| garbage (Garbage) | 10 | 10 street-garbage photos from Wikimedia Commons | CC BY / CC BY-SA / public domain, credited per photo |
| streetlight (Streetlight) | 9 | 9 street-lamp photos from Wikimedia Commons | CC BY / CC BY-SA, credited per photo |

The photos illustrate each problem in general; they do not show the places named in the CSV, and the
streetlight photos show lamps that work, not broken ones. `images/image_manifest.csv` maps each
complaint to the original file.

### Regenerating or replacing photos

`prepare_images.py` links photos in a folder to the complaints of one category (any combination of
flags; categories you leave out are not touched):

```bash
python prepare_images.py --pothole-src ~/Downloads/pothole \
                         --garbage-src ~/Downloads/garbage \
                         --streetlight-src ~/Downloads/streetlights \
                         --streetlight-credit "where these photos came from, with licence"
```

For each complaint of a category it sets a blank `image_path` to `images/<id>.jpg` (rewriting
`complaints.csv`), copies one photo there resized to 800 px, and updates `image_manifest.csv` and
`SOURCES.md` (one attribution section per category). Photos are picked evenly and deterministically.
Each folder needs at least as many photos as there are complaints of that category. Do not commit
raw downloads, and always pass the credit text for photos that need attribution.
