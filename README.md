# CivicLens KG - Member 1 deliverable (Knowledge Graph)

Everything the other modules need to know what a complaint is connected to.
**All data is illustrative sample data** (Mumbai-area coordinates, invented complaints).

```
CivicLensKG/
├── complaints.csv          # shared dataset: 80 complaints (the agreed schema)
├── knowledge_graph.py      # the module: build graph, query relationships, explain
├── prepare_images.py       # links Kaggle pothole photos to complaints (see below)
├── images/                 # 61 photos (C101-C121, C141-C180) appear here after prepare_images.py
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
| `image_path` | `images/<ID>.jpg`. Filled for all 61 pothole / damaged-road complaints (C101-C121, C141-C180). **Blank for garbage and streetlight complaints; Member 4 adds those photos and fills the column.** |

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

Sample graph: 169 nodes (80 Complaint, 9 Location, 16 Road, 3 Department, 61 EvidenceImage), 325 edges.

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
- **Member 4:** `get_evidence(G, id)` gives the photo path and whether the file exists; to add
  garbage/streetlight photos, drop `images/Cxxx.jpg` and fill `image_path`.
- **Member 5:** each complaint's `department` is its index; call `load_graph` once and pass `G`
  around (e.g. `@st.cache_resource` in Streamlit).

`G.graph["validation_warnings"]` lists non-fatal problems (e.g. an `image_path` whose file is missing).

## Pothole photos (Kaggle)

The 61 photos are **already in `images/`**; teammates do not need to download anything. The steps
below are only for regenerating them.

Source: https://www.kaggle.com/datasets/andrewmvd/pothole-detection (665 images, 351 MB,
Database Contents License v1.0; credit "Potholes Dataset", Make ML). Kaggle needs a login, so
download it yourself:

```bash
pip install kaggle      # needs an API token in ~/.kaggle/kaggle.json
kaggle datasets download -d andrewmvd/pothole-detection -p ~/Downloads/pothole --unzip
```
(or use "Download" on the Kaggle page and unzip it.) Then:

```bash
python prepare_images.py --src ~/Downloads/pothole
```

This copies 61 photos (picked evenly and deterministically, resized to 800 px) to
`images/<complaint id>.jpg` and writes `images/image_manifest.csv` and `images/SOURCES.md`
with the attribution. Everyone running it gets identical files. Do not commit the raw
351 MB download.

The photos illustrate potholes in general; they do not show the places named in the CSV.
