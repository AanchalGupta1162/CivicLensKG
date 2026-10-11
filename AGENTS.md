# CivicLens KG - context for LLM coding assistants

Read this before changing anything. It describes the whole project, what exists today, and the
rules that keep five people's modules compatible.

## What this project is
CivicLens KG is a one-day academic prototype: explainable retrieval of civic complaints
(potholes, garbage, broken streetlights, damaged roads). A municipal officer asks, e.g.
"unresolved pothole complaints within 2 km of Andheri that affect major roads", and gets a ranked,
explained list, a map, a knowledge-graph reasoning path, and image search.
SDGs: 11 (Sustainable Cities) and 16 (Strong Institutions). **All data is illustrative sample data**
(invented complaints at Mumbai-area coordinates), so never present it as real.

Stack: Python, Pandas, NetworkX, scikit-learn, Folium, Streamlit, Pillow (+ CLIP/FAISS optional).
No Neo4j, no real cluster: "distributed" = three separately queried in-memory indexes that are merged.

## Team modules (one owner each)
| # | Module | File | Status |
|---|--------|------|--------|
| 1 | Knowledge graph | `knowledge_graph.py` | **Done, tested** |
| 2 | Retrieval, ranking, explanations, duplicates | `retrieval.py` | not started |
| 3 | Geospatial (Haversine, radius, hotspots, Folium map) | `geo.py` | not started |
| 4 | Multimedia / image similarity | `multimedia.py` | not started |
| 5 | Distributed IR + Streamlit dashboard | `distributed_ir.py`, `app.py` | not started |

Do not edit another member's file unless asked. Import from it instead.

## The shared dataset - `complaints.csv` (the contract)
80 rows, IDs `C101`-`C180`. Columns, exact names, all read as strings except lat/lon:

`complaint_id, issue_type, description, latitude, longitude, area, road_name, road_importance, department, status, image_path, created_date`

- `issue_type`: Pothole, Damaged Road (-> Roads); Garbage (-> Sanitation); Streetlight (-> Streetlight)
- `department`: Roads | Sanitation | Streetlight. These are the three separate indexes for distributed IR.
- `status`: Unresolved | In Progress | Resolved
- `road_importance`: Major | Minor. One value per `road_name` (a road is one graph node).
- `created_date`: YYYY-MM-DD
- `image_path`: `images/<complaint_id>.jpg` or blank. Filled for the 61 Pothole/Damaged Road rows
  (C101-C121, C141-C180). Garbage/Streetlight rows (C122-C140) are blank until Member 4 adds photos.
- Areas: Andheri, Jogeshwari, Vile Parle, Santacruz, Bandra, Kurla, Powai, Goregaon, Malad.

Do not rename/reorder columns or change types. Adding rows is fine: keep IDs unique, run the tests.
Planted test cases: C101/C102/C141/C142 near-duplicate potholes on Main Road (Andheri); C113/C114
Kurla duplicates; C122/C131 duplicate garbage; C104 is a *minor-road* pothole near C101 (C101 should
outrank it); some rows are Resolved/In Progress to exercise status filters.

## Knowledge graph (`knowledge_graph.py`)
NetworkX `DiGraph`. Node ids are namespaced strings (`"Complaint:C101"`, `"Road:Main Road"`); build them
with `node_id(kind, key)`, never by hand. Node types: Complaint, Location, Road, Department, EvidenceImage.
Edges carry `relation`:
`Complaint-LOCATED_AT->Location`, `Complaint-AFFECTS_ROAD->Road`, `Complaint-ASSIGNED_TO->Department`,
`Complaint-HAS_EVIDENCE->EvidenceImage`, `Road-PASSES_THROUGH->Location` (derived).
Sample graph: 169 nodes, 325 edges. Location nodes hold centroid lat/lon; Road nodes hold `importance`.

Public API (use these; see the module docstrings for details):
```python
G = load_graph("complaints.csv")            # validates, raises ValueError on bad data
get_relationships(G, "C101")                # dict: attributes, location, road, department, evidence, edges, paths
reasoning_paths(G, "C101")                  # ['C101 -[AFFECTS_ROAD]-> Main Road (Major)', ...]
graph_explanation(G, "C101")                # one sentence for result cards
get_road_importance / get_road / get_department / get_location / get_evidence
related_complaints(G, "C101", same_issue_only=True)   # shared road/area + connecting graph path
complaints_on_road / complaints_at_location / complaints_for_department / roads_in_location
graph_summary(G); draw_complaint_subgraph(G, id, "out.png", show_related=True); export_graph_json(G, "g.json")
```
`G.graph["validation_warnings"]` holds non-fatal issues (e.g. image file missing). Load the graph once
and reuse it (in Streamlit use `@st.cache_resource`).

## Retrieval and ranking rules (for Members 2 and 5)
- Hard filters first - issue type, status, distance radius. A high score must never rescue a result
  that fails a filter.
- Then rank survivors with `S = 0.5*T + 0.3*G + 0.2*R`, each term normalised to 0-1:
  T = text relevance (TF-IDF cosine on `description`), G = geographic proximity, R = road importance
  (Major high, Minor low; get it from `get_road_importance`).
- Every result needs an explanation: matched filters, the three score terms, distance, and the graph
  path from `reasoning_paths` / `graph_explanation`.
- Distributed IR: query Roads, Sanitation and Streetlight collections independently, merge, dedupe by
  `complaint_id`, rank once. Use the three `department` values as the split.
- Evaluation: 5-10 test queries with expected relevant IDs; report Precision@5, Recall@5, latency.
  Image search: show top 3 matches and manually check relevance.

## Images
`images/` holds 61 Kaggle pothole photos (JPEG, max 800 px), one per Pothole/Damaged Road complaint, plus
`image_manifest.csv` (complaint -> original file) and `SOURCES.md` (attribution). Source: Pothole
Detection by Larxel (andrewmvd) on Kaggle, built on Make ML's Potholes Dataset, Database Contents
License v1.0. Photos are generic pothole pictures, not the places named in the CSV. Keep the attribution.
`prepare_images.py --src <unzipped kaggle folder>` regenerates them deterministically; the raw 351 MB
download is not in the repo and must not be committed. If you do image search with a fallback instead of
real CLIP embeddings, label it honestly rather than calling it visual-semantic search.

## Commands
```bash
pip install -r requirements.txt
python -m pytest -q                  # 17 tests, must stay green
python knowledge_graph.py            # demo: summary + C101 reasoning paths and related reports
python knowledge_graph.py --complaint C113 --draw demo/c113.png --related
```

## Conventions and gotchas
- Match existing style: type hints, `from __future__ import annotations`, short docstrings, no heavy frameworks.
- Paths: `image_path` is relative to the folder holding `complaints.csv`.
- Don't commit `__pycache__`, `.DS_Store`, `.pytest_cache` or the raw Kaggle `archive*` folder.
- Don't invent real-looking data (real addresses, phone numbers, names); keep it labelled as sample data.
- The graph builder rejects bad data on purpose (duplicate IDs, bad status/department/importance,
  conflicting road importance, bad dates, missing coordinates). Fix the CSV rather than loosening the checks.
- Keep scope small: one-day prototype. Prefer a working simple version over new infrastructure.
