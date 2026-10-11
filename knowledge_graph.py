"""CivicLens KG - Member 1: Knowledge Graph module.

Builds a NetworkX knowledge graph from complaints.csv and answers
"what is this complaint connected to, and why?" for the rest of the team.

Entities (node types)
    Complaint, Location, Road, Department, EvidenceImage

Relationships (edge `relation`)
    Complaint -LOCATED_AT->    Location
    Complaint -AFFECTS_ROAD->  Road
    Complaint -ASSIGNED_TO->   Department
    Complaint -HAS_EVIDENCE->  EvidenceImage
    Road      -PASSES_THROUGH-> Location      (derived: a road serves every area it has complaints in)

Node ids are namespaced strings such as "Complaint:C101" or "Road:Main Road".
Other modules should not build these by hand; use `node_id(kind, key)`.

Quick use
    from knowledge_graph import load_graph, get_relationships, graph_explanation
    G = load_graph("complaints.csv")
    get_relationships(G, "C101")["paths"]
    graph_explanation(G, "C101")
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx
import pandas as pd

# ---------------------------------------------------------------------------
# Shared schema (agreed by all members)
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS = [
    "complaint_id", "issue_type", "description", "latitude", "longitude",
    "area", "road_name", "road_importance", "department", "status",
    "image_path", "created_date",
]
VALID_STATUSES = {"Unresolved", "In Progress", "Resolved"}
VALID_DEPARTMENTS = {"Roads", "Sanitation", "Streetlight"}
VALID_IMPORTANCE = {"Major", "Minor"}

# Node types
COMPLAINT, LOCATION, ROAD, DEPARTMENT, IMAGE = (
    "Complaint", "Location", "Road", "Department", "EvidenceImage",
)
# Relations
LOCATED_AT, AFFECTS_ROAD, ASSIGNED_TO, HAS_EVIDENCE, PASSES_THROUGH = (
    "LOCATED_AT", "AFFECTS_ROAD", "ASSIGNED_TO", "HAS_EVIDENCE", "PASSES_THROUGH",
)

_COMPLAINT_ATTRS = [
    "issue_type", "description", "latitude", "longitude", "status", "created_date",
]


def node_id(kind: str, key: str) -> str:
    """Build the namespaced graph id, e.g. node_id('Road', 'Main Road')."""
    return f"{kind}:{key}"


# ---------------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------------
def load_complaints(csv_path: str | Path) -> pd.DataFrame:
    """Read complaints.csv as strings (blank -> '') and coerce coordinates."""
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    df = df.apply(lambda col: col.str.strip())
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    return df


def validate_complaints(df: pd.DataFrame, base_dir: str | Path | None = None) -> dict:
    """Check the dataset. Returns {"errors": [...], "warnings": [...]}.

    Errors make the graph unreliable (build_graph refuses to run).
    Warnings are things to fix but harmless (e.g. an image file not copied yet).
    """
    errors: list[str] = []
    warnings: list[str] = []

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return {"errors": [f"missing columns: {missing}"], "warnings": []}

    dupes = df.loc[df["complaint_id"].duplicated(), "complaint_id"].tolist()
    if dupes:
        errors.append(f"duplicate complaint_id: {sorted(set(dupes))}")

    for row in df.itertuples():
        cid = row.complaint_id or "<blank id>"
        if not row.complaint_id:
            errors.append("row with blank complaint_id")
        if pd.isna(row.latitude) or pd.isna(row.longitude):
            errors.append(f"{cid}: missing or non-numeric coordinates")
        elif not (-90 <= row.latitude <= 90 and -180 <= row.longitude <= 180):
            errors.append(f"{cid}: coordinates out of range")
        if row.status not in VALID_STATUSES:
            errors.append(f"{cid}: invalid status '{row.status}'")
        if row.department not in VALID_DEPARTMENTS:
            errors.append(f"{cid}: invalid department '{row.department}'")
        if row.road_importance not in VALID_IMPORTANCE:
            errors.append(f"{cid}: invalid road_importance '{row.road_importance}'")
        for field in ("issue_type", "area", "road_name"):
            if not getattr(row, field):
                errors.append(f"{cid}: blank {field}")
        if pd.isna(pd.to_datetime(row.created_date, errors="coerce", format="%Y-%m-%d")):
            errors.append(f"{cid}: created_date '{row.created_date}' is not YYYY-MM-DD")
        if row.image_path and base_dir is not None:
            if not (Path(base_dir) / row.image_path).is_file():
                warnings.append(f"{cid}: image file not found: {row.image_path}")

    # A road is one node, so it can only have one importance.
    per_road = df.groupby("road_name")["road_importance"].nunique()
    for road in per_road[per_road > 1].index:
        errors.append(f"road '{road}' has conflicting road_importance values")

    return {"errors": errors, "warnings": warnings}


# ---------------------------------------------------------------------------
# Building the graph
# ---------------------------------------------------------------------------
def build_graph(df: pd.DataFrame, base_dir: str | Path | None = None) -> nx.DiGraph:
    """Build the knowledge graph. Raises ValueError if the data has errors.

    base_dir is where image_path is resolved from (normally the CSV's folder);
    it only decides the `exists` flag on EvidenceImage nodes.
    """
    report = validate_complaints(df, base_dir)
    if report["errors"]:
        raise ValueError("Invalid complaints data:\n  " + "\n  ".join(report["errors"]))

    G = nx.DiGraph()
    G.graph["validation_warnings"] = report["warnings"]

    for row in df.itertuples():
        c = node_id(COMPLAINT, row.complaint_id)
        loc = node_id(LOCATION, row.area)
        road = node_id(ROAD, row.road_name)
        dept = node_id(DEPARTMENT, row.department)

        G.add_node(
            c, node_type=COMPLAINT, label=row.complaint_id,
            **{a: getattr(row, a) for a in _COMPLAINT_ATTRS},
        )
        G.add_node(loc, node_type=LOCATION, label=row.area)
        G.add_node(road, node_type=ROAD, label=row.road_name,
                   importance=row.road_importance)
        G.add_node(dept, node_type=DEPARTMENT, label=row.department)

        G.add_edge(c, loc, relation=LOCATED_AT)
        G.add_edge(c, road, relation=AFFECTS_ROAD)
        G.add_edge(c, dept, relation=ASSIGNED_TO)
        G.add_edge(road, loc, relation=PASSES_THROUGH)

        if row.image_path:
            img = node_id(IMAGE, row.image_path)
            exists = None if base_dir is None else (Path(base_dir) / row.image_path).is_file()
            G.add_node(img, node_type=IMAGE, label=Path(row.image_path).name,
                       path=row.image_path, exists=exists)
            G.add_edge(c, img, relation=HAS_EVIDENCE)

    # Location centroid = mean of its complaints (used for map / distance work).
    coords = df.groupby("area")[["latitude", "longitude"]].mean()
    for area, (lat, lon) in coords.iterrows():
        G.nodes[node_id(LOCATION, area)].update(latitude=lat, longitude=lon)

    return G


def load_graph(csv_path: str | Path = "complaints.csv") -> nx.DiGraph:
    """Convenience: read the CSV, validate it and build the graph."""
    csv_path = Path(csv_path)
    return build_graph(load_complaints(csv_path), base_dir=csv_path.parent)


# ---------------------------------------------------------------------------
# Querying: what is a complaint connected to?
# ---------------------------------------------------------------------------
def _cnode(G: nx.DiGraph, complaint_id: str) -> str:
    n = node_id(COMPLAINT, complaint_id)
    if n not in G:
        raise KeyError(f"unknown complaint id: {complaint_id}")
    return n


def _targets(G: nx.DiGraph, complaint_id: str, relation: str) -> list[str]:
    n = _cnode(G, complaint_id)
    return [v for _, v, d in G.out_edges(n, data=True) if d["relation"] == relation]


def get_road(G: nx.DiGraph, complaint_id: str) -> dict:
    """{'name': 'Main Road', 'importance': 'Major'} for the affected road."""
    (r,) = _targets(G, complaint_id, AFFECTS_ROAD)
    return {"name": G.nodes[r]["label"], "importance": G.nodes[r]["importance"]}


def get_road_importance(G: nx.DiGraph, complaint_id: str) -> str:
    return get_road(G, complaint_id)["importance"]


def get_department(G: nx.DiGraph, complaint_id: str) -> str:
    (d,) = _targets(G, complaint_id, ASSIGNED_TO)
    return G.nodes[d]["label"]


def get_location(G: nx.DiGraph, complaint_id: str) -> str:
    (loc,) = _targets(G, complaint_id, LOCATED_AT)
    return G.nodes[loc]["label"]


def get_evidence(G: nx.DiGraph, complaint_id: str) -> list[dict]:
    """[{'path': 'images/C101.jpg', 'exists': True}, ...] (empty if no image)."""
    return [
        {"path": G.nodes[i]["path"], "exists": G.nodes[i]["exists"]}
        for i in _targets(G, complaint_id, HAS_EVIDENCE)
    ]


def reasoning_paths(G: nx.DiGraph, complaint_id: str) -> list[str]:
    """One readable line per outgoing relationship, e.g.
    'C101 -[AFFECTS_ROAD]-> Main Road (Major)'."""
    n = _cnode(G, complaint_id)
    lines = []
    for _, v, d in G.out_edges(n, data=True):
        attrs = G.nodes[v]
        target = attrs["label"]
        if attrs["node_type"] == ROAD:
            target += f" ({attrs['importance']})"
        lines.append(f"{complaint_id} -[{d['relation']}]-> {target}")
    order = {LOCATED_AT: 0, AFFECTS_ROAD: 1, ASSIGNED_TO: 2, HAS_EVIDENCE: 3}
    lines.sort(key=lambda s: order[s.split("[")[1].split("]")[0]])
    return lines


def get_relationships(G: nx.DiGraph, complaint_id: str) -> dict:
    """Everything the graph knows about one complaint, as plain data.

    {
      'complaint_id': 'C101',
      'attributes':  {issue_type, description, latitude, longitude, status, created_date},
      'location':    'Andheri',
      'road':        {'name': 'Main Road', 'importance': 'Major'},
      'department':  'Roads',
      'evidence':    [{'path': 'images/C101.jpg', 'exists': True}],
      'edges':       [{'source','relation','target','target_type'}, ...],
      'paths':       ['C101 -[LOCATED_AT]-> Andheri', ...],
    }
    """
    n = _cnode(G, complaint_id)
    node = G.nodes[n]
    edges = [
        {"source": complaint_id, "relation": d["relation"],
         "target": G.nodes[v]["label"], "target_type": G.nodes[v]["node_type"]}
        for _, v, d in G.out_edges(n, data=True)
    ]
    return {
        "complaint_id": complaint_id,
        "attributes": {a: node[a] for a in _COMPLAINT_ATTRS},
        "location": get_location(G, complaint_id),
        "road": get_road(G, complaint_id),
        "department": get_department(G, complaint_id),
        "evidence": get_evidence(G, complaint_id),
        "edges": edges,
        "paths": reasoning_paths(G, complaint_id),
    }


def graph_explanation(G: nx.DiGraph, complaint_id: str) -> str:
    """One human-readable sentence about the graph links, for result cards."""
    rel = get_relationships(G, complaint_id)
    road = rel["road"]
    text = (
        f"{complaint_id} is located at {rel['location']}, affects "
        f"{road['name']} (a {road['importance'].lower()} road) and is assigned "
        f"to the {rel['department']} department"
    )
    if rel["evidence"]:
        text += f"; evidence photo: {rel['evidence'][0]['path']}"
    return text + "."


def related_complaints(
    G: nx.DiGraph,
    complaint_id: str,
    same_issue_only: bool = False,
) -> list[dict]:
    """Other complaints reachable through a shared Road or Location.

    Each result carries the graph path that connects them, so the dashboard can
    show *why* two reports are related (and Member 2 can use it for duplicate
    review):
        {'complaint_id': 'C102', 'issue_type': 'Pothole', 'shared': ['road', 'location'],
         'paths': ['C101 -[AFFECTS_ROAD]-> Main Road <-[AFFECTS_ROAD]- C102', ...]}
    Sorted by number of shared links (most related first), then by id.
    """
    n = _cnode(G, complaint_id)
    issue = G.nodes[n]["issue_type"]
    found: dict[str, dict] = {}

    for rel, kind, hub in (
        (AFFECTS_ROAD, "road", _targets(G, complaint_id, AFFECTS_ROAD)[0]),
        (LOCATED_AT, "location", _targets(G, complaint_id, LOCATED_AT)[0]),
    ):
        hub_label = G.nodes[hub]["label"]
        for other, _, d in G.in_edges(hub, data=True):
            if d["relation"] != rel or other == n:
                continue
            o = G.nodes[other]
            if same_issue_only and o["issue_type"] != issue:
                continue
            entry = found.setdefault(other, {
                "complaint_id": o["label"], "issue_type": o["issue_type"],
                "shared": [], "paths": [],
            })
            entry["shared"].append(kind)
            entry["paths"].append(
                f"{complaint_id} -[{rel}]-> {hub_label} <-[{rel}]- {o['label']}"
            )

    return sorted(found.values(), key=lambda e: (-len(e["shared"]), e["complaint_id"]))


def complaints_on_road(G: nx.DiGraph, road_name: str) -> list[str]:
    return _complaints_into(G, node_id(ROAD, road_name), AFFECTS_ROAD)


def complaints_at_location(G: nx.DiGraph, area: str) -> list[str]:
    return _complaints_into(G, node_id(LOCATION, area), LOCATED_AT)


def complaints_for_department(G: nx.DiGraph, department: str) -> list[str]:
    return _complaints_into(G, node_id(DEPARTMENT, department), ASSIGNED_TO)


def _complaints_into(G: nx.DiGraph, hub: str, relation: str) -> list[str]:
    if hub not in G:
        return []
    return sorted(
        G.nodes[u]["label"] for u, _, d in G.in_edges(hub, data=True)
        if d["relation"] == relation
    )


def roads_in_location(G: nx.DiGraph, area: str) -> list[dict]:
    """Roads that pass through an area, via PASSES_THROUGH."""
    loc = node_id(LOCATION, area)
    if loc not in G:
        return []
    return sorted(
        ({"name": G.nodes[u]["label"], "importance": G.nodes[u]["importance"]}
         for u, _, d in G.in_edges(loc, data=True) if d["relation"] == PASSES_THROUGH),
        key=lambda r: r["name"],
    )


# ---------------------------------------------------------------------------
# Summary, export and drawing
# ---------------------------------------------------------------------------
def graph_summary(G: nx.DiGraph) -> dict:
    """Counts of nodes per type and edges per relation."""
    nodes: dict[str, int] = {}
    for _, d in G.nodes(data=True):
        nodes[d["node_type"]] = nodes.get(d["node_type"], 0) + 1
    edges: dict[str, int] = {}
    for _, _, d in G.edges(data=True):
        edges[d["relation"]] = edges.get(d["relation"], 0) + 1
    return {"nodes": nodes, "edges": edges,
            "total_nodes": G.number_of_nodes(), "total_edges": G.number_of_edges()}


def export_graph_json(G: nx.DiGraph, path: str | Path) -> None:
    """Write the graph as node-link JSON (easy to inspect or load elsewhere)."""
    data = nx.node_link_data(G, edges="edges")
    Path(path).write_text(json.dumps(data, indent=2, default=str))


_TYPE_COLORS = {
    COMPLAINT: "#e4572e", LOCATION: "#2e86ab", ROAD: "#f2a541",
    DEPARTMENT: "#5aa469", IMAGE: "#8d6a9f",
}


def draw_complaint_subgraph(
    G: nx.DiGraph,
    complaint_id: str,
    out_path: str | Path,
    show_related: bool = False,
) -> Path:
    """Save a PNG of a complaint and its direct links (for slides/demo).

    With show_related=True, other complaints on the same road are added.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = _cnode(G, complaint_id)
    sub = nx.DiGraph()
    edge_list = [(n, v) for v in G.successors(n)]
    if show_related:
        road = _targets(G, complaint_id, AFFECTS_ROAD)[0]
        edge_list += [(u, road) for u, _, d in G.in_edges(road, data=True)
                      if d["relation"] == AFFECTS_ROAD and u != n]
    for u, v in edge_list:
        sub.add_node(u, **G.nodes[u])
        sub.add_node(v, **G.nodes[v])
        sub.add_edge(u, v, **G.edges[u, v])

    pos = nx.spring_layout(sub, seed=7, k=1.6)
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = [_TYPE_COLORS[sub.nodes[x]["node_type"]] for x in sub]
    sizes = [1400 if x == n else 900 for x in sub]
    nx.draw_networkx_nodes(sub, pos, node_color=colors, node_size=sizes, ax=ax)
    nx.draw_networkx_edges(sub, pos, arrows=True, arrowsize=14,
                           node_size=sizes, ax=ax, edge_color="#666666")
    nx.draw_networkx_labels(sub, pos, {x: sub.nodes[x]["label"] for x in sub},
                            font_size=8, ax=ax)
    nx.draw_networkx_edge_labels(
        sub, pos, {(u, v): d["relation"] for u, v, d in sub.edges(data=True)},
        font_size=6, ax=ax)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=t)
               for t, c in _TYPE_COLORS.items()]
    ax.legend(handles=handles, loc="lower left", fontsize=8)
    ax.set_title(f"CivicLens KG - {complaint_id} (sample data)")
    ax.axis("off")
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Command line: python knowledge_graph.py [--complaint C101] ...
# ---------------------------------------------------------------------------
def _main() -> None:
    p = argparse.ArgumentParser(description="CivicLens KG knowledge graph demo")
    p.add_argument("--csv", default=str(Path(__file__).with_name("complaints.csv")))
    p.add_argument("--complaint", default="C101", help="complaint id to explain")
    p.add_argument("--draw", metavar="PNG", help="save a diagram for --complaint")
    p.add_argument("--related", action="store_true", help="include related complaints in the diagram")
    p.add_argument("--export", metavar="JSON", help="export the whole graph as JSON")
    args = p.parse_args()

    G = load_graph(args.csv)
    s = graph_summary(G)
    print(f"Graph: {s['total_nodes']} nodes, {s['total_edges']} edges")
    print("  nodes:", s["nodes"])
    print("  edges:", s["edges"])
    warns = G.graph["validation_warnings"]
    if warns:
        print(f"  {len(warns)} warning(s), e.g. {warns[0]}")

    cid = args.complaint
    print(f"\nReasoning paths for {cid}:")
    for line in reasoning_paths(G, cid):
        print("  ", line)
    print("\n" + graph_explanation(G, cid))
    print(f"\nRelated complaints for {cid} (same issue only):")
    for r in related_complaints(G, cid, same_issue_only=True):
        print(f"  {r['complaint_id']} via {', '.join(r['shared'])}")
        for path in r["paths"]:
            print("      ", path)

    if args.draw:
        print("\nSaved diagram:", draw_complaint_subgraph(G, cid, args.draw, args.related))
    if args.export:
        export_graph_json(G, args.export)
        print("Exported graph:", args.export)


if __name__ == "__main__":
    _main()
