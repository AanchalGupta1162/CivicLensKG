import sys
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import knowledge_graph as kg  # noqa: E402
import prepare_images  # noqa: E402

CSV = ROOT / "complaints.csv"


@pytest.fixture(scope="module")
def G():
    return kg.load_graph(CSV)


@pytest.fixture
def df():
    return kg.load_complaints(CSV)


# --- dataset ---------------------------------------------------------------
def test_dataset_is_valid_and_sized(df):
    report = kg.validate_complaints(df)
    assert report["errors"] == []
    assert len(df) >= 30
    assert set(df["department"]) == kg.VALID_DEPARTMENTS
    assert {"Unresolved", "Resolved"} <= set(df["status"])


# --- graph structure -------------------------------------------------------
def test_c101_demo_relationships(G):
    rel = kg.get_relationships(G, "C101")
    assert rel["location"] == "Andheri"
    assert rel["road"] == {"name": "Main Road", "importance": "Major"}
    assert rel["department"] == "Roads"
    assert rel["evidence"][0]["path"] == "images/C101.jpg"
    assert "C101 -[AFFECTS_ROAD]-> Main Road (Major)" in rel["paths"]
    assert "C101 -[ASSIGNED_TO]-> Roads" in rel["paths"]


def test_every_complaint_has_exactly_one_location_road_department(G):
    for n, d in G.nodes(data=True):
        if d["node_type"] != kg.COMPLAINT:
            continue
        cid = d["label"]
        for rel in (kg.LOCATED_AT, kg.AFFECTS_ROAD, kg.ASSIGNED_TO):
            assert len(kg._targets(G, cid, rel)) == 1


def test_summary_counts(G, df):
    s = kg.graph_summary(G)
    assert s["nodes"]["Complaint"] == len(df)
    assert s["nodes"]["Road"] == df["road_name"].nunique()
    assert s["nodes"]["Location"] == df["area"].nunique()
    assert s["nodes"]["Department"] == 3
    assert s["edges"]["AFFECTS_ROAD"] == len(df)
    assert s["nodes"]["EvidenceImage"] == (df["image_path"] != "").sum()


def test_complaints_without_image_have_no_evidence_node(G):
    assert kg.get_evidence(G, "C122") == []
    assert "evidence" not in kg.graph_explanation(G, "C122")


def test_graph_explanation_mentions_graph_links(G):
    text = kg.graph_explanation(G, "C101")
    assert "Andheri" in text and "Main Road" in text
    assert "major road" in text and "Roads department" in text


# --- traversal -------------------------------------------------------------
def test_related_complaints_finds_c102_with_graph_path(G):
    related = {r["complaint_id"]: r for r in kg.related_complaints(G, "C101", same_issue_only=True)}
    assert "C102" in related
    assert set(related["C102"]["shared"]) == {"road", "location"}
    assert "C101 -[AFFECTS_ROAD]-> Main Road <-[AFFECTS_ROAD]- C102" in related["C102"]["paths"]
    assert "C101" not in related
    assert all(r["issue_type"] == "Pothole" for r in related.values())


def test_related_complaints_ranked_by_shared_links(G):
    shared = [len(r["shared"]) for r in kg.related_complaints(G, "C101")]
    assert shared == sorted(shared, reverse=True)


def test_c104_is_on_a_minor_road_and_c101_on_major(G):
    assert kg.get_road_importance(G, "C101") == "Major"
    assert kg.get_road_importance(G, "C104") == "Minor"


def test_lookup_helpers(G):
    assert "C101" in kg.complaints_on_road(G, "Main Road")
    assert "C101" in kg.complaints_at_location(G, "Andheri")
    assert "C101" in kg.complaints_for_department(G, "Roads")
    assert kg.complaints_on_road(G, "Nowhere Road") == []
    names = {r["name"] for r in kg.roads_in_location(G, "Andheri")}
    assert {"Main Road", "S.V. Road"} <= names


def test_unknown_complaint_raises_keyerror(G):
    with pytest.raises(KeyError):
        kg.get_relationships(G, "C999")


# --- validation failures ---------------------------------------------------
def test_invalid_data_is_rejected(df):
    bad = df.copy()
    bad.loc[0, "status"] = "Pending"
    bad.loc[1, "latitude"] = float("nan")
    bad.loc[2, "complaint_id"] = bad.loc[3, "complaint_id"]
    bad.loc[4, "created_date"] = "10/10/2026"
    errors = kg.validate_complaints(bad)["errors"]
    assert any("invalid status" in e for e in errors)
    assert any("coordinates" in e for e in errors)
    assert any("duplicate complaint_id" in e for e in errors)
    assert any("created_date" in e for e in errors)
    with pytest.raises(ValueError):
        kg.build_graph(bad)


def test_conflicting_road_importance_is_an_error(df):
    bad = df.copy()
    bad.loc[bad["complaint_id"] == "C102", "road_importance"] = "Minor"
    assert any("conflicting" in e for e in kg.validate_complaints(bad)["errors"])


def test_missing_image_file_is_only_a_warning(df, tmp_path):
    report = kg.validate_complaints(df, base_dir=tmp_path)
    assert report["errors"] == []
    assert any("C101" in w for w in report["warnings"])


def test_exists_flag_follows_the_filesystem(df, tmp_path):
    (tmp_path / "images").mkdir()
    Image.new("RGB", (10, 10)).save(tmp_path / "images" / "C101.jpg")
    G2 = kg.build_graph(df, base_dir=tmp_path)
    assert kg.get_evidence(G2, "C101")[0]["exists"] is True
    assert kg.get_evidence(G2, "C102")[0]["exists"] is False


# --- export / drawing ------------------------------------------------------
def test_export_and_draw(G, tmp_path):
    out = tmp_path / "g.json"
    kg.export_graph_json(G, out)
    assert "Complaint:C101" in out.read_text()
    png = kg.draw_complaint_subgraph(G, "C101", tmp_path / "c101.png", show_related=True)
    assert png.stat().st_size > 1000


# --- prepare_images --------------------------------------------------------
def test_prepare_images_links_one_photo_per_complaint(tmp_path):
    src = tmp_path / "kaggle" / "images"
    src.mkdir(parents=True)
    for i in range(1, 101):
        Image.new("RGB", (1600, 1200), (i, 0, 0)).save(src / f"img-{i}.jpg")
    project = tmp_path / "proj"
    project.mkdir()
    (project / "complaints.csv").write_text(CSV.read_text())

    chosen = prepare_images.prepare(tmp_path / "kaggle", project / "complaints.csv")

    expected = (kg.load_complaints(CSV)["image_path"] != "").sum()
    assert len(chosen) == expected
    assert len({s for _, _, s in chosen}) == expected  # no photo reused
    with Image.open(project / "images" / "C101.jpg") as im:
        assert max(im.size) == 800
    assert (project / "images" / "SOURCES.md").is_file()
    assert (project / "images" / "image_manifest.csv").is_file()
    G2 = kg.load_graph(project / "complaints.csv")
    assert all(e["exists"] for e in kg.get_evidence(G2, "C101"))
    assert not G2.graph["validation_warnings"]
