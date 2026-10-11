"""Deterministic image-similarity tests; the CLIP backend is not exercised."""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import multimedia  # noqa: E402


def _make_image(kind: str, size: int = 96) -> Image.Image:
    """Distinct synthetic pictures: grey road with dark blob, red, blue stripes."""
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    if kind == "road":
        arr[:] = (120, 120, 120)
        arr[30:60, 30:70] = (30, 30, 30)
    elif kind == "road2":  # slightly shifted/noisier version of "road"
        arr[:] = (125, 122, 118)
        arr[32:62, 28:68] = (35, 32, 30)
    elif kind == "red":
        arr[:] = (200, 20, 20)
    elif kind == "stripes":
        arr[:] = (20, 20, 200)
        arr[::8] = (240, 240, 240)
    return Image.fromarray(arr)


@pytest.fixture
def dataset(tmp_path):
    (tmp_path / "images").mkdir()
    rows = [("C1", "road"), ("C2", "road2"), ("C3", "red"), ("C4", "stripes")]
    lines = ["complaint_id,image_path"]
    for cid, kind in rows:
        _make_image(kind).save(tmp_path / "images" / f"{cid}.png")
        lines.append(f"{cid},images/{cid}.png")
    lines.append("C5,")                       # no photo: skipped silently
    lines.append("C6,images/missing.png")     # file absent: warning
    (tmp_path / "complaints.csv").write_text("\n".join(lines) + "\n")
    return tmp_path


@pytest.fixture
def index(dataset):
    return multimedia.build_image_index(dataset / "complaints.csv", backend="handcrafted")


def test_index_contents_and_warnings(index):
    assert index.backend == "handcrafted"
    assert index.complaint_ids == ["C1", "C2", "C3", "C4"]
    assert index.embeddings.shape[0] == 4
    assert np.allclose(np.linalg.norm(index.embeddings, axis=1), 1.0, atol=1e-5)
    assert len(index.warnings) == 1 and "C6" in index.warnings[0]
    stats = multimedia.index_stats(index)
    assert stats["images_indexed"] == 4 and "not semantic" in stats["backend_label"]


def test_identical_image_ranks_first_with_unit_score(index, dataset):
    results = multimedia.search_similar(index, dataset / "images" / "C3.png", k=3)
    assert results[0]["complaint_id"] == "C3"
    assert results[0]["score"] == pytest.approx(1.0, abs=1e-3)
    assert [r["rank"] for r in results] == [1, 2, 3]
    assert all(r["backend"] == "handcrafted" for r in results)


def test_similar_pictures_outrank_different_ones(index, dataset):
    results = multimedia.search_similar(
        index, dataset / "images" / "C1.png", k=3, exclude_ids=["C1"])
    assert [r["complaint_id"] for r in results][0] == "C2"
    assert "C1" not in {r["complaint_id"] for r in results}
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_query_input_types(index, dataset):
    path = dataset / "images" / "C4.png"
    raw = path.read_bytes()
    for query in (str(path), path, raw, Image.open(path), io.BytesIO(raw)):
        assert multimedia.search_similar(index, query, k=1)[0]["complaint_id"] == "C4"


def test_k_larger_than_index_and_invalid_k(index, dataset):
    path = dataset / "images" / "C1.png"
    assert len(multimedia.search_similar(index, path, k=50)) == 4
    for bad in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            multimedia.search_similar(index, path, k=bad)


def test_unreadable_query_raises(index):
    with pytest.raises(ValueError):
        multimedia.search_similar(index, b"not an image")
    with pytest.raises(ValueError):
        multimedia.search_similar(index, "no/such/file.png")


def test_corrupt_indexed_file_becomes_warning(dataset):
    (dataset / "images" / "C4.png").write_bytes(b"garbage")
    built = multimedia.build_image_index(dataset / "complaints.csv", backend="handcrafted")
    assert "C4" not in built.complaint_ids
    assert any("C4" in w for w in built.warnings)


def test_empty_index_returns_no_results(tmp_path):
    (tmp_path / "complaints.csv").write_text("complaint_id,image_path\nC1,\n")
    empty = multimedia.build_image_index(tmp_path / "complaints.csv", backend="handcrafted")
    assert len(empty) == 0
    assert multimedia.search_similar(empty, _make_image("red")) == []


def test_csv_validation(tmp_path):
    (tmp_path / "a.csv").write_text("complaint_id,x\nC1,1\n")
    with pytest.raises(ValueError, match="Missing required columns"):
        multimedia.build_image_index(tmp_path / "a.csv", backend="handcrafted")
    (tmp_path / "b.csv").write_text("complaint_id,image_path\nC1,\nC1,\n")
    with pytest.raises(ValueError, match="duplicate"):
        multimedia.build_image_index(tmp_path / "b.csv", backend="handcrafted")


def test_save_and_load_round_trip(index, dataset, tmp_path):
    target = tmp_path / "index.npz"
    multimedia.save_index(index, target)
    loaded = multimedia.load_index(target)
    assert loaded.complaint_ids == index.complaint_ids
    assert loaded.backend == "handcrafted"
    query = dataset / "images" / "C2.png"
    assert (multimedia.search_similar(loaded, query, k=4)
            == multimedia.search_similar(index, query, k=4))
    target.write_bytes(b"junk")
    with pytest.raises(ValueError):
        multimedia.load_index(target)


def test_backend_selection():
    with pytest.raises(ValueError):
        multimedia.embed_image(_make_image("red"), "nonsense")
    assert multimedia._resolve_backend("handcrafted") == "handcrafted"
    auto = multimedia._resolve_backend("auto")
    assert auto in multimedia.BACKENDS
    if not multimedia.clip_available():
        assert auto == "handcrafted"
        with pytest.raises(ImportError):
            multimedia._resolve_backend("clip")


def test_embedding_is_deterministic_and_handles_odd_modes():
    a = multimedia.handcrafted_embedding(_make_image("road"))
    b = multimedia.handcrafted_embedding(_make_image("road"))
    assert np.array_equal(a, b)
    flat = multimedia.handcrafted_embedding(Image.new("RGB", (10, 10), "black"))
    assert np.isfinite(flat).all()
    rgba = Image.new("RGBA", (20, 30), (10, 200, 10, 128))
    assert multimedia.load_image(rgba).mode == "RGB"


@pytest.mark.skipif(not (ROOT / "images" / "C101.jpg").is_file(), reason="sample photos absent")
def test_real_sample_photos_find_themselves():
    built = multimedia.build_image_index(ROOT / "complaints.csv", backend="handcrafted")
    assert len(built) == 80 and not built.warnings
    for cid in ("C101", "C125", "C135", "C180"):  # Pothole, Garbage, Streetlight, Damaged Road
        top = multimedia.search_similar(built, ROOT / "images" / f"{cid}.jpg", k=1)[0]
        assert top["complaint_id"] == cid


@pytest.mark.skipif(not multimedia.clip_available(), reason="torch/transformers not installed")
def test_clip_backend_end_to_end(dataset):
    """Needs the CLIP weights (cached after the first download)."""
    built = multimedia.build_image_index(dataset / "complaints.csv", backend="clip")
    assert built.backend == "clip" and "visual-semantic" in built.backend_label
    assert built.embeddings.shape == (4, 512)
    assert np.allclose(np.linalg.norm(built.embeddings, axis=1), 1.0, atol=1e-4)
    top = multimedia.search_similar(built, dataset / "images" / "C3.png", k=1)[0]
    assert top["complaint_id"] == "C3" and top["backend"] == "clip"
    assert top["score"] == pytest.approx(1.0, abs=1e-3)
