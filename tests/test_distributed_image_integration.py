
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from distributed_ir import DistributedIR

CSV = ROOT / "complaints.csv"


def test_distributed_ir_image_search():
    ir = DistributedIR(csv_path=str(CSV))

    ir.build_image_search_index(backend="handcrafted")

    results = ir.search_similar_images(
        ROOT / "images" / "C101.jpg",
        k=3,
        exclude_ids=["C101"],
    )

    assert isinstance(results, list)
    assert len(results) > 0
    assert all("complaint_id" in item for item in results)
    assert all("score" in item for item in results)
    assert "C101" not in {
        item["complaint_id"] for item in results
    }
