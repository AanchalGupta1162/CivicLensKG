
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import knowledge_graph as kg
from distributed_ir import DistributedIR

CSV = ROOT / "complaints.csv"


def test_distributed_ir_builds_graph():
    ir = DistributedIR(csv_path=str(CSV))

    graph = ir.build_knowledge_graph()

    assert graph is not None
    assert kg.node_id(graph, "C101") is not None


def test_distributed_ir_finds_related_complaints():
    ir = DistributedIR(csv_path=str(CSV))
    ir.build_knowledge_graph()

    related = ir.get_related_complaints("C101")
    related_by_id = {
        item["complaint_id"]: item
        for item in related
    }

    assert "C102" in related_by_id
    assert set(related_by_id["C102"]["shared"]) == {
        "road",
        "location",
    }


def test_distributed_ir_returns_graph_explanation():
    ir = DistributedIR(csv_path=str(CSV))
    ir.build_knowledge_graph()

    explanation = ir.get_graph_explanation("C101")

    assert "Andheri" in explanation
    assert "Main Road" in explanation
    assert "Roads department" in explanation
