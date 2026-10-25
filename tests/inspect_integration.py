
import sys
import inspect
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from distributed_ir import DistributedIR

for name in (
    "build_knowledge_graph",
    "get_related_complaints",
    "get_graph_explanation",
    "build_image_search_index",
    "search_similar_images",
):
    method = getattr(DistributedIR, name, None)

    print(f"\n{name}:")
    if method is None:
        print("  NOT FOUND")
    else:
        print(" ", inspect.signature(method))
