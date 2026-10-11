"""CivicLens KG - Member 4: Multimedia (image similarity) module.

Indexes the complaint photos listed in complaints.csv and finds the most similar
ones for a query image. Two backends, chosen automatically:

    "clip"         CLIP image embeddings (visual-semantic). Needs torch + transformers.
    "handcrafted"  Colour histogram + grey thumbnail + edge-orientation features
                   (Pillow + numpy only). It matches *appearance* (colour, layout,
                   texture), NOT meaning, so label it that way in the UI.

Every result carries the backend and a human-readable `backend_label`, so the
dashboard never presents the fallback as visual-semantic search.

Photos are generic civic-issue examples, not the places named in the CSV
(see images/SOURCES.md). All complaint data is illustrative sample data.

Quick use
    from multimedia import build_image_index, search_similar
    index = build_image_index("complaints.csv")
    search_similar(index, "images/C101.jpg", k=3, exclude_ids=["C101"])
"""
from __future__ import annotations

import argparse
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Union

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

ImageInput = Union[str, Path, bytes, Image.Image, "io.IOBase"]

CLIP_MODEL = "openai/clip-vit-base-patch32"
BACKENDS = ("clip", "handcrafted")
BACKEND_LABELS = {
    "clip": "CLIP image embeddings (visual-semantic similarity)",
    "handcrafted": "colour/texture/edge features (appearance match only, not semantic search)",
}

_clip_cache: dict = {}


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------
def load_image(source: ImageInput) -> Image.Image:
    """Open a path, bytes, file-like object or PIL image as an upright RGB image.

    Raises ValueError if the data is not a readable image.
    """
    try:
        if isinstance(source, Image.Image):
            image = source
        elif isinstance(source, (bytes, bytearray)):
            image = Image.open(io.BytesIO(bytes(source)))
        elif isinstance(source, (str, Path)):
            image = Image.open(source)
        else:  # file-like, e.g. a Streamlit UploadedFile
            image = Image.open(source)
        image = ImageOps.exif_transpose(image)
        return image.convert("RGB")
    except (OSError, ValueError, TypeError, Image.DecompressionBombError) as error:
        raise ValueError(f"could not read image: {error}") from None


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
def _unit(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


def handcrafted_embedding(image: Image.Image) -> np.ndarray:
    """Return a unit vector of colour, layout and edge-direction features.

    Blocks: 4x4x4 joint RGB histogram (64), 16x16 grey thumbnail with the mean
    removed (256), and a 12-bin gradient-orientation histogram weighted by
    magnitude (12). Blocks are L2-normalised, weighted, concatenated and
    normalised again, so cosine similarity lies in [-1, 1].
    """
    rgb = np.asarray(image.resize((64, 64), Image.BILINEAR), dtype=np.float32)

    bins = (rgb // 64).astype(int).reshape(-1, 3)
    hist = np.bincount(bins[:, 0] * 16 + bins[:, 1] * 4 + bins[:, 2], minlength=64)
    hist = _unit(np.sqrt(hist.astype(np.float32)))  # sqrt damps dominant colours

    grey = np.asarray(image.convert("L").resize((16, 16), Image.BILINEAR), dtype=np.float32)
    layout = _unit((grey - grey.mean()).ravel())

    grey64 = np.asarray(image.convert("L").resize((64, 64), Image.BILINEAR), dtype=np.float32)
    gy, gx = np.gradient(grey64)
    magnitude = np.hypot(gx, gy).ravel()
    angle = (np.arctan2(gy, gx).ravel() % np.pi)  # orientation, direction ignored
    edges = np.bincount(np.minimum((angle / np.pi * 12).astype(int), 11),
                        weights=magnitude, minlength=12).astype(np.float32)
    edges = _unit(edges)

    return _unit(np.concatenate([1.0 * hist, 1.0 * layout, 0.7 * edges]))


def clip_available() -> bool:
    """True if torch and transformers can be imported (the model may still need a download)."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except Exception:
        return False
    return True


def _clip_embedding(image: Image.Image) -> np.ndarray:
    if "model" not in _clip_cache:
        from transformers import CLIPModel, CLIPProcessor
        model = CLIPModel.from_pretrained(CLIP_MODEL).eval()
        processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
        _clip_cache.update(model=model, processor=processor)
    import torch
    inputs = _clip_cache["processor"](images=image, return_tensors="pt")
    with torch.no_grad():
        features = _clip_cache["model"].get_image_features(**inputs)
    if not isinstance(features, torch.Tensor):  # transformers >= 5 returns an output object
        features = features.pooler_output       # the projected 512-d CLIP embedding
    return _unit(features[0].numpy().astype(np.float32))


def embed_image(image: Image.Image, backend: str) -> np.ndarray:
    """Embed one RGB image with the named backend."""
    if backend == "handcrafted":
        return handcrafted_embedding(image)
    if backend == "clip":
        return _clip_embedding(image)
    raise ValueError(f"unknown backend '{backend}'; choose from {BACKENDS}")


def _resolve_backend(backend: str) -> str:
    if backend == "auto":
        return "clip" if clip_available() else "handcrafted"
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend '{backend}'; choose from 'auto' or {BACKENDS}")
    if backend == "clip" and not clip_available():
        raise ImportError("backend 'clip' needs torch and transformers; "
                          "pip install torch transformers, or use backend='handcrafted'")
    return backend


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------
@dataclass
class ImageIndex:
    """Normalised embeddings aligned with complaint ids and image paths."""
    backend: str
    complaint_ids: list[str]
    image_paths: list[str]
    embeddings: np.ndarray  # shape (n, d), unit rows
    warnings: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.complaint_ids)

    @property
    def backend_label(self) -> str:
        return BACKEND_LABELS[self.backend]


def build_image_index(
    csv_path: str | Path = "complaints.csv", backend: str = "auto",
    base_dir: str | Path | None = None,
) -> ImageIndex:
    """Embed every complaint photo listed in the CSV.

    image_path is resolved relative to `base_dir` (default: the CSV's folder).
    Rows with a blank path are skipped silently; missing or unreadable files
    are skipped with a warning in
    `index.warnings`. Raises ValueError if the CSV lacks the required columns
    or has duplicate complaint ids.
    In auto mode, a CLIP embedding failure rebuilds the entire index with
    handcrafted features and records a warning. Explicit clip mode raises.
    """
    csv_path = Path(csv_path)
    base = Path(base_dir) if base_dir is not None else csv_path.parent
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    missing = [c for c in ("complaint_id", "image_path") if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    df = df.apply(lambda col: col.str.strip())
    if df["complaint_id"].duplicated().any():
        raise ValueError("duplicate complaint_id values in CSV")

    requested_backend = backend
    backend = _resolve_backend(backend)
    ids: list[str] = []
    paths: list[str] = []
    vectors: list[np.ndarray] = []
    warnings: list[str] = []
    for cid, rel in zip(df["complaint_id"], df["image_path"]):
        if not rel:
            continue
        try:
            image = load_image(base / rel)
        except ValueError as error:
            warnings.append(f"{cid}: skipped {rel} ({error})")
            continue
        try:
            vector = embed_image(image, backend)
        except Exception as error:
            if requested_backend != "auto" or backend != "clip":
                raise
            fallback = build_image_index(csv_path, backend="handcrafted", base_dir=base)
            fallback.warnings.insert(
                0, f"CLIP embedding failed ({error}); using handcrafted fallback for all images.",
            )
            return fallback
        ids.append(cid)
        paths.append(rel)
        vectors.append(vector)

    matrix = np.vstack(vectors).astype(np.float32) if vectors else np.zeros((0, 0), np.float32)
    return ImageIndex(backend, ids, paths, matrix, warnings)


def search_similar(
    index: ImageIndex, query: ImageInput, k: int = 3,
    exclude_ids: Iterable[str] | None = None,
) -> list[dict]:
    """Return the top-k most similar indexed photos, best first.

    Each result: {rank, complaint_id, image_path, score, backend, backend_label}
    where score is cosine similarity (1.0 = identical embedding). The query is
    embedded with the index's own backend. `exclude_ids` drops complaints, e.g.
    the query's own photo. Ties break by complaint id. Raises ValueError for
    k < 1 or an unreadable query image; an empty index returns [].
    """
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    image = load_image(query)
    if len(index) == 0:
        return []
    vector = embed_image(image, index.backend)
    scores = index.embeddings @ vector
    skip = set(exclude_ids or ())
    order = sorted(
        (i for i, cid in enumerate(index.complaint_ids) if cid not in skip),
        key=lambda i: (-float(scores[i]), index.complaint_ids[i]),
    )[:k]
    return [
        {"rank": rank, "complaint_id": index.complaint_ids[i],
         "image_path": index.image_paths[i], "score": round(float(scores[i]), 4),
         "backend": index.backend, "backend_label": index.backend_label}
        for rank, i in enumerate(order, start=1)
    ]


def index_stats(index: ImageIndex) -> dict:
    """Plain-data summary for the dashboard or a README."""
    return {"backend": index.backend, "backend_label": index.backend_label,
            "images_indexed": len(index),
            "embedding_dim": int(index.embeddings.shape[1]) if len(index) else 0,
            "warnings": list(index.warnings)}


def save_index(index: ImageIndex, path: str | Path) -> None:
    """Write the index to a .npz file (no pickle)."""
    np.savez_compressed(
        path, embeddings=index.embeddings,
        meta=np.array(json.dumps({
            "backend": index.backend, "complaint_ids": index.complaint_ids,
            "image_paths": index.image_paths, "warnings": index.warnings,
        })),
    )


def load_index(path: str | Path) -> ImageIndex:
    """Read an index written by save_index. Raises ValueError if it is malformed."""
    try:
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            embeddings = data["embeddings"]
        backend = meta["backend"]
        ids, paths = meta["complaint_ids"], meta["image_paths"]
    except (OSError, KeyError, ValueError, TypeError) as error:
        raise ValueError(f"not a valid image index: {error}") from None
    if backend not in BACKENDS or not (len(ids) == len(paths) == len(embeddings)):
        raise ValueError("not a valid image index: inconsistent contents")
    return ImageIndex(backend, ids, paths, embeddings.astype(np.float32), meta.get("warnings", []))


# ---------------------------------------------------------------------------
# Command line: python multimedia.py --query images/C101.jpg
# ---------------------------------------------------------------------------
def _main() -> None:
    p = argparse.ArgumentParser(description="CivicLens KG image similarity demo")
    p.add_argument("--csv", default=str(Path(__file__).with_name("complaints.csv")))
    p.add_argument("--query", required=True, help="path of the query image")
    p.add_argument("-k", type=int, default=3)
    p.add_argument("--backend", default="auto", choices=("auto",) + BACKENDS)
    p.add_argument("--keep-self", action="store_true",
                   help="do not drop the query's own complaint from the results")
    args = p.parse_args()

    index = build_image_index(args.csv, args.backend)
    stats = index_stats(index)
    print(f"Indexed {stats['images_indexed']} images with: {stats['backend_label']}")
    for warning in stats["warnings"]:
        print("  warning:", warning)
    exclude = None if args.keep_self else [Path(args.query).stem]
    print(f"\nTop {args.k} matches for {args.query}:")
    for r in search_similar(index, args.query, args.k, exclude):
        print(f"  {r['rank']}. {r['complaint_id']}  score={r['score']:.3f}  {r['image_path']}")


if __name__ == "__main__":
    _main()
