"""Link Kaggle pothole photos to complaints.

Usage (after downloading https://www.kaggle.com/datasets/andrewmvd/pothole-detection):
    python prepare_images.py --src /path/to/unzipped/pothole-detection

For every complaint in complaints.csv that has an image_path (e.g. images/C101.jpg),
one Kaggle photo is copied to that path, resized to at most 800 px on the long side.
Photos are picked evenly across the dataset so the gallery is varied, and the choice is
deterministic, so everyone on the team gets the same C101.jpg.

Also writes images/image_manifest.csv (complaint -> original Kaggle file) and
images/SOURCES.md (attribution).
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
MAX_SIDE = 800

ATTRIBUTION = """# Image sources

Pothole photographs in this folder are a sample of the **Pothole Detection** dataset
(665 images, PASCAL VOC annotations) by Larxel (andrewmvd) on Kaggle:
https://www.kaggle.com/datasets/andrewmvd/pothole-detection

- Original data: "Potholes Dataset", Make ML, https://makeml.app/datasets/potholes
- License: Database Contents License (DbCL) v1.0, https://opendatacommons.org/licenses/dbcl/1.0/
- Images were resized to at most 800 px. See image_manifest.csv for the original file names.

The *complaints* these photos are attached to are illustrative sample data. The photos do
not show the places named in complaints.csv.
"""


def _natural_key(path: Path) -> list:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.name)]


def find_source_images(src: Path) -> list[Path]:
    files = [p for p in src.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES and p.is_file()]
    return sorted(files, key=_natural_key)


def prepare(src: Path, csv_path: Path) -> list[tuple[str, str, Path]]:
    """Copy photos for complaints with an image_path. Returns (complaint_id, image_path, source)."""
    with open(csv_path, newline="") as f:
        wanted = [(r["complaint_id"].strip(), r["image_path"].strip())
                  for r in csv.DictReader(f) if r["image_path"].strip()]
    sources = find_source_images(src)
    if len(sources) < len(wanted):
        raise SystemExit(
            f"Found {len(sources)} images under {src}, but {len(wanted)} complaints need one."
        )

    base = csv_path.parent
    step = len(sources) / len(wanted)
    chosen = []
    for i, (cid, image_path) in enumerate(wanted):
        source = sources[int(i * step)]
        dest = base / image_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source) as im:
            im = im.convert("RGB")
            im.thumbnail((MAX_SIDE, MAX_SIDE))
            im.save(dest, "JPEG", quality=88)
        chosen.append((cid, image_path, source))

    images_dir = base / "images"
    images_dir.mkdir(exist_ok=True)
    with open(images_dir / "image_manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["complaint_id", "image_path", "source_file"])
        w.writerows((cid, ip, s.name) for cid, ip, s in chosen)
    (images_dir / "SOURCES.md").write_text(ATTRIBUTION)
    return chosen


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--src", required=True, type=Path, help="folder with the unzipped Kaggle dataset")
    p.add_argument("--csv", type=Path, default=Path(__file__).with_name("complaints.csv"))
    args = p.parse_args()
    if not args.src.is_dir():
        raise SystemExit(f"--src is not a folder: {args.src}")
    chosen = prepare(args.src, args.csv)
    print(f"Linked {len(chosen)} photos, e.g. {chosen[0][1]} <- {chosen[0][2].name}")
    print("Wrote images/image_manifest.csv and images/SOURCES.md")


if __name__ == "__main__":
    main()
