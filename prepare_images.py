"""Link downloaded photos to complaints, one photo per complaint.

Three photo categories, matched to complaints by `issue_type`:

    category     issue_type values          suggested source
    pothole      Pothole, Damaged Road      Kaggle andrewmvd/pothole-detection
    garbage      Garbage                    Kaggle hammadarshad18/garbage-detection (piles of garbage)
    streetlight  Streetlight                your own / openly licensed photos of street lamps

Usage (any combination; categories you leave out are not touched):
    python prepare_images.py --pothole-src ~/Downloads/pothole \\
                             --garbage-src ~/Downloads/garbage \\
                             --streetlight-src ~/Downloads/streetlights \\
                             --streetlight-credit "Photos from Wikimedia Commons, CC BY-SA 4.0 (authors in file pages)"

For every complaint of that category the script
  * sets image_path to images/<complaint_id>.jpg if it is blank (complaints.csv is rewritten),
  * copies one photo there, resized to at most 800 px on the long side.
Photos are picked evenly across the source folder and deterministically, so everyone on the team
gets the same C101.jpg. It also updates images/image_manifest.csv (complaint -> original file) and
images/SOURCES.md (attribution; one section per category).

`--src` is kept as an alias for `--pothole-src` (the original single-dataset usage).
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
MAX_SIDE = 800

# category -> (complaint issue types, SOURCES.md heading)
CATEGORIES: dict[str, tuple[set[str], str]] = {
    "pothole": ({"Pothole", "Damaged Road"}, "Potholes and damaged roads"),
    "garbage": ({"Garbage"}, "Garbage"),
    "streetlight": ({"Streetlight"}, "Streetlights"),
}

DEFAULT_CREDITS = {
    "pothole": (
        "Sample of the **Pothole Detection** dataset (665 images) by Larxel (andrewmvd) on Kaggle: "
        "https://www.kaggle.com/datasets/andrewmvd/pothole-detection\n"
        "- Original data: \"Potholes Dataset\", Make ML, https://makeml.app/datasets/potholes\n"
        "- License: Database Contents License (DbCL) v1.0, https://opendatacommons.org/licenses/dbcl/1.0/"
    ),
    "garbage": (
        "Sample of the **Garbage-Detection (Piles of Garbage)** dataset by Hammad Arshad (hammadarshad18) "
        "on Kaggle, photos of garbage piles on streets in Pakistan: "
        "https://www.kaggle.com/datasets/hammadarshad18/garbage-detection\n"
        "- License: CC0 1.0 Public Domain, https://creativecommons.org/publicdomain/zero/1.0/"
    ),
    "streetlight": (
        "Street-lamp photographs collected by the team. **Add the original source and licence of "
        "these photos here** (pass `--streetlight-credit \"...\"` when running prepare_images.py)."
    ),
}

HEADER = (
    "# Image sources\n\n"
    "Photos are resized to at most 800 px; `image_manifest.csv` lists the original file for each "
    "complaint. The *complaints* these photos are attached to are illustrative sample data, and the "
    "photos do not show the places named in complaints.csv.\n"
)


def _natural_key(path: Path) -> list:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.name)]


def find_source_images(src: Path) -> list[Path]:
    files = [p for p in src.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES and p.is_file()]
    return sorted(files, key=_natural_key)


def category_of(issue_type: str) -> str | None:
    for category, (types, _) in CATEGORIES.items():
        if issue_type in types:
            return category
    return None


def _read_csv(csv_path: Path) -> tuple[list[str], list[dict]]:
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames), [dict(r) for r in reader]


def _write_csv(csv_path: Path, fields: list[str], rows: list[dict]) -> None:
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def _update_sources_md(path: Path, categories_present: list[str], credits: dict[str, str]) -> None:
    """Rewrite SOURCES.md: one section per category, keeping existing credit text unless a new one is given."""
    existing: dict[str, str] = {}
    if path.is_file():
        for block in re.split(r"(?m)^## ", path.read_text())[1:]:
            heading, _, body = block.partition("\n")
            existing[heading.strip()] = body.strip()
    out = [HEADER]
    for category in CATEGORIES:
        if category not in categories_present:
            continue
        heading = CATEGORIES[category][1]
        body = credits.get(category) or existing.get(heading) or DEFAULT_CREDITS[category]
        out.append(f"## {heading}\n\n{body}\n")
    path.write_text("\n".join(out))


def prepare(
    sources: dict[str, Path],
    csv_path: Path,
    credits: dict[str, str] | None = None,
) -> list[tuple[str, str, Path]]:
    """Copy photos for the given categories. Returns (complaint_id, image_path, source_file)."""
    credits = credits or {}
    unknown = set(sources) - set(CATEGORIES)
    if unknown:
        raise SystemExit(f"unknown categories: {sorted(unknown)}")

    fields, rows = _read_csv(csv_path)
    base = csv_path.parent

    # Plan everything first so a short source folder fails before any file is written.
    plan: list[tuple[dict, Path]] = []
    for category, src in sources.items():
        targets = [r for r in rows if category_of(r["issue_type"].strip()) == category]
        images = find_source_images(src)
        if len(images) < len(targets):
            raise SystemExit(
                f"[{category}] found {len(images)} images under {src}, "
                f"but {len(targets)} complaints need one."
            )
        step = len(images) / len(targets)
        plan += [(row, images[int(i * step)]) for i, row in enumerate(targets)]

    chosen = []
    for row, source in plan:
        if not row["image_path"].strip():
            row["image_path"] = f"images/{row['complaint_id'].strip()}.jpg"
        dest = base / row["image_path"].strip()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source) as im:
            im = im.convert("RGB")
            im.thumbnail((MAX_SIDE, MAX_SIDE))
            im.save(dest, "JPEG", quality=88)
        chosen.append((row["complaint_id"].strip(), row["image_path"].strip(), source))
    _write_csv(csv_path, fields, rows)

    # Manifest: keep earlier categories' rows, replace the ones just regenerated.
    images_dir = base / "images"
    images_dir.mkdir(exist_ok=True)
    manifest_path = images_dir / "image_manifest.csv"
    manifest: dict[str, tuple[str, str]] = {}
    if manifest_path.is_file():
        with open(manifest_path, newline="") as f:
            manifest = {r["complaint_id"]: (r["image_path"], r["source_file"]) for r in csv.DictReader(f)}
    manifest.update({cid: (ip, s.name) for cid, ip, s in chosen})
    with open(manifest_path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["complaint_id", "image_path", "source_file"])
        w.writerows((cid, ip, sf) for cid, (ip, sf) in sorted(manifest.items()))

    present = {category_of(r["issue_type"].strip()) for r in rows if r["complaint_id"].strip() in manifest}
    _update_sources_md(images_dir / "SOURCES.md", [c for c in CATEGORIES if c in present], credits)
    return chosen


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--pothole-src", "--src", dest="pothole", type=Path, help="folder with pothole photos")
    p.add_argument("--garbage-src", dest="garbage", type=Path, help="folder with garbage photos")
    p.add_argument("--streetlight-src", dest="streetlight", type=Path, help="folder with street-lamp photos")
    p.add_argument("--garbage-credit", help="attribution text for SOURCES.md (replaces the default)")
    p.add_argument("--streetlight-credit", help="attribution text for SOURCES.md")
    p.add_argument("--pothole-credit", help="attribution text for SOURCES.md (replaces the default)")
    p.add_argument("--csv", type=Path, default=Path(__file__).with_name("complaints.csv"))
    args = p.parse_args()

    sources = {c: getattr(args, c) for c in CATEGORIES if getattr(args, c)}
    if not sources:
        p.error("give at least one of --pothole-src, --garbage-src, --streetlight-src")
    for c, src in sources.items():
        if not src.is_dir():
            p.error(f"--{c}-src is not a folder: {src}")
    credits = {c: getattr(args, f"{c}_credit") for c in CATEGORIES if getattr(args, f"{c}_credit")}

    chosen = prepare(sources, args.csv, credits)
    print(f"Linked {len(chosen)} photos ({', '.join(sources)}), e.g. {chosen[0][1]} <- {chosen[0][2].name}")
    print("Updated complaints.csv, images/image_manifest.csv and images/SOURCES.md")


if __name__ == "__main__":
    main()
