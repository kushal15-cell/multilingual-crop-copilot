from __future__ import annotations

import argparse
import csv
import hashlib
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, UnidentifiedImageError

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def image_digest(path: Path) -> str:
    """Hash normalized pixels so renamed exact duplicates stay in the same split."""
    with Image.open(path) as image:
        normalized = image.convert("RGB").resize((64, 64))
        return hashlib.sha256(normalized.tobytes()).hexdigest()


def discover(source: Path) -> list[tuple[Path, str, str]]:
    records: list[tuple[Path, str, str]] = []
    seen: dict[str, str] = {}
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label = path.parent.name
        try:
            digest = image_digest(path)
        except (UnidentifiedImageError, OSError):
            print(f"Skipping unreadable image: {path}")
            continue
        previous_label = seen.get(digest)
        if previous_label is not None and previous_label != label:
            raise ValueError(
                f"Identical image assigned to conflicting classes: {previous_label!r} and {label!r}"
            )
        seen[digest] = label
        records.append((path, label, digest))
    if not records:
        raise ValueError(f"No class-directory images found under {source}")
    return records


def assign_splits(
    records: list[tuple[Path, str, str]], seed: int, train_ratio: float, val_ratio: float
) -> list[tuple[Path, str, str, str]]:
    by_label: dict[str, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
    for path, label, digest in records:
        by_label[label][digest].append(path)
    rng = random.Random(seed)
    assigned: list[tuple[Path, str, str, str]] = []
    for label, groups_map in by_label.items():
        groups = list(groups_map.items())
        rng.shuffle(groups)
        if len(groups) < 3:
            raise ValueError(f"Class {label!r} has fewer than 3 unique images")
        n_groups = len(groups)
        train_end = max(1, round(n_groups * train_ratio))
        val_end = min(n_groups - 1, train_end + max(1, round(n_groups * val_ratio)))
        for index, (digest, paths) in enumerate(groups):
            split = "train" if index < train_end else "val" if index < val_end else "test"
            for path in paths:
                assigned.append((path, label, digest, split))
    return assigned


def prepare(
    source: Path,
    output: Path,
    seed: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> None:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Train, validation, and test ratios must sum to 1")
    if min(train_ratio, val_ratio, test_ratio) <= 0:
        raise ValueError("Every split ratio must be positive")
    records = assign_splits(discover(source), seed, train_ratio, val_ratio)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, str]] = []
    for index, (source_path, label, digest, split) in enumerate(records):
        destination_dir = output / split / label
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / f"{digest[:12]}_{index:05d}{source_path.suffix.lower()}"
        shutil.copy2(source_path, destination)
        manifest_rows.append(
            {
                "source_path": str(source_path),
                "output_path": str(destination),
                "label": label,
                "digest": digest,
                "split": split,
            }
        )
    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_rows[0].keys())
        writer.writeheader()
        writer.writerows(manifest_rows)
    counts = Counter((row["split"], row["label"]) for row in manifest_rows)
    for (split, label), count in sorted(counts.items()):
        print(f"{split:5s} | {label:40s} | {count:5d}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare class-directory PlantDoc images")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    args = parser.parse_args()
    prepare(
        args.source,
        args.output,
        args.seed,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
    )


if __name__ == "__main__":
    main()
