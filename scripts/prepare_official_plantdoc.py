from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, UnidentifiedImageError

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def resolve_dataset_root(source: Path) -> Path:
    """Resolve either the repository root or one accidental wrapper directory."""
    direct_train = source / "train"
    direct_test = source / "test"
    if direct_train.is_dir() and direct_test.is_dir():
        return source

    candidates = sorted(
        child
        for child in source.iterdir()
        if child.is_dir()
        and not child.name.startswith(".")
        and (child / "train").is_dir()
        and (child / "test").is_dir()
    ) if source.is_dir() else []
    if len(candidates) == 1:
        print(f"Detected PlantDoc dataset inside wrapper directory: {candidates[0]}")
        return candidates[0]
    if len(candidates) > 1:
        rendered = ", ".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(
            "Multiple directories contain PlantDoc train/ and test/ folders; "
            f"pass the intended directory with --source. Candidates: {rendered}"
        )

    visible_entries = []
    if source.is_dir():
        visible_entries = sorted(
            path.name for path in source.iterdir() if not path.name.startswith(".")
        )
    listing = ", ".join(visible_entries[:20]) or "<empty or missing>"
    raise FileNotFoundError(
        "Expected official PlantDoc train/ and test/ directories beneath "
        f"{source}. Visible entries: {listing}. If this is a Git checkout, restore "
        "the tracked dataset folders or rerun scripts/train_real_plantdoc.ps1."
    )


def normalized_pixel_digest(path: Path) -> str:
    with Image.open(path) as image:
        normalized = image.convert("RGB").resize((64, 64))
        return hashlib.sha256(normalized.tobytes()).hexdigest()


def discover(split_root: Path, label_pattern: re.Pattern[str]) -> list[tuple[Path, str, str]]:
    records: list[tuple[Path, str, str]] = []
    for class_dir in sorted(path for path in split_root.iterdir() if path.is_dir()):
        if not label_pattern.search(class_dir.name):
            continue
        for path in sorted(class_dir.rglob("*")):
            if not path.is_file() or path.suffix.casefold() not in IMAGE_EXTENSIONS:
                continue
            try:
                digest = normalized_pixel_digest(path)
            except (UnidentifiedImageError, OSError):
                print(f"Skipping unreadable image: {path}")
                continue
            records.append((path, class_dir.name, digest))
    return records


def copy_record(
    source_path: Path,
    label: str,
    digest: str,
    split: str,
    output: Path,
    index: int,
    origin: str,
) -> dict[str, str]:
    destination_dir = output / split / label
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"{digest[:12]}_{index:05d}{source_path.suffix.casefold()}"
    shutil.copy2(source_path, destination)
    return {
        "source_path": str(source_path),
        "output_path": str(destination),
        "label": label,
        "digest": digest,
        "origin": origin,
        "split": split,
    }


def prepare(
    source: Path,
    output: Path,
    label_regex: str = "tomato",
    val_ratio: float = 0.15,
    seed: int = 42,
) -> dict:
    if not 0 < val_ratio < 0.5:
        raise ValueError("Validation ratio must be between 0 and 0.5")
    dataset_root = resolve_dataset_root(source)
    official_train = dataset_root / "train"
    official_test = dataset_root / "test"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")

    pattern = re.compile(label_regex, flags=re.IGNORECASE)
    train_records = discover(official_train, pattern)
    test_records = discover(official_test, pattern)
    if not train_records or not test_records:
        raise ValueError(f"No matching images found for label regex: {label_regex!r}")

    train_labels = {label for _, label, _ in train_records}
    test_labels = {label for _, label, _ in test_records}
    selected_labels = train_labels & test_labels
    dropped_train_only_labels = sorted(train_labels - test_labels)
    dropped_test_only_labels = sorted(test_labels - train_labels)
    if not selected_labels:
        raise ValueError("No classes occur in both the official train and test splits")
    train_records = [record for record in train_records if record[1] in selected_labels]
    test_records = [record for record in test_records if record[1] in selected_labels]

    digest_labels: dict[str, set[str]] = defaultdict(set)
    for _, label, digest in train_records + test_records:
        digest_labels[digest].add(label)
    conflicting_digests = {
        digest: sorted(labels)
        for digest, labels in digest_labels.items()
        if len(labels) > 1
    }
    conflicting_details = []
    for digest, labels in sorted(conflicting_digests.items()):
        matching_train = [record for record in train_records if record[2] == digest]
        matching_test = [record for record in test_records if record[2] == digest]
        conflicting_details.append(
            {
                "digest": digest,
                "labels": labels,
                "train_images_removed": len(matching_train),
                "test_images_removed": len(matching_test),
            }
        )
        print(
            "Quarantining identical pixels with conflicting labels: "
            f"{labels} ({len(matching_train)} train, {len(matching_test)} test)"
        )
    conflicting_train_images = sum(
        digest in conflicting_digests for _, _, digest in train_records
    )
    conflicting_test_images = sum(
        digest in conflicting_digests for _, _, digest in test_records
    )
    train_records = [
        record for record in train_records if record[2] not in conflicting_digests
    ]
    test_records = [
        record for record in test_records if record[2] not in conflicting_digests
    ]

    test_digests = {digest for _, _, digest in test_records}
    removed_train_test_duplicates = sum(
        digest in test_digests for _, _, digest in train_records
    )
    train_records = [record for record in train_records if record[2] not in test_digests]

    grouped: dict[str, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
    for path, label, digest in train_records:
        grouped[label][digest].append(path)

    rng = random.Random(seed)
    assignments: list[tuple[Path, str, str, str, str]] = []
    for label, digest_to_paths in grouped.items():
        groups = list(digest_to_paths.items())
        rng.shuffle(groups)
        if len(groups) < 3:
            raise ValueError(f"Class {label!r} has fewer than three unique training images")
        validation_groups = max(1, round(len(groups) * val_ratio))
        validation_groups = min(validation_groups, len(groups) - 1)
        for group_index, (digest, paths) in enumerate(groups):
            split = "val" if group_index < validation_groups else "train"
            for path in paths:
                assignments.append((path, label, digest, split, "official_train"))
    assignments.extend(
        (path, label, digest, "test", "official_test")
        for path, label, digest in test_records
    )

    output.mkdir(parents=True, exist_ok=True)
    manifest = [
        copy_record(path, label, digest, split, output, index, origin)
        for index, (path, label, digest, split, origin) in enumerate(assignments)
    ]
    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest[0].keys())
        writer.writeheader()
        writer.writerows(manifest)

    license_source = dataset_root / "LICENSE.txt"
    if license_source.exists():
        shutil.copy2(license_source, output / "LICENSE_PLANTDOC_CC_BY_4.0.txt")

    counts = Counter((row["split"], row["label"]) for row in manifest)
    summary = {
        "source": "https://github.com/pratikkayal/PlantDoc-Dataset",
        "scope_regex": label_regex,
        "seed": seed,
        "validation_ratio_from_official_train": val_ratio,
        "official_test_preserved": True,
        "classes": sorted(selected_labels),
        "class_count": len(selected_labels),
        "dropped_train_only_labels": dropped_train_only_labels,
        "dropped_test_only_labels": dropped_test_only_labels,
        "image_count": len(manifest),
        "conflicting_duplicate_groups_removed": len(conflicting_digests),
        "conflicting_duplicate_images_removed": {
            "train": conflicting_train_images,
            "test": conflicting_test_images,
        },
        "conflicting_duplicate_details": conflicting_details,
        "removed_exact_train_test_duplicates": removed_train_test_duplicates,
        "counts": {
            f"{split}/{label}": count for (split, label), count in sorted(counts.items())
        },
    }
    (output / "preparation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare an official-test-preserving PlantDoc classification dataset"
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label-regex", default="tomato")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(args.source, args.output, args.label_regex, args.val_ratio, args.seed)


if __name__ == "__main__":
    main()
