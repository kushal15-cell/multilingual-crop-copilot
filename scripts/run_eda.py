from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError

from crop_copilot.tabular.features import validate_market_frame

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def profile_images(root: Path) -> dict:
    class_counts: Counter[str] = Counter()
    widths: list[int] = []
    heights: list[int] = []
    modes: Counter[str] = Counter()
    corrupt: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.casefold() not in IMAGE_EXTENSIONS:
            continue
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                widths.append(image.width)
                heights.append(image.height)
                modes[image.mode] += 1
            class_counts[path.parent.name] += 1
        except (UnidentifiedImageError, OSError):
            corrupt.append(str(path))
    total = sum(class_counts.values())
    largest = max(class_counts.values(), default=0)
    smallest = min(class_counts.values(), default=0)
    return {
        "total_readable_images": total,
        "classes": len(class_counts),
        "class_counts": dict(class_counts.most_common()),
        "imbalance_ratio_largest_to_smallest": (
            largest / smallest if smallest else None
        ),
        "width": pd.Series(widths).describe().to_dict() if widths else {},
        "height": pd.Series(heights).describe().to_dict() if heights else {},
        "color_modes": dict(modes),
        "corrupt_count": len(corrupt),
        "corrupt_examples": corrupt[:20],
    }


def profile_market(path: Path) -> dict:
    raw = pd.read_csv(path)
    missing_before = raw.isna().sum().to_dict()
    frame = validate_market_frame(raw)
    price = frame["modal_price"].describe().to_dict()
    return {
        "rows": len(frame),
        "date_min": frame["date"].min().date().isoformat(),
        "date_max": frame["date"].max().date().isoformat(),
        "markets": frame["market"].nunique(),
        "crops": frame["crop"].nunique(),
        "rows_by_crop": frame["crop"].value_counts().to_dict(),
        "rows_by_market": frame["market"].value_counts().to_dict(),
        "missing_values": missing_before,
        "modal_price_summary": price,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile raw image and normalized market data")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--market", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/eda/summary.json"))
    args = parser.parse_args()
    report = {
        "images": profile_images(args.images),
        "market": profile_market(args.market),
        "warnings": [
            "Class imbalance and corrupt files require investigation before training.",
            "This profile does not prove geographic, device, or seasonal representativeness.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()

