from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from crop_copilot.cv.inference import VisionClassifier


@dataclass(frozen=True)
class ScoredImage:
    path: Path
    true_label: str
    predicted_label: str
    confidence: float
    correct: bool


def score_test_images(
    classifier: VisionClassifier, test_dir: Path
) -> tuple[list[ScoredImage], list[dict[str, str]]]:
    supported = {".jpg", ".jpeg", ".png", ".webp"}
    records: list[ScoredImage] = []
    skipped: list[dict[str, str]] = []
    for class_dir in sorted(path for path in test_dir.iterdir() if path.is_dir()):
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.suffix.casefold() not in supported:
                continue
            try:
                prediction = classifier.predict(str(image_path))
            except (OSError, ValueError) as exc:
                skipped.append(
                    {
                        "path": f"{class_dir.name}/{image_path.name}",
                        "reason": str(exc),
                    }
                )
                continue
            records.append(
                ScoredImage(
                    path=image_path,
                    true_label=class_dir.name,
                    predicted_label=prediction.display_name,
                    confidence=prediction.confidence,
                    correct=prediction.display_name == class_dir.name,
                )
            )
    if not records:
        raise ValueError(f"No supported images found below {test_dir}")
    return records, skipped


def first_unused(records: list[ScoredImage], used: set[Path]) -> ScoredImage | None:
    return next((record for record in records if record.path not in used), None)


def choose_cases(records: list[ScoredImage]) -> list[tuple[str, str, ScoredImage, str]]:
    used: set[Path] = set()
    selected: list[tuple[str, str, ScoredImage, str]] = []

    def add(case_id: str, title: str, candidates: list[ScoredImage], question: str) -> None:
        chosen = first_unused(candidates, used)
        if chosen is None:
            return
        used.add(chosen.path)
        selected.append((case_id, title, chosen, question))

    correct = [record for record in records if record.correct]
    covered = sorted(
        (record for record in correct if record.confidence >= 0.65),
        key=lambda record: record.confidence,
        reverse=True,
    )
    review_band = sorted(
        (record for record in correct if 0.65 <= record.confidence < 0.80),
        key=lambda record: abs(record.confidence - 0.72),
    )
    abstentions = sorted(
        (record for record in records if record.confidence < 0.65),
        key=lambda record: record.confidence,
    )
    healthy = sorted(
        (record for record in correct if record.true_label == "Tomato leaf"),
        key=lambda record: record.confidence,
        reverse=True,
    )

    add(
        "human_review",
        "Human-review gate (dosage question)",
        covered,
        "Which pesticide should I spray, and what exact dosage should I use?",
    )
    add(
        "confidence_review",
        "Confidence-band review",
        review_band or covered,
        "What should I do next to protect this crop?",
    )
    add(
        "safe_abstention",
        "Safe abstention",
        abstentions,
        "What disease is this, and what should I do next?",
    )
    add(
        "healthy_leaf",
        "Healthy-leaf example",
        healthy,
        "Does this leaf show a supported disease?",
    )
    if not selected:
        raise ValueError("No suitable held-out demo cases could be selected")
    return selected


def export_cases(
    selected: list[tuple[str, str, ScoredImage, str]], output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, object]] = []
    for case_id, title, record, question in selected:
        filename = f"{case_id}{record.path.suffix.casefold()}"
        shutil.copy2(record.path, output_dir / filename)
        cases.append(
            {
                "id": case_id,
                "title": title,
                "filename": filename,
                "source": "official_plantdoc_test",
                "seen_during_training": False,
                "true_label": record.true_label,
                "measured_prediction": record.predicted_label,
                "measured_confidence": round(record.confidence, 6),
                "measured_correct": record.correct,
                "recommended_question": question,
            }
        )
    manifest = {
        "description": "Curated, held-out PlantDoc test cases for a reproducible demo.",
        "confidence_threshold": 0.65,
        "cases": cases,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def write_scores(records: list[ScoredImage], destination: Path) -> None:
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "path",
                "true_label",
                "predicted_label",
                "confidence",
                "correct",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "path": f"{record.path.parent.name}/{record.path.name}",
                    "true_label": record.true_label,
                    "predicted_label": record.predicted_label,
                    "confidence": record.confidence,
                    "correct": record.correct,
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    classifier = VisionClassifier(args.checkpoint)
    records, skipped = score_test_images(classifier, args.test_dir)
    selected = choose_cases(records)
    export_cases(selected, args.output_dir)
    write_scores(records, args.output_dir / "all_test_predictions.csv")
    (args.output_dir / "skipped_images.json").write_text(
        json.dumps(skipped, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "scored_images": len(records),
                "skipped_images": len(skipped),
                "demo_cases": len(selected),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
