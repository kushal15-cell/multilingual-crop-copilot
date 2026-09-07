from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from torchvision import datasets

from crop_copilot.cv.data import build_transforms
from crop_copilot.cv.model import create_model, expected_calibration_error


def evaluate(checkpoint_path: Path, data_dir: Path, output_dir: Path) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    class_names = checkpoint["class_names"]
    _, transform = build_transforms(checkpoint["image_size"])
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    if dataset.classes != class_names:
        raise ValueError("Evaluation class order differs from checkpoint class order")
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    model = create_model(checkpoint["model_name"], len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    logits_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []
    with torch.inference_mode():
        for images, labels in loader:
            logits_list.append(model(images))
            labels_list.append(labels)
    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)
    probabilities = torch.softmax(logits / checkpoint.get("temperature", 1.0), dim=1)
    confidence, predictions = probabilities.max(dim=1)
    y_true, y_pred = labels.numpy(), predictions.numpy()

    coverage_mask = confidence.numpy() >= 0.65
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "ece": expected_calibration_error(probabilities, labels),
        "coverage_at_0_65": float(coverage_mask.mean()),
        "selective_accuracy_at_0_65": (
            float((y_true[coverage_mask] == y_pred[coverage_mask]).mean())
            if coverage_mask.any()
            else None
        ),
        "per_class": classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    pd.DataFrame(matrix, index=class_names, columns=class_names).to_csv(
        output_dir / "confusion_matrix.csv"
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/cv"))
    args = parser.parse_args()
    print(json.dumps(evaluate(args.checkpoint, args.data_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
