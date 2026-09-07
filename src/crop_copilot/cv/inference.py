from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError

from crop_copilot.cv.data import build_transforms
from crop_copilot.cv.model import create_model
from crop_copilot.schemas import DiseasePrediction


def normalize_label(label: str) -> str:
    return "_".join(label.strip().lower().replace("___", "_").replace(" ", "_").split("_"))


class VisionClassifier:
    def __init__(self, checkpoint_path: Path) -> None:
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Vision checkpoint not found at {checkpoint_path}. Train the model first."
            )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.class_names: list[str] = checkpoint["class_names"]
        self.temperature = float(checkpoint.get("temperature", 1.0))
        self.image_size = int(checkpoint["image_size"])
        self.model = create_model(
            checkpoint["model_name"], len(self.class_names), pretrained=False
        )
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(self.device).eval()
        _, self.transform = build_transforms(self.image_size)
        self.version = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()[:12]

    def predict(self, image_path: str) -> DiseasePrediction:
        try:
            with Image.open(image_path) as source:
                if source.width * source.height > 20_000_000:
                    raise ValueError("Image exceeds 20 megapixels; resize it before uploading")
                image = source.convert("RGB")
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise ValueError("Uploaded file is not a readable image") from exc
        if min(image.size) < 96:
            raise ValueError("Image is too small; use a clear image at least 96x96 pixels")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = self.model(tensor) / self.temperature
            probabilities = torch.softmax(logits, dim=1)[0]
        top_values, top_indices = probabilities.topk(min(3, len(self.class_names)))
        label = self.class_names[int(top_indices[0])]
        alternatives = [
            (normalize_label(self.class_names[int(index)]), float(value))
            for value, index in zip(top_values[1:], top_indices[1:], strict=True)
        ]
        return DiseasePrediction(
            label=normalize_label(label),
            display_name=label,
            confidence=float(top_values[0]),
            alternatives=alternatives,
            calibrated=True,
            model_version=self.version,
        )
