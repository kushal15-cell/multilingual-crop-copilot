from __future__ import annotations

from pathlib import Path

from crop_copilot.schemas import DiseasePrediction


class DemoClassifier:
    """Fixed synthetic result for plumbing tests; it never analyzes image pixels."""

    def predict(self, image_path: str) -> DiseasePrediction:
        if not Path(image_path).exists():
            raise FileNotFoundError(image_path)
        return DiseasePrediction(
            label="tomato_leaf_early_blight",
            display_name="DEMO: Tomato leaf early blight",
            confidence=0.93,
            calibrated=False,
            model_version="demo-not-a-diagnosis",
        )

