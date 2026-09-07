from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"


def set_env_value(path: Path, key: str, value: str) -> None:
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    prefix = f"{key}="
    updated: list[str] = []
    replaced = False
    for line in lines:
        if line.startswith(prefix):
            updated.append(f"{key}={value}")
            replaced = True
        else:
            updated.append(line)
    if not replaced:
        updated.append(f"{key}={value}")
    path.write_text("\n".join(updated) + "\n", encoding="utf-8")


def create_sample_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (640, 480), "#d8edc8")
    draw = ImageDraw.Draw(image)
    draw.ellipse((130, 40, 510, 440), fill="#448b3a", outline="#245c28", width=8)
    draw.line((320, 50, 320, 430), fill="#d2e6a3", width=8)
    for x, y, radius in [(245, 165, 25), (375, 220, 32), (280, 320, 22)]:
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill="#8a5b31",
            outline="#e0b75e",
            width=6,
        )
    draw.text((15, 450), "SYNTHETIC DEMO IMAGE - NOT A REAL LEAF", fill="#111111")
    image.save(path, quality=92)


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    if not ENV_PATH.exists():
        ENV_PATH.write_text(
            (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8"), encoding="utf-8"
        )
    set_env_value(ENV_PATH, "ENVIRONMENT", "development")
    set_env_value(ENV_PATH, "ALLOW_DEMO_CLASSIFIER", "true")
    run([sys.executable, "scripts/bootstrap_demo.py"])
    run(
        [
            sys.executable,
            "-m",
            "crop_copilot.tabular.train_price_model",
            "--input",
            "data/processed/market_prices.csv",
            "--output",
            "artifacts/price/price_forecaster.joblib",
            "--test-days",
            "30",
        ]
    )
    sample_path = PROJECT_ROOT / "data/sample/demo_leaf.jpg"
    create_sample_image(sample_path)
    print("\nDemo setup complete.")
    print(f"Synthetic image: {sample_path}")
    print("ALLOW_DEMO_CLASSIFIER=true was written to .env for local testing only.")
    print("The demo prediction is fixed and must never be treated as a diagnosis.")


if __name__ == "__main__":
    main()

