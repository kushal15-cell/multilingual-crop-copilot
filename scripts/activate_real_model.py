from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def set_env_value(path: Path, key: str, value: str) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    prefix = f"{key}="
    replaced = False
    updated: list[str] = []
    for line in lines:
        if line.startswith(prefix):
            updated.append(f"{key}={value}")
            replaced = True
        else:
            updated.append(line)
    if not replaced:
        updated.append(f"{key}={value}")
    path.write_text("\n".join(updated) + "\n", encoding="utf-8")


def main() -> None:
    checkpoint = PROJECT_ROOT / "artifacts/cv/best_model.pt"
    metrics_path = PROJECT_ROOT / "artifacts/cv/test_metrics.json"
    env_path = PROJECT_ROOT / ".env"
    missing = [str(path) for path in (checkpoint, metrics_path, env_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Cannot activate real model; missing: {missing}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    required = {"accuracy", "macro_f1", "ece", "coverage_at_0_65"}
    if missing_metrics := required - set(metrics):
        raise ValueError(f"Evaluation metrics are incomplete: {sorted(missing_metrics)}")
    set_env_value(env_path, "ALLOW_DEMO_CLASSIFIER", "false")
    print("Real PlantDoc checkpoint activated. Restart the API and Streamlit applications.")
    print(json.dumps({key: metrics[key] for key in sorted(required)}, indent=2))


if __name__ == "__main__":
    main()

