from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from crop_copilot.cv.data import create_data_bundle
from crop_copilot.cv.model import TemperatureScaler, create_model, expected_calibration_error


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    gradient_accumulation_steps: int = 1,
) -> tuple[float, np.ndarray, np.ndarray, torch.Tensor]:
    training = optimizer is not None
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be at least 1")
    model.train(training)
    losses: list[float] = []
    all_targets: list[int] = []
    all_predictions: list[int] = []
    all_logits: list[torch.Tensor] = []

    if training:
        optimizer.zero_grad(set_to_none=True)
    for batch_index, (images, targets) in enumerate(loader):
        images, targets = images.to(device), targets.to(device)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, targets)
            if training:
                (loss / gradient_accumulation_steps).backward()
                should_step = (
                    (batch_index + 1) % gradient_accumulation_steps == 0
                    or batch_index + 1 == len(loader)
                )
                if should_step:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu()))
        predictions = logits.argmax(dim=1)
        all_targets.extend(targets.detach().cpu().tolist())
        all_predictions.extend(predictions.detach().cpu().tolist())
        all_logits.append(logits.detach().cpu())

    return (
        float(np.mean(losses)),
        np.asarray(all_targets),
        np.asarray(all_predictions),
        torch.cat(all_logits),
    )


def train(config_path: Path) -> dict[str, float]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed_everything(config["seed"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = create_data_bundle(
        Path(config["data_dir"]),
        config["image_size"],
        config["batch_size"],
        config["num_workers"],
    )
    model = create_model(config["model_name"], len(data.class_names), config["pretrained"])
    model.to(device)
    criterion = nn.CrossEntropyLoss(
        weight=data.class_weights.to(device), label_smoothing=config["label_smoothing"]
    )
    optimizer = AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.3, patience=2)

    best_loss = float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    checkpoint_path = output_dir / "best_model.pt"
    last_checkpoint_path = output_dir / "last_training_checkpoint.pt"
    start_epoch = 1
    resumed_from_epoch = 0

    if config.get("resume", False):
        resume_path = (
            last_checkpoint_path if last_checkpoint_path.exists() else checkpoint_path
        )
        if resume_path.exists():
            resume_checkpoint = torch.load(
                resume_path, map_location=device, weights_only=True
            )
            if resume_checkpoint.get("model_name") != config["model_name"]:
                raise ValueError("Resume checkpoint model does not match the configuration")
            if resume_checkpoint.get("class_names") != data.class_names:
                raise ValueError("Resume checkpoint classes do not match the prepared dataset")
            model.load_state_dict(resume_checkpoint["model_state"])
            resumed_from_epoch = int(resume_checkpoint.get("epoch", 0))
            start_epoch = resumed_from_epoch + 1
            history = list(resume_checkpoint.get("history", []))
            best_loss = float(resume_checkpoint.get("best_val_loss", float("inf")))
            epochs_without_improvement = int(
                resume_checkpoint.get("epochs_without_improvement", 0)
            )
            if "optimizer_state" in resume_checkpoint:
                optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
            if "scheduler_state" in resume_checkpoint:
                scheduler.load_state_dict(resume_checkpoint["scheduler_state"])
            if not np.isfinite(best_loss):
                best_loss, _, _, _ = run_epoch(
                    model, data.val_loader, criterion, device
                )
            print(
                json.dumps(
                    {
                        "resumed_from": str(resume_path),
                        "resumed_from_epoch": resumed_from_epoch,
                        "best_val_loss": best_loss,
                    }
                )
            )

    for epoch in range(start_epoch, config["epochs"] + 1):
        train_loss, y_train, p_train, _ = run_epoch(
            model,
            data.train_loader,
            criterion,
            device,
            optimizer,
            config.get("gradient_accumulation_steps", 1),
        )
        val_loss, y_val, p_val, _ = run_epoch(model, data.val_loader, criterion, device)
        scheduler.step(val_loss)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_macro_f1": f1_score(y_train, p_train, average="macro", zero_division=0),
            "val_macro_f1": f1_score(y_val, p_val, average="macro", zero_division=0),
        }
        history.append(row)
        print(json.dumps(row))
        improved = val_loss < best_loss - 1e-4
        if improved:
            best_loss = val_loss
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_name": config["model_name"],
                    "class_names": data.class_names,
                    "image_size": config["image_size"],
                    "epoch": epoch,
                    "best_val_loss": best_loss,
                    "history": history,
                    "temperature": 1.0,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "model_name": config["model_name"],
                "class_names": data.class_names,
                "image_size": config["image_size"],
                "epoch": epoch,
                "best_val_loss": best_loss,
                "epochs_without_improvement": epochs_without_improvement,
                "history": history,
            },
            last_checkpoint_path,
        )
        if not improved and epochs_without_improvement >= config["patience"]:
            break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    val_loss, y_val, p_val, val_logits = run_epoch(model, data.val_loader, criterion, device)
    scaler = TemperatureScaler()
    temperature = scaler.fit(val_logits, torch.tensor(y_val))
    checkpoint["temperature"] = temperature
    torch.save(checkpoint, checkpoint_path)

    probabilities = torch.softmax(val_logits / temperature, dim=1)
    metrics = {
        "best_val_loss": best_loss,
        "val_accuracy": accuracy_score(y_val, p_val),
        "val_macro_f1": f1_score(y_val, p_val, average="macro", zero_division=0),
        "val_ece_calibrated": expected_calibration_error(probabilities, torch.tensor(y_val)),
        "temperature": temperature,
        "epochs_completed": len(history),
        "resumed_from_epoch": resumed_from_epoch,
        "physical_batch_size": config["batch_size"],
        "gradient_accumulation_steps": config.get(
            "gradient_accumulation_steps", 1
        ),
    }
    (output_dir / "class_names.json").write_text(
        json.dumps(data.class_names, indent=2), encoding="utf-8"
    )
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "train_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    if os.getenv("MLFLOW_TRACKING_URI"):
        try:
            import mlflow
        except ImportError as exc:
            raise RuntimeError('Install MLflow with: pip install -e ".[mlops]"') from exc
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "crop-copilot-cv"))
        with mlflow.start_run():
            mlflow.log_params(
                {
                    "model_name": config["model_name"],
                    "image_size": config["image_size"],
                    "batch_size": config["batch_size"],
                    "seed": config["seed"],
                    "classes": len(data.class_names),
                }
            )
            mlflow.log_metrics(metrics)
            mlflow.log_artifacts(str(output_dir), artifact_path="cv")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/training.yaml"))
    args = parser.parse_args()
    print(json.dumps(train(args.config), indent=2))


if __name__ == "__main__":
    main()
