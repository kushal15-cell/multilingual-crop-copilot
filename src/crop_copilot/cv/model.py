from __future__ import annotations

import timm
import torch
from torch import nn


def create_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)


class TemperatureScaler(nn.Module):
    """Post-hoc calibration fitted on validation logits only."""

    def __init__(self, initial_temperature: float = 1.0) -> None:
        super().__init__()
        self.log_temperature = nn.Parameter(torch.tensor(initial_temperature).log())

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp(0.05, 20.0)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 50) -> float:
        self.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.log_temperature], lr=0.05, max_iter=max_iter)

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss = criterion(self(logits), labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        return float(self.temperature.detach().cpu())


def expected_calibration_error(
    probabilities: torch.Tensor, labels: torch.Tensor, bins: int = 15
) -> float:
    confidences, predictions = probabilities.max(dim=1)
    accuracies = predictions.eq(labels)
    ece = torch.zeros((), device=probabilities.device)
    boundaries = torch.linspace(0, 1, bins + 1, device=probabilities.device)
    for lower, upper in zip(boundaries[:-1], boundaries[1:], strict=True):
        mask = confidences.gt(lower) & confidences.le(upper)
        if mask.any():
            ece += mask.float().mean() * (
                confidences[mask].mean() - accuracies[mask].float().mean()
            ).abs()
    return float(ece.cpu())

