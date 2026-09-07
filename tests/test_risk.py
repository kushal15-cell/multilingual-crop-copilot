from pathlib import Path

from crop_copilot.agent.risk import RiskPolicy
from crop_copilot.schemas import AdviceDraft, DiseasePrediction, Evidence


def test_pesticide_and_quantity_require_review() -> None:
    policy = RiskPolicy(Path("configs/risk_policy.yaml"))
    prediction = DiseasePrediction(
        label="tomato_early_blight", display_name="Early blight", confidence=0.93
    )
    draft = AdviceDraft(
        summary="Possible early blight",
        actions=["Spray 2 ml per litre of fungicide"],
        limitations="Requires field confirmation",
    )
    risk = policy.assess("What pesticide dose?", prediction, draft, [])
    assert risk.requires_review is True
    assert "chemical_or_dosage" in risk.categories
    assert "quantity_instruction" in risk.categories


def test_validated_low_risk_advice_can_pass() -> None:
    policy = RiskPolicy(Path("configs/risk_policy.yaml"))
    prediction = DiseasePrediction(
        label="tomato_early_blight", display_name="Early blight", confidence=0.92
    )
    draft = AdviceDraft(
        summary="Monitor the affected plant",
        actions=["Photograph both leaf surfaces"],
        limitations="Image-only assessment",
    )
    evidence = [
        Evidence(
            source_id="expert_1",
            title="Reviewed guidance",
            text="Monitor symptoms",
            expert_validated=True,
        )
    ]
    risk = policy.assess("What should I observe?", prediction, draft, evidence)
    assert risk.requires_review is False

