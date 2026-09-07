from __future__ import annotations

import re
from pathlib import Path

import yaml

from crop_copilot.schemas import AdviceDraft, DiseasePrediction, Evidence, RiskAssessment


class RiskPolicy:
    def __init__(self, policy_path: Path) -> None:
        self.policy = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
        self.terms = [str(term).casefold() for term in self.policy["review_terms"]]
        self.quantity = re.compile(self.policy["quantity_pattern"])
        self.low_confidence = float(self.policy["low_confidence_threshold"])
        self.review_confidence = float(self.policy["expert_review_confidence_threshold"])

    def assess(
        self,
        question: str,
        prediction: DiseasePrediction,
        draft: AdviceDraft,
        evidence: list[Evidence],
    ) -> RiskAssessment:
        combined = f"{question}\n{draft.as_text()}".casefold()
        categories: list[str] = []
        reasons: list[str] = []
        matched_terms = sorted({term for term in self.terms if term in combined})
        if matched_terms:
            categories.append("chemical_or_dosage")
            reasons.append(f"Matched high-stakes terms: {', '.join(matched_terms[:8])}")
        if self.quantity.search(combined):
            categories.append("quantity_instruction")
            reasons.append("Detected a numeric chemical or concentration instruction")
        if prediction.confidence < self.review_confidence:
            categories.append("diagnostic_uncertainty")
            reasons.append(
                f"Vision confidence {prediction.confidence:.1%} is below review threshold "
                f"{self.review_confidence:.1%}"
            )
        if not evidence:
            categories.append("missing_knowledge")
            reasons.append("No advisory evidence was retrieved")
        if any(not item.expert_validated for item in evidence):
            categories.append("unvalidated_knowledge")
            reasons.append("At least one retrieved advisory record lacks expert validation")
        return RiskAssessment(
            requires_review=bool(categories),
            categories=list(dict.fromkeys(categories)),
            reasons=reasons,
        )
