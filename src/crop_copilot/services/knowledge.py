from __future__ import annotations

from pathlib import Path

import yaml

from crop_copilot.schemas import Evidence


class KnowledgeBase:
    def __init__(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Knowledge base not found: {path}")
        content = yaml.safe_load(path.read_text(encoding="utf-8")) or []
        if not isinstance(content, list):
            raise ValueError("Knowledge base must contain a list of records")
        self.records = content

    def search(self, crop: str, disease: str, language: str, limit: int = 4) -> list[Evidence]:
        crop_key, disease_key = crop.casefold(), disease.casefold()
        scored: list[tuple[int, dict]] = []
        for record in self.records:
            score = 0
            record_crop = str(record.get("crop", "")).casefold()
            record_disease = str(record.get("disease", "")).casefold()
            aliases = {str(alias).casefold() for alias in record.get("aliases", [])}
            if record_crop == crop_key:
                score += 3
            elif record_crop == "any":
                score += 1
            if record_disease == disease_key or disease_key in aliases:
                score += 5
            elif record_disease == "any":
                score += 1
            elif record_disease == "healthy" and "healthy" in disease_key:
                score += 2
            if language in record.get("languages", []):
                score += 1
            if score:
                scored.append((score, record))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            Evidence(
                source_id=record["id"],
                title=record["source_title"],
                text=record["text"],
                url=record.get("source_url"),
                expert_validated=bool(record.get("expert_validated", False)),
            )
            for _, record in scored[:limit]
        ]
