from crop_copilot.services.knowledge import KnowledgeBase


def test_generic_tomato_evidence_covers_unmapped_tomato_disease(tmp_path) -> None:
    path = tmp_path / "knowledge.yaml"
    path.write_text(
        """
- id: tomato_generic
  crop: tomato
  disease: any
  languages: [en]
  text: Seek expert confirmation.
  source_title: Reviewed source
  source_url: null
  expert_validated: true
""".strip(),
        encoding="utf-8",
    )
    evidence = KnowledgeBase(path).search("tomato", "tomato_mosaic_virus", "en")
    assert [item.source_id for item in evidence] == ["tomato_generic"]

