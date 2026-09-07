.PHONY: install test lint api farmer reviewer demo monitor

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check .

api:
	uvicorn crop_copilot.api.main:app --reload --port 8000

farmer:
	streamlit run src/crop_copilot/ui/farmer_app.py

reviewer:
	streamlit run src/crop_copilot/ui/reviewer_app.py --server.port 8502

demo:
	python scripts/bootstrap_demo.py

monitor:
	python -m crop_copilot.monitoring.report --events artifacts/events.jsonl

