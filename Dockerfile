FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY data/knowledge ./data/knowledge
RUN pip install --no-cache-dir ".[speech]"

RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser
EXPOSE 8000
CMD ["uvicorn", "crop_copilot.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
