# Multilingual Crop Health & Advisory Copilot

An end-to-end responsible-AI project for field-condition crop disease diagnosis and localized farm advice. A farmer submits a leaf photograph and a spoken or written question. The system combines a calibrated PlantDoc vision model with weather context, a mandi-price forecasting model, a grounded advisory knowledge base, multilingual generation, and a mandatory agronomist gate for high-risk recommendations.

> **Safety boundary:** this is an educational decision-support system, not an autonomous agronomist. It abstains on uncertain images. Pesticide names, dosage, mixing, application frequency, withholding periods, and similarly high-stakes instructions are never released before human approval.

## Current real-model evidence

The vision component is not trained on synthetic or PlantVillage imagery. It uses the official
CC BY 4.0 PlantDoc classification repository, which contains internet-sourced field-condition
images. After label-scope filtering, duplicate removal, and conflicting-label quarantine, the
tomato split contains 552 training, 97 validation, and 68 official test images across eight
classes.

| Metric | Result |
| --- | ---: |
| Official test accuracy | 0.529 |
| Official test macro F1 | 0.509 |
| Expected calibration error | 0.108 |
| Coverage at confidence >= 0.65 | 0.265 |
| Accuracy among covered predictions | 0.833 |

The low overall score and low coverage are known limitations, not hidden failures. The deployed
prototype abstains below 0.65 and exposes high-risk advice only after review. External phone-photo
generalization has not yet been established. The bundled market-price records remain synthetic
software-test data and must not be represented as live mandi evidence.

## System flow

1. Validate the image and transcribe optional audio.
2. Predict disease with an EfficientNet model trained on field-condition data.
3. Abstain or request review when confidence is insufficient.
4. Fetch current weather and local price context.
5. Retrieve curated agronomy passages and generate advice in the requested language.
6. Run the generated draft through a deterministic risk policy.
7. Return low-risk advice immediately or store a private draft in the agronomist queue.
8. Record predictions, confidence, latency, review outcomes, and feedback for monitoring.

## Repository map

```text
configs/                   labels, prompts and safety policy
data/knowledge/            curated advisory records and source metadata
scripts/                   data preparation and local demo utilities
src/crop_copilot/cv/       PlantDoc preparation, training and inference
src/crop_copilot/tabular/  mandi-price feature engineering and forecasting
src/crop_copilot/services/ weather, market, speech, LLM and review storage
src/crop_copilot/agent/    grounding, orchestration and risk gate
src/crop_copilot/api/      FastAPI endpoints
src/crop_copilot/ui/       farmer and agronomist Streamlit applications
src/crop_copilot/monitoring/ structured events and monitoring report
tests/                     unit and integration-style tests with fakes
```

## Quick start

Python 3.11 is recommended.

### Windows: complete setup and demo

From PowerShell in the extracted project folder, one command creates the correct Python 3.11
environment, installs all application/development/speech/MLOps dependencies, prepares synthetic
demo assets, trains the demonstration price model, and runs the offline safety smoke test:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/install_windows.ps1
```

Start the API and both interfaces:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_demo.ps1
```

This opens the farmer UI at `http://localhost:8501`. The reviewer UI is
`http://localhost:8502`, and API documentation is `http://localhost:8000/docs`.
The demonstration classifier returns a fixed synthetic label so the complete review workflow can
be tested before model training. It does not analyze the uploaded image and cannot be deployed as
a diagnostic model. The application blocks demo mode when `ENVIRONMENT=production`.

### Train the real PlantDoc tomato model

Close the running API and Streamlit terminals, then run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/train_real_plantdoc.ps1
```

The workflow clones the authors' official CC BY 4.0 classification repository, filters the first
product version to tomato labels, preserves the official test set, removes exact train/test pixel
duplicates from training, creates a fixed validation split, runs EDA, fine-tunes EfficientNet-B0,
evaluates calibration and selective accuracy, and asks for explicit approval before disabling demo
mode. Training can take several hours on a CPU.

After activation, start the real checkpoint without resetting demo mode:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_app.ps1
```

`start_demo.ps1` deliberately recreates synthetic demo assets and re-enables the fixed demo
classifier; do not use it after activating a real checkpoint.

### Deploy the portfolio prototype

The hosted entry point is `streamlit_app.py`. It runs the farmer interface, password-gated review
queue, model inference, risk policy, and deterministic advisory generator in one memory-conscious
process. Whisper Base is loaded once on the first transcription request and reused; its CPU int8
configuration is a deliberate compromise between multilingual quality, speed, and memory. The
public-demo SQLite review queue and feedback log are ephemeral and may reset whenever the host
restarts.

Prepare the trained checkpoint and required small runtime assets:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/prepare_deployment.ps1
```

Test the exact single-site portfolio build locally:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_portfolio_app.ps1
```

This opens one site at `http://localhost:8501`; Farmer, Reviewer, and Model Card are tabs. It uses
the local `REVIEWER_API_KEY` as the temporary reviewer password unless `REVIEWER_PASSWORD` is set.
The older `start_app.ps1` remains available for development because it deliberately runs the API,
farmer UI, and reviewer UI as separate services.

Verify that `deploy_assets/best_model.pt`, `class_names.json`, `test_metrics.json`,
`price_forecaster.joblib`, `market_prices.csv`, and `demo_cases/manifest.json` exist. The market
table covers seven Karnataka locations but is synthetic and is prominently labeled as simulated
in the UI and generated advice. The demo cases are selected from the official held-out PlantDoc
test split, never the training split. Commit the project to GitHub, excluding `.env`, raw/processed
training images, training-only checkpoints, and local databases. In Streamlit Community Cloud,
choose `streamlit_app.py` as the entry point, select Python 3.11, and configure a strong
`REVIEWER_PASSWORD` in the secret settings. Do not store the real password in Git.

### Manual setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -e ".[dev,speech,mlops]"
cp .env.example .env
```

Create small synthetic market data and initialize local folders:

```bash
python scripts/bootstrap_demo.py
```

For a complete local product smoke test, use `python scripts/setup_demo.py` followed by
`python scripts/smoke_test.py`.

Profile the real raw images and normalized market history before training:

```bash
python scripts/run_eda.py \
  --images data/raw/plantdoc \
  --market data/processed/market_prices.csv
```

If a downloaded mandi CSV uses different headings or price units, map it with
`scripts/prepare_market_data.py --help` rather than editing the training code.

### 1. Prepare PlantDoc

Download PlantDoc yourself and follow its license/attribution requirements. Place classification images in one directory per class:

```text
data/raw/plantdoc/
  Tomato leaf early blight/
    image_001.jpg
  Tomato leaf late blight/
    image_002.jpg
  Tomato leaf/
    image_003.jpg
```

Then create leakage-safe, stratified train/validation/test manifests. Files are copied into ImageFolder-compatible directories; perceptual hashing prevents exact duplicate images crossing splits.

```bash
python scripts/prepare_plantdoc.py \
  --source data/raw/plantdoc \
  --output data/processed/plantdoc
```

Train and evaluate:

```bash
python -m crop_copilot.cv.train --config configs/training.yaml
python -m crop_copilot.cv.evaluate \
  --checkpoint artifacts/cv/best_model.pt \
  --data-dir data/processed/plantdoc/test
```

The training command saves model weights, class names, metrics, and the validation-fitted temperature used for confidence calibration.

### 2. Train the price model

Supply a CSV with these columns:

```text
date,market,crop,modal_price,min_price,max_price,rainfall_mm,temp_c,humidity
```

Rows must be ordered observations for each `market + crop`. The trainer creates lag and rolling-window features internally and uses a chronological holdout.

```bash
python -m crop_copilot.tabular.train_price_model \
  --input data/processed/market_prices.csv \
  --output artifacts/price/price_forecaster.joblib
```

### 3. Run the product

```bash
uvicorn crop_copilot.api.main:app --reload --port 8000
streamlit run src/crop_copilot/ui/farmer_app.py
streamlit run src/crop_copilot/ui/reviewer_app.py --server.port 8502
```

API documentation is available at `http://localhost:8000/docs`.

### 4. Optional multilingual speech and LLM

- STT: install the `speech` extra to use `faster-whisper` locally.
- TTS: the `speech` extra uses gTTS for Kannada (`kn`), Hindi (`hi`), and other supported languages.
- LLM: configure an OpenAI-compatible endpoint in `.env`. With no endpoint, the application uses a transparent deterministic template rather than pretending a model was called.

```bash
pip install -e ".[speech]"
```

## Human-review behavior

The farmer endpoint never returns an unapproved high-risk draft. It returns only a neutral holding message and a review ID. The reviewer UI shows the evidence, source passages, confidence, and private draft. An agronomist may edit and approve it or reject it with a reason. Only the approved text becomes retrievable by the farmer.

## MLOps

- DVC stages: image preparation, CV training, CV evaluation, and price training.
- MLflow: optional experiment logging when `MLFLOW_TRACKING_URI` is configured.
- Monitoring: JSONL inference events plus an aggregate report covering confidence, abstention, review rate, class mix, latency, and feedback.
- CI: formatting, linting, compilation, and unit tests without downloading a model or dataset.

```bash
dvc repro
python -m crop_copilot.monitoring.report --events artifacts/events.jsonl
pytest
ruff check .
```

Install the MLOps tools before running DVC/MLflow commands: `pip install -e ".[mlops]"`.

## Data and evaluation principles

- Split near-duplicate images as a group; never allow versions of the same photograph into different splits.
- Report macro F1, per-class precision/recall, confusion matrix, calibration error, coverage, and selective accuracy—not accuracy alone.
- Inspect performance by crop, disease, lighting/quality metadata, and confidence band where labels exist.
- Fit preprocessing and confidence calibration on training/validation data only.
- Treat market forecasting as context, not a causal estimate of disease-related yield loss.

## Production gaps to document honestly

- PlantDoc is still too small to represent all Indian farms, devices, seasons, and disease stages.
- The bundled knowledge records are examples and must be replaced or reviewed by qualified regional agronomists.
- Live mandi ingestion is region/provider-specific; the default adapter reads a normalized local CSV.
- Authentication, encrypted object storage, consent, retention policies, rate limits, and a managed database are required before real deployment.
- A field pilot with labeled follow-up outcomes is needed before claiming agronomic benefit.
