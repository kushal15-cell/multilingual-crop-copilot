# Portfolio deployment: Streamlit Community Cloud

This deployment is an educational, review-gated portfolio prototype. It is not a field-ready
diagnosis service. It includes visibly labeled simulated market-price context for demonstrating
the tabular pipeline; these values are not live mandi evidence. Voice input uses a cached
multilingual Whisper Base model with CPU int8 inference; the first request is slower because the
model must be downloaded and initialized.

Demo curation applies the same minimum 96×96 input rule as production inference. Unreadable or
undersized official test images are skipped and documented in
`deploy_assets/demo_cases/skipped_images.json` rather than causing deployment preparation to fail.

## 1. Prepare runtime assets

Run this after real-model training and evaluation:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/prepare_deployment.ps1
```

Confirm these files exist:

```text
deploy_assets/best_model.pt
deploy_assets/class_names.json
deploy_assets/test_metrics.json
deploy_assets/deployment_manifest.json
deploy_assets/price_forecaster.joblib
deploy_assets/market_prices.csv
deploy_assets/demo_cases/manifest.json
```

The script records the checkpoint SHA-256 and measured test results, regenerates synthetic data
for seven Karnataka markets, trains the demonstration price forecaster, scores all 68 held-out
PlantDoc test images, and exports curated walkthrough cases. It does not copy `.env`, the full raw
or processed image dataset, or the resumable training checkpoint.

Before publishing, test the same combined entry point locally with
`scripts/start_portfolio_app.ps1`. It opens one Streamlit site with Farmer, Reviewer, and Model Card
tabs. Close any older service already using port 8501 first.

## 2. Publish the repository

Create an empty GitHub repository and, from the project root, run the commands shown by GitHub.
Before committing, inspect the staged files and ensure `.env`, databases, secrets, raw images, and
training-only artifacts are absent.

```powershell
git init
git add .
git status
git commit -m "Deploy calibrated PlantDoc crop copilot"
git branch -M main
git remote add origin <YOUR_GITHUB_REPOSITORY_URL>
git push -u origin main
```

The deployable checkpoint is approximately 21 MB, below GitHub's ordinary 100 MB per-file limit.
Do not commit future checkpoints above that limit without choosing an explicit artifact registry.

## 3. Create the Streamlit application

1. Sign in to Streamlit Community Cloud with GitHub.
2. Select the repository and `main` branch.
3. Set the entry point to `streamlit_app.py`.
4. Select Python 3.11.
5. Add this secret, using a long random value:

```toml
REVIEWER_PASSWORD = "replace-with-a-long-random-password"
```

6. Deploy and wait for the model to load.

Never place the real reviewer password in `.streamlit/secrets.toml`, `.env`, source code, screenshots,
or Git history.

## 4. Acceptance test

Run `python scripts/verify_deployment.py` before publishing. It checks the checkpoint SHA-256,
all four real-model walkthrough cases in three languages, and review rejection/status retrieval
using temporary records and offline weather. `requirements.txt` uses `constraints.txt` to retain
the tested model-library versions. Reviewer access stays disabled when no password is configured.
The API also disables its default development reviewer key in production. Low-confidence API
responses include confidence in metadata but contain no disease label or alternative predictions.

- Open the Model card tab and confirm the published metrics.
- Select each held-out demo case and confirm that abstention and review routing match its label.
- Upload a valid tomato image and confirm a confidence value is shown.
- Confirm a confidence below 0.65 releases no preliminary disease label.
- Ask a pesticide/dosage question and confirm it enters `pending_review`.
- Sign in to the Reviewer tab, reject the test draft, and refresh the farmer status.
- Restart the app and verify that the UI explains that public-demo review records are ephemeral.

## Known hosting limitations

- Review records, monitoring events, and feedback use ephemeral local files and can disappear on
  restart or redeployment.
- The hosted MVP uses one process and one model replica.
- The reviewer password is an MVP gate, not organizational authentication or RBAC.
- Weather can be temporarily unavailable. Market context is synthetic demonstration data, not a
  current or historical mandi source, and must not be used for sale decisions.
- Speech transcription quality varies by language, accent, microphone, noise, and recording length.
- External phone-photo accuracy remains unmeasured and low-confidence coverage is expected.
