$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$DeployRoot = Join-Path $ProjectRoot "deploy_assets"
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

function Assert-LastExit([string]$Step) {
    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed with exit code $LASTEXITCODE"
    }
}

if (-not (Test-Path $PythonExe)) {
    throw "Virtual-environment Python was not found: $PythonExe"
}

$RequiredFiles = @{
    "best_model.pt" = Join-Path $ProjectRoot "artifacts\cv\best_model.pt"
    "class_names.json" = Join-Path $ProjectRoot "artifacts\cv\class_names.json"
    "test_metrics.json" = Join-Path $ProjectRoot "artifacts\cv\test_metrics.json"
}

foreach ($Entry in $RequiredFiles.GetEnumerator()) {
    if (-not (Test-Path $Entry.Value)) {
        throw "Deployment asset missing: $($Entry.Value)"
    }
}

New-Item -ItemType Directory -Force -Path $DeployRoot | Out-Null
foreach ($Entry in $RequiredFiles.GetEnumerator()) {
    Copy-Item -Force $Entry.Value (Join-Path $DeployRoot $Entry.Key)
}

Push-Location $ProjectRoot
try {
    & $PythonExe scripts\bootstrap_demo.py
    Assert-LastExit "Synthetic market generation"
    & $PythonExe -m crop_copilot.tabular.train_price_model `
        --input data\processed\market_prices.csv `
        --output artifacts\price\price_forecaster.joblib
    Assert-LastExit "Price-model training"
    & $PythonExe scripts\select_demo_images.py `
        --checkpoint artifacts\cv\best_model.pt `
        --test-dir data\processed\plantdoc\test `
        --output-dir deploy_assets\demo_cases
    Assert-LastExit "Held-out demo selection"
}
finally {
    Pop-Location
}

Copy-Item -Force `
    (Join-Path $ProjectRoot "data\processed\market_prices.csv") `
    (Join-Path $DeployRoot "market_prices.csv")
Copy-Item -Force `
    (Join-Path $ProjectRoot "artifacts\price\price_forecaster.joblib") `
    (Join-Path $DeployRoot "price_forecaster.joblib")

$Metrics = Get-Content (Join-Path $DeployRoot "test_metrics.json") -Raw | ConvertFrom-Json
$CheckpointHash = (Get-FileHash (Join-Path $DeployRoot "best_model.pt") -Algorithm SHA256).Hash
$Manifest = [ordered]@{
    prepared_at_utc = [DateTime]::UtcNow.ToString("o")
    checkpoint_sha256 = $CheckpointHash
    test_accuracy = $Metrics.accuracy
    test_macro_f1 = $Metrics.macro_f1
    ece = $Metrics.ece
    coverage_at_0_65 = $Metrics.coverage_at_0_65
    selective_accuracy_at_0_65 = $Metrics.selective_accuracy_at_0_65
    market_data_source = "synthetic_demo_v2"
    market_data_is_synthetic = $true
    markets = 7
    demo_cases = (Get-Content `
        (Join-Path $DeployRoot "demo_cases\manifest.json") -Raw | `
        ConvertFrom-Json).cases.Count
}
$Manifest | ConvertTo-Json | Set-Content (Join-Path $DeployRoot "deployment_manifest.json")

Write-Host "Deployment assets prepared in $DeployRoot"
Write-Host "Checkpoint SHA256: $CheckpointHash"
Write-Warning "Market values are synthetic demo data and are visibly labeled in the UI."
Write-Warning "Commit deploy_assets for the portfolio demo, but never commit .env or secrets."
