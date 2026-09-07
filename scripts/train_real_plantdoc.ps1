$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$DatasetRoot = Join-Path $ProjectRoot "data\raw\plantdoc-official"
$ProcessedRoot = Join-Path $ProjectRoot "data\processed\plantdoc"

function Assert-LastExit([string]$Step) {
    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed with exit code $LASTEXITCODE"
    }
}

if (-not (Test-Path $PythonExe)) {
    throw "Python environment missing. Run scripts/install_windows.ps1 first."
}

Set-Location $ProjectRoot

if (Test-Path (Join-Path $DatasetRoot ".git")) {
    git -C $DatasetRoot pull --ff-only
    Assert-LastExit "PlantDoc update"

    $OfficialTrain = Join-Path $DatasetRoot "train"
    $OfficialTest = Join-Path $DatasetRoot "test"
    if (-not (Test-Path $OfficialTrain) -or -not (Test-Path $OfficialTest)) {
        Write-Warning "Official dataset folders are missing from the working tree; restoring them from Git."
        git -C $DatasetRoot restore --source=HEAD --worktree -- train test LICENSE.txt
        Assert-LastExit "PlantDoc working-tree restore"
    }
} elseif (Test-Path $DatasetRoot) {
    throw "$DatasetRoot exists but is not the official Git repository. Move it and rerun."
} else {
    git clone --depth 1 https://github.com/pratikkayal/PlantDoc-Dataset.git $DatasetRoot
    Assert-LastExit "PlantDoc download"
}

if (-not (Test-Path (Join-Path $DatasetRoot "train")) -or
    -not (Test-Path (Join-Path $DatasetRoot "test"))) {
    throw "PlantDoc checkout is incomplete: expected train and test under $DatasetRoot"
}

if (-not (Test-Path (Join-Path $ProcessedRoot "manifest.csv"))) {
    & $PythonExe scripts/prepare_official_plantdoc.py `
        --source $DatasetRoot `
        --output $ProcessedRoot `
        --label-regex tomato `
        --val-ratio 0.15 `
        --seed 42
    Assert-LastExit "PlantDoc preparation"
} else {
    Write-Host "Prepared PlantDoc manifest already exists; preserving the existing split."
}

& $PythonExe scripts/run_eda.py `
    --images $ProcessedRoot `
    --market data/processed/market_prices.csv `
    --output artifacts/eda/real_plantdoc_summary.json
Assert-LastExit "PlantDoc EDA"

$Device = & $PythonExe -c "import torch; print('CUDA GPU' if torch.cuda.is_available() else 'CPU')"
Assert-LastExit "hardware detection"
Write-Host "Training device: $Device"
if ($Device -eq "CPU") {
    Write-Warning "EfficientNet training can take several hours on CPU. Do not close the terminal."
}

& $PythonExe -m crop_copilot.cv.train --config configs/training_real.yaml
Assert-LastExit "PlantDoc training"

& $PythonExe -m crop_copilot.cv.evaluate `
    --checkpoint artifacts/cv/best_model.pt `
    --data-dir data/processed/plantdoc/test `
    --output-dir artifacts/cv
Assert-LastExit "PlantDoc evaluation"

Write-Host "Training and evaluation completed. Review artifacts/cv/test_metrics.json."
$Approval = Read-Host "Type ACTIVATE to replace demo predictions with this real checkpoint"
if ($Approval -ceq "ACTIVATE") {
    & $PythonExe scripts/activate_real_model.py
    Assert-LastExit "real-model activation"
    Write-Host "Start the real application with scripts/start_app.ps1"
} else {
    Write-Warning "Checkpoint was not activated. Demo mode remains unchanged."
}
