$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$EntryPoint = Join-Path $ProjectRoot "streamlit_app.py"
$Checkpoint = Join-Path $ProjectRoot "deploy_assets\best_model.pt"

if (-not (Test-Path $PythonExe)) {
    throw "Virtual environment not found. Run scripts/install_windows.ps1 first."
}
if (-not (Test-Path $Checkpoint)) {
    throw "Deployment assets not found. Run scripts/prepare_deployment.ps1 first."
}

if (-not $env:REVIEWER_PASSWORD) {
    $EnvPath = Join-Path $ProjectRoot ".env"
    if (Test-Path $EnvPath) {
        $ReviewerEntry = Get-Content $EnvPath | Where-Object {
            $_ -match '^REVIEWER_API_KEY='
        } | Select-Object -First 1
        if ($ReviewerEntry) {
            $env:REVIEWER_PASSWORD = ($ReviewerEntry -split '=', 2)[1].Trim()
        }
    }
}

$Existing = Get-NetTCPConnection -LocalPort 8501 -State Listen -ErrorAction SilentlyContinue
if ($Existing) {
    Write-Host "A Streamlit app is already running on http://localhost:8501"
    Write-Host "Close its terminal before switching to the unified portfolio app."
    Start-Process "http://localhost:8501"
    exit 0
}

$Command = "Set-Location '$ProjectRoot'; & '$PythonExe' -m streamlit run '$EntryPoint' --server.port 8501"
Start-Process powershell.exe -ArgumentList "-NoExit", "-Command", $Command
Start-Sleep -Seconds 5
Start-Process "http://localhost:8501"

Write-Host "Unified portfolio app: http://localhost:8501"
Write-Host "Farmer, Reviewer, and Model Card are tabs in this single site."
