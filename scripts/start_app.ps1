$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $PythonExe)) {
    throw "Virtual environment not found. Run scripts/install_windows.ps1 first."
}

Set-Location $ProjectRoot

function Test-ListeningPort([int]$Port) {
    return $null -ne (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
}

$ApiCommand = "Set-Location '$ProjectRoot'; & '$PythonExe' -m uvicorn crop_copilot.api.main:app --port 8000"
$FarmerCommand = "Set-Location '$ProjectRoot'; & '$PythonExe' -m streamlit run src/crop_copilot/ui/farmer_app.py --server.port 8501"
$ReviewerCommand = "Set-Location '$ProjectRoot'; & '$PythonExe' -m streamlit run src/crop_copilot/ui/reviewer_app.py --server.port 8502"

$ApiWasRunning = Test-ListeningPort 8000
$FarmerWasRunning = Test-ListeningPort 8501
$ReviewerWasRunning = Test-ListeningPort 8502

if (-not $ApiWasRunning) {
    Start-Process powershell.exe -ArgumentList "-NoExit", "-Command", $ApiCommand
    Start-Sleep -Seconds 4
} else {
    Write-Host "API already running on port 8000; not starting a duplicate."
}
if (-not $FarmerWasRunning) {
    Start-Process powershell.exe -ArgumentList "-NoExit", "-Command", $FarmerCommand
} else {
    Write-Host "Farmer UI already running on port 8501; not starting a duplicate."
}
if (-not $ReviewerWasRunning) {
    Start-Process powershell.exe -ArgumentList "-NoExit", "-Command", $ReviewerCommand
} else {
    Write-Host "Reviewer UI already running on port 8502; not starting a duplicate."
}
if (-not $FarmerWasRunning) {
    Start-Sleep -Seconds 5
    Start-Process "http://localhost:8501"
}

Write-Host "Farmer UI:   http://localhost:8501"
Write-Host "Reviewer UI: http://localhost:8502"
Write-Host "API docs:    http://localhost:8000/docs"
