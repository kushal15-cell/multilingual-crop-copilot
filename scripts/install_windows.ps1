$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$VenvPath = Join-Path $ProjectRoot ".venv"
$PythonExe = Join-Path $VenvPath "Scripts\python.exe"

function Assert-LastExit([string]$Step) {
    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed with exit code $LASTEXITCODE"
    }
}

Set-Location $ProjectRoot

$Python311 = py -3.11 -c "import sys; print(sys.executable)"
if (-not $Python311) {
    throw "Python 3.11 was not found. Install it and run this script again."
}

if (Test-Path $VenvPath) {
    if (Test-Path $PythonExe) {
        $ExistingVersion = & $PythonExe -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
    } else {
        $ExistingVersion = "broken"
    }
    if ($ExistingVersion -ne "3.11") {
        Write-Host "Removing incompatible virtual environment: $ExistingVersion"
        Remove-Item -Recurse -Force $VenvPath
    }
}

if (-not (Test-Path $PythonExe)) {
    py -3.11 -m venv $VenvPath
}

& $PythonExe -m pip install --upgrade pip
Assert-LastExit "pip upgrade"
& $PythonExe -m pip install -e ".[dev,speech,mlops]"
Assert-LastExit "dependency installation"
& $PythonExe scripts/setup_demo.py
Assert-LastExit "demo setup"
& $PythonExe scripts/smoke_test.py
Assert-LastExit "smoke test"

Write-Host "Installation and smoke test completed successfully."
Write-Host "Run: powershell -ExecutionPolicy Bypass -File scripts/start_demo.ps1"
