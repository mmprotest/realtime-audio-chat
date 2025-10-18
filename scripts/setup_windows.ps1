param(
    [string]$Python = "python"
)

$venv = ".venv"
if (-not (Test-Path $venv)) {
    & $Python -m venv $venv
}

$activate = Join-Path $venv "Scripts\\Activate.ps1"
if (-not (Test-Path $activate)) {
    Write-Error "Virtual environment activation script not found at $activate"
    exit 1
}

& $activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pytest -q
