$ErrorActionPreference = "Stop"
python -V
python -m venv .venv-stt
. .\.venv-stt\Scripts\Activate.ps1

python -m pip install --upgrade pip wheel
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

python -m pip install -r stt_service/requirements-stt.txt
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "venv-stt ready"
