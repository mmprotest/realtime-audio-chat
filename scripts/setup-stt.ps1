$ErrorActionPreference = "Stop"
python -V
python -m venv .venv-stt
. .\.venv-stt\Scripts\Activate.ps1
pip install --upgrade pip wheel
pip install -r stt_service/requirements-stt.txt
Write-Host "venv-stt ready"
