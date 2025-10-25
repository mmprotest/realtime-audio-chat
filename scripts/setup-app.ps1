$ErrorActionPreference = "Stop"
python -V
python -m venv .venv-app
. .\.venv-app\Scripts\Activate.ps1
pip install --upgrade pip wheel
if (Test-Path ".\requirements.txt") {
  pip install -r .\requirements.txt
}
if (Test-Path ".\requirements-fish-speech-deps.txt") {
  pip install -r .\requirements-fish-speech-deps.txt
}
if (Test-Path ".\requirements-fish-speech.txt") {
  pip install --no-deps -r .\requirements-fish-speech.txt
}
if (Test-Path ".\constraints-app.txt") { pip install -r .\constraints-app.txt }
Write-Host "venv-app ready"
