$ErrorActionPreference = "Stop"
. .\.venv-stt\Scripts\Activate.ps1
$env:STT_PORT = $env:STT_PORT -as [string]
if (-not $env:STT_PORT) { $env:STT_PORT = "5007" }
python -m stt_service.main
