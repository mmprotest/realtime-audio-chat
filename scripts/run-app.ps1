$ErrorActionPreference = "Stop"
. .\.venv-app\Scripts\Activate.ps1
# Load .env if present
if (Test-Path ".\.env") {
  Write-Host "Loading .env"
  Get-Content ".\.env" | ForEach-Object {
    if (-not $_ -or $_.TrimStart().StartsWith("#")) { return }
    $parts = $_ -split "=", 2
    if ($parts.Length -eq 2) {
      $key = $parts[0].Trim()
      $value = $parts[1].Trim()
      if ($key) { Set-Item -Path Env:$key -Value $value }
    }
  }
}
# Ensure app sees STT_URL if you want remote STT
if (-not $env:STT_URL) { $env:STT_URL = "http://127.0.0.1:5007" }
# replace with your actual app entrypoint
python -m app
