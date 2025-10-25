$ErrorActionPreference = "Stop"
python -V
python -m venv .venv-app
. .\.venv-app\Scripts\Activate.ps1
pip install --upgrade pip wheel
if (Test-Path ".\requirements.txt") {
  $tempRequirements = New-TemporaryFile
  Get-Content ".\requirements.txt" |
    Where-Object { $_ -and -not $_.TrimStart().StartsWith("#") } |
    Where-Object { $_ -notmatch "pywhispercpp" } |
    Set-Content $tempRequirements
  if ((Get-Content $tempRequirements | Measure-Object -Line).Lines -gt 0) {
    pip install -r $tempRequirements
  }
  Remove-Item $tempRequirements -ErrorAction SilentlyContinue
}
if (Test-Path ".\requirements-fish-speech-deps.txt") {
  pip install -r .\requirements-fish-speech-deps.txt
}
if (Test-Path ".\requirements-fish-speech.txt") {
  pip install --no-deps -r .\requirements-fish-speech.txt
}
if (Test-Path ".\constraints-app.txt") { pip install -r .\constraints-app.txt }
Write-Host "venv-app ready"
