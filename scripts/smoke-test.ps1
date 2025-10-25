$ErrorActionPreference = "Stop"

# 1) start STT
$sttProcess = Start-Process -NoNewWindow -PassThru powershell -ArgumentList "-NoProfile","-ExecutionPolicy Bypass","-File","scripts/run-stt.ps1"
Start-Sleep -Seconds 2

try {
  # 2) health check
  $health = Invoke-WebRequest -UseBasicParsing http://127.0.0.1:5007/health
  if ($health.StatusCode -ne 200) { throw "STT health check failed" }

  # 3) transcribe test file
  $Form = @{
    file = Get-Item ".\samples\hello.wav"
  }
  $r = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:5007/v1/transcribe" -Method Post -Form $Form
  $r.Content | Write-Host

  Write-Host "STT smoke test OK"
}
finally {
  if ($sttProcess -and -not $sttProcess.HasExited) {
    Stop-Process -Id $sttProcess.Id -Force
  }
}
