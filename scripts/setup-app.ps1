$ErrorActionPreference = "Stop"

python -V
python -m venv .venv-app
. .\.venv-app\Scripts\Activate.ps1

function Invoke-Pip {
  param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string[]]$Arguments
  )

  Write-Host "python -m pip $($Arguments -join ' ')"
  python -m pip @Arguments
}

function Install-FromFile {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Path,
    [string[]]$AdditionalArgs
  )

  if (-not (Test-Path $Path)) {
    return
  }

  $args = @("install")
  if ($AdditionalArgs) {
    $args += $AdditionalArgs
  }
  $args += @("-r", $Path)
  Invoke-Pip -Arguments $args
}

Invoke-Pip -Arguments @("install", "--upgrade", "pip", "wheel", "setuptools")

$preferCuda = $true
if ($env:USE_CUDA -and $env:USE_CUDA -match "^(0|false)$") {
  $preferCuda = $false
}

function Install-CudaTorch {
  param(
    [string]$IndexUrl
  )

  try {
    Invoke-Pip -Arguments @(
      "install",
      "--index-url", $IndexUrl,
      "torch==2.4.1",
      "torchaudio==2.4.1",
      "torchvision==0.19.1"
    )
    Write-Host "Installed CUDA-enabled PyTorch wheels from $IndexUrl"
    return $true
  }
  catch {
    Write-Warning "Failed to install CUDA PyTorch wheels: $($_.Exception.Message)"
    return $false
  }
}

$torchInstalled = $false
if ($preferCuda) {
  $cudaIndex = if ($env:PYTORCH_CUDA_INDEX_URL) { $env:PYTORCH_CUDA_INDEX_URL } else { "https://download.pytorch.org/whl/cu121" }
  $torchInstalled = Install-CudaTorch -IndexUrl $cudaIndex
}

if (-not $torchInstalled) {
  Invoke-Pip -Arguments @(
    "install",
    "torch==2.4.1",
    "torchaudio==2.4.1",
    "torchvision==0.19.1"
  )
}

Install-FromFile -Path ".\requirements.txt"
Install-FromFile -Path ".\requirements-fish-speech-deps.txt"
Install-FromFile -Path ".\requirements-fish-speech.txt" -AdditionalArgs @("--no-deps")
Install-FromFile -Path ".\constraints-app.txt"

Write-Host "venv-app ready"
