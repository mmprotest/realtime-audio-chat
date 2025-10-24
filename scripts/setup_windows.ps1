<#
.SYNOPSIS
    End-to-end setup script for configuring the realtime-audio-chat project on Windows with CUDA-enabled PyTorch.

.DESCRIPTION
    This script bootstraps a Windows development environment that targets CUDA 13.0-capable GPUs.
    It performs the following steps:
        * Verifies administrative privileges and the presence of CUDA 13.0 toolkits/drivers.
        * Installs system-level dependencies (Python, Visual C++ runtimes, FFmpeg) via winget when missing.
        * Creates an isolated Python virtual environment for the project.
        * Installs CUDA-enabled PyTorch wheels (cu121 build, compatible with CUDA 13 drivers) and the
          Python dependencies required by the application.
        * Validates the resulting Python environment with `pip check` and prints activation guidance.

    Run this script from the repository root using an elevated Windows PowerShell session:

        powershell.exe -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1

.PARAMETER PythonVersion
    Major/minor Python version to install and target for the virtual environment.
    The default (3.10) is compatible with all project dependencies.

.PARAMETER CudaVersion
    CUDA toolkit version expected to be available on the host. Defaults to 13.0.
    PyTorch wheels are installed from the cu121 channel which is forward compatible with CUDA 13 drivers.

.PARAMETER VenvDir
    Directory (relative to the repository root) where the Python virtual environment will be created.

.NOTES
    Requires Windows 10/11 with the winget package manager available. Run from an elevated prompt because
    the script installs system packages.
#>
param(
    [string]$PythonVersion = "3.12",
    [string]$CudaVersion = "13.0",
    [string]$VenvDir = ".venv"
)

$ErrorActionPreference = "Stop"

function Write-Section {
    param([string]$Message)
    Write-Host "`n==== $Message ====\n" -ForegroundColor Cyan
}

function Assert-Administrator {
    $currentIdentity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentIdentity)
    if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        throw "This script must be run from an elevated PowerShell session."
    }
}

function Ensure-WingetAvailable {
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        throw "winget is required but was not found. Install winget from the Microsoft Store and rerun the script."
    }
}

function Install-WingetPackage {
    param(
        [Parameter(Mandatory=$true)][string]$Id,
        [Parameter(Mandatory=$true)][string]$Description
    )

    Write-Section "Ensuring $Description ($Id)"
    $alreadyInstalled = winget list --exact --id $Id --accept-source-agreements 2>$null | Select-String -SimpleMatch $Id
    if ($alreadyInstalled) {
        Write-Host "$Description is already installed." -ForegroundColor Green
        return
    }

    winget install --id $Id --exact --silent --accept-package-agreements --accept-source-agreements
}

function Get-CudaVersion {
    $nvcc = Get-Command nvcc -ErrorAction SilentlyContinue
    if (-not $nvcc) {
        return $null
    }

    $versionLine = & $nvcc.Source --version | Select-String "release"
    if ($versionLine -and $versionLine.Line -match "release\s+([0-9]+\.[0-9]+)") {
        return $Matches[1]
    }
    return $null
}

function Resolve-PythonExecutable {
    param([string]$Version)

    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        $pyResult = & $pyLauncher.Source -$Version -c "import sys; print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $pyResult) {
            return $pyResult.Trim()
        }
    }

    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCmd) {
        $versionCheck = & $pythonCmd.Source -c "import sys; print('.'.join(map(str, sys.version_info[:2])))"
        if ($versionCheck.Trim() -eq $Version) {
            return $pythonCmd.Source
        }
    }

    return $null
}

function Ensure-Python {
    param([string]$Version)

    $pythonExe = Resolve-PythonExecutable -Version $Version
    if ($pythonExe) {
        Write-Host "Found Python $Version at $pythonExe" -ForegroundColor Green
        return $pythonExe
    }

    Write-Section "Installing Python $Version"
    $pythonPackageId = "Python.Python.$Version"
    Install-WingetPackage -Id $pythonPackageId -Description "Python $Version"
    $pythonExe = Resolve-PythonExecutable -Version $Version
    if (-not $pythonExe) {
        throw "Python $Version installation was requested but the interpreter could not be located."
    }
    return $pythonExe
}

function New-VirtualEnvironment {
    param(
        [string]$PythonExe,
        [string]$Directory
    )

    $resolvedDir = Resolve-Path -Path $Directory -ErrorAction SilentlyContinue
    if (-not $resolvedDir) {
        Write-Section "Creating virtual environment in $Directory"
        & $PythonExe -m venv $Directory
        $resolvedDir = Resolve-Path -Path $Directory
    } else {
        Write-Section "Virtual environment already exists in $Directory"
    }

    $venvPython = Join-Path ($resolvedDir.Path) "Scripts\python.exe"
    if (-not (Test-Path $venvPython)) {
        throw "Virtual environment Python executable not found at $venvPython"
    }
    return $venvPython
}

function Invoke-Pip {
    param(
        [string]$PythonExe,
        [string[]]$Arguments
    )

    & $PythonExe -m pip @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "pip command failed: $Arguments"
    }
}

Assert-Administrator
Ensure-WingetAvailable

$detectedCuda = Get-CudaVersion
if ($detectedCuda) {
    if ($detectedCuda -ne $CudaVersion) {
        Write-Warning "Detected CUDA version $detectedCuda does not match the requested $CudaVersion. Ensure CUDA 13.0 toolkit and compatible drivers are installed."
    } else {
        Write-Host "Detected CUDA $detectedCuda" -ForegroundColor Green
    }
} else {
    Write-Warning "nvcc was not found on PATH. Install the CUDA $CudaVersion toolkit from NVIDIA before running GPU workloads."
}

Install-WingetPackage -Id "Microsoft.VCRedist.2015+.x64" -Description "Microsoft Visual C++ 2015-2022 Redistributable"
Install-WingetPackage -Id "Gyan.FFmpeg" -Description "FFmpeg"

$pythonExe = Ensure-Python -Version $PythonVersion
$venvPython = New-VirtualEnvironment -PythonExe $pythonExe -Directory $VenvDir

Write-Section "Upgrading pip and build tooling"
Invoke-Pip -PythonExe $venvPython -Arguments @("install", "--upgrade", "pip", "setuptools", "wheel")

Write-Section "Installing CUDA-enabled PyTorch"
$torchIndex = "https://download.pytorch.org/whl/cu121"
$torchPackages = @(
    "torch==2.3.1+cu121",
    "torchaudio==2.3.1+cu121",
    "torchvision==0.18.1+cu121"
)
Invoke-Pip -PythonExe $venvPython -Arguments @("install", "--index-url", $torchIndex, "--extra-index-url", "https://pypi.org/simple") + $torchPackages

Write-Section "Installing project requirements"
Invoke-Pip -PythonExe $venvPython -Arguments @("install", "-r", (Join-Path ((Get-Location).Path) "requirements.txt"))

Write-Section "Validating installation"
Invoke-Pip -PythonExe $venvPython -Arguments @("check")

Write-Section "Setup complete"
Write-Host "Activate the environment with:`n    $((Resolve-Path $VenvDir).Path)\\Scripts\\Activate.ps1" -ForegroundColor Green
Write-Host "Then launch the app with:`n    python app.py" -ForegroundColor Green
