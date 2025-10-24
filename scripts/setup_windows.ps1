<#
.SYNOPSIS
    End-to-end setup script for configuring the realtime-audio-chat project on Windows with CUDA-enabled PyTorch.

.DESCRIPTION
    This script bootstraps a Windows development environment that targets CUDA 13.0-capable GPUs.
    It performs the following steps:
        * Verifies administrative privileges and the presence of CUDA 13.0 toolkits/drivers.
        * Installs system-level dependencies (Python, Visual C++ runtimes, FFmpeg) by downloading the official installers when missing.
        * Creates an isolated Python virtual environment for the project.
        * Installs CUDA-enabled PyTorch wheels (cu121 build, compatible with CUDA 13 drivers) and the
          Python dependencies required by the application.
        * Validates the resulting Python environment with `pip check` and prints activation guidance.

    Run this script from the repository root using an elevated Windows PowerShell session:

        powershell.exe -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1

.PARAMETER PythonVersion
    Full Python version (major.minor.patch) to install and target for the virtual environment.
    The default (3.10.14) is compatible with all project dependencies.

.PARAMETER CudaVersion
    CUDA toolkit version expected to be available on the host. Defaults to 13.0.
    PyTorch wheels are installed from the cu121 channel which is forward compatible with CUDA 13 drivers.

.PARAMETER VenvDir
    Directory (relative to the repository root) where the Python virtual environment will be created.

.NOTES
    Requires Windows 10/11 with administrative privileges. The script downloads installers from their
    official distribution points, so an active internet connection is necessary.
#>
param(
    [string]$PythonVersion = "3.10.14",
    [string]$CudaVersion = "13.0",
    [string]$VenvDir = ".venv"
)

$ErrorActionPreference = "Stop"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

function Write-Section {
    param([string]$Message)
    Write-Host "`n==== $Message ====\n" -ForegroundColor Cyan
}

function Assert-Administrator {
    $currentIdentity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentIdentity)
    if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        Write-Warning "Administrative privileges are required. Requesting elevation..."

        try {
            $argumentList = New-Object System.Collections.Generic.List[string]
            $argumentList.Add("-NoProfile") | Out-Null
            $argumentList.Add("-ExecutionPolicy") | Out-Null
            $argumentList.Add("Bypass") | Out-Null
            $argumentList.Add("-File") | Out-Null
            $argumentList.Add($PSCommandPath) | Out-Null

            foreach ($entry in $PSBoundParameters.GetEnumerator()) {
                if ($null -ne $entry.Value -and $entry.Value -ne "") {
                    $argumentList.Add("-{0}" -f $entry.Key) | Out-Null
                    $argumentList.Add([string]$entry.Value) | Out-Null
                }
            }

            $startInfo = @{
                FilePath        = "powershell.exe"
                ArgumentList    = $argumentList
                Verb            = "RunAs"
                WorkingDirectory = (Get-Location).Path
                PassThru        = $true
                Wait            = $true
            }

            $process = Start-Process @startInfo
            if ($process -and $process.HasExited) {
                exit $process.ExitCode
            } else {
                exit
            }
        } catch {
            throw "Administrative privileges are required. Please rerun this script from an elevated PowerShell session."
        }
    }
}

function Get-MajorMinorVersionString {
    param([string]$Version)

    try {
        $parsed = [System.Version]::Parse($Version)
        return "{0}.{1}" -f $parsed.Major, $parsed.Minor
    } catch {
        throw "PythonVersion must be a valid semantic version (e.g. 3.10.11)."
    }
}

function Download-File {
    param(
        [Parameter(Mandatory=$true)][string]$Uri,
        [Parameter(Mandatory=$true)][string]$Description
    )

    $destination = Join-Path ([System.IO.Path]::GetTempPath()) (Split-Path -Path $Uri -Leaf)
    Write-Section "Downloading $Description"
    Invoke-WebRequest -Uri $Uri -OutFile $destination -UseBasicParsing | Out-Null
    return $destination
}

function Add-ToSystemPath {
    param([string]$Directory)

    if (-not (Test-Path $Directory)) {
        throw "Cannot add missing directory '$Directory' to PATH."
    }

    $machinePath = [Environment]::GetEnvironmentVariable("Path", [EnvironmentVariableTarget]::Machine)
    $pathEntries = @()
    if ($machinePath) {
        $pathEntries = $machinePath.Split(';') | Where-Object { $_ }
    }

    if ($pathEntries -contains $Directory) {
        return
    }

    $newPath = if ($machinePath) { "$machinePath;$Directory" } else { $Directory }
    [Environment]::SetEnvironmentVariable("Path", $newPath, [EnvironmentVariableTarget]::Machine)
    $env:Path = "$Directory;" + $env:Path
}

function Test-UriAvailable {
    param([string]$Uri)

    try {
        $response = Invoke-WebRequest -Uri $Uri -Method Head -UseBasicParsing -MaximumRedirection 0 -ErrorAction Stop
        return $response.StatusCode -ge 200 -and $response.StatusCode -lt 400
    } catch [System.Net.WebException] {
        $webResponse = $_.Exception.Response
        if ($webResponse -and $webResponse.StatusCode) {
            $status = [int]$webResponse.StatusCode
            if ($status -eq 404) {
                return $false
            }
        }
        throw
    }
}

function Get-LatestPythonPatchVersion {
    param([string]$MajorMinor)

    $indexUri = "https://www.python.org/ftp/python/$MajorMinor/"
    try {
        $response = Invoke-WebRequest -Uri $indexUri -UseBasicParsing -ErrorAction Stop
    } catch {
        throw "Unable to query available Python releases for $MajorMinor from $indexUri"
    }

    $pattern = "python-$MajorMinor\.([0-9]+)-amd64\.exe"
    $matches = [regex]::Matches($response.Content, $pattern)
    if ($matches.Count -eq 0) {
        throw "No patch releases were found for Python $MajorMinor on python.org"
    }

    $latestPatch = ($matches | ForEach-Object { [int]$_.Groups[1].Value } | Sort-Object -Descending | Select-Object -First 1)
    return "{0}.{1}" -f $MajorMinor, $latestPatch
}

function Resolve-PythonInstaller {
    param([string]$RequestedVersion)

    $candidateUri = "https://www.python.org/ftp/python/$RequestedVersion/python-$RequestedVersion-amd64.exe"
    if (Test-UriAvailable -Uri $candidateUri) {
        return [PSCustomObject]@{
            Version = $RequestedVersion
            Uri     = $candidateUri
        }
    }

    $majorMinor = Get-MajorMinorVersionString -Version $RequestedVersion
    $fallbackVersion = Get-LatestPythonPatchVersion -MajorMinor $majorMinor
    $fallbackUri = "https://www.python.org/ftp/python/$fallbackVersion/python-$fallbackVersion-amd64.exe"

    if (-not (Test-UriAvailable -Uri $fallbackUri)) {
        throw "Unable to locate a downloadable Python installer for version $RequestedVersion or fallback $fallbackVersion"
    }

    Write-Warning "Python $RequestedVersion installer was not found. Falling back to latest $majorMinor patch release ($fallbackVersion)."

    return [PSCustomObject]@{
        Version = $fallbackVersion
        Uri     = $fallbackUri
    }
}

function Install-Python {
    param(
        [string]$Version,
        [string]$InstallerUri
    )

    $installerPath = Download-File -Uri $InstallerUri -Description "Python $Version installer"

    try {
        Write-Section "Installing Python $Version"
        & $installerPath /quiet InstallAllUsers=1 PrependPath=1 Include_test=0 SimpleInstall=1
        if ($LASTEXITCODE -ne 0) {
            throw "Python installer exited with code $LASTEXITCODE"
        }
    } finally {
        if (Test-Path $installerPath) {
            Remove-Item $installerPath -Force
        }
    }
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
    param(
        [string]$Version,
        [switch]$AllowPatchMismatch
    )

    $majorMinor = Get-MajorMinorVersionString -Version $Version

    $candidates = @()

    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        $pyResult = & $pyLauncher.Source -$majorMinor -c "import sys; print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $pyResult) {
            $candidates += $pyResult.Trim()
        }
    }

    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCmd) {
        $candidates += $pythonCmd.Source
    }

    foreach ($candidate in $candidates | Select-Object -Unique) {
        if (-not (Test-Path $candidate)) {
            continue
        }

        $detectedVersion = Get-PythonVersionString -PythonExe $candidate
        if (-not $detectedVersion) {
            continue
        }

        if ($detectedVersion -eq $Version -or ($AllowPatchMismatch -and $detectedVersion.StartsWith("$majorMinor."))) {
            return [PSCustomObject]@{
                Path    = $candidate
                Version = $detectedVersion
            }
        }
    }

    return $null
}

function Get-PythonVersionString {
    param([string]$PythonExe)

    $result = & $PythonExe -c "import platform; print(platform.python_version())" 2>$null
    if ($LASTEXITCODE -eq 0 -and $result) {
        return $result.Trim()
    }
    return $null
}

function Ensure-Python {
    param([string]$Version)

    $existing = Resolve-PythonExecutable -Version $Version -AllowPatchMismatch
    if ($existing) {
        if ($existing.Version -eq $Version) {
            Write-Host "Found Python $($existing.Version) at $($existing.Path)" -ForegroundColor Green
            return $existing.Path
        }

        Write-Warning "Python $($existing.Version) is installed at $($existing.Path), which satisfies the requested $Version major/minor version. Using the existing interpreter."
        return $existing.Path
    }

    $installerInfo = Resolve-PythonInstaller -RequestedVersion $Version
    Install-Python -Version $installerInfo.Version -InstallerUri $installerInfo.Uri

    $postInstall = Resolve-PythonExecutable -Version $installerInfo.Version
    if (-not $postInstall) {
        throw "Python $($installerInfo.Version) installation was requested but the interpreter could not be located."
    }

    if ($postInstall.Version -ne $installerInfo.Version) {
        throw "Python $($installerInfo.Version) installation verification failed (detected $($postInstall.Version))."
    }

    Write-Host "Installed Python $($postInstall.Version) at $($postInstall.Path)" -ForegroundColor Green
    return $postInstall.Path
}

function Test-VCRedistributableInstalled {
    $vcKey = Get-ItemProperty -Path "HKLM:\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" -ErrorAction SilentlyContinue
    if ($vcKey -and $vcKey.Installed -eq 1) {
        return $true
    }
    return $false
}

function Ensure-VCRedistributable {
    if (Test-VCRedistributableInstalled) {
        Write-Host "Microsoft Visual C++ Redistributable already installed." -ForegroundColor Green
        return
    }

    $vcUrl = "https://aka.ms/vs/17/release/vc_redist.x64.exe"
    $vcInstaller = Download-File -Uri $vcUrl -Description "Microsoft Visual C++ 2015-2022 Redistributable"

    try {
        Write-Section "Installing Microsoft Visual C++ Redistributable"
        & $vcInstaller /install /quiet /norestart
        if ($LASTEXITCODE -ne 0) {
            throw "VC++ redistributable installer exited with code $LASTEXITCODE"
        }
    } finally {
        if (Test-Path $vcInstaller) {
            Remove-Item $vcInstaller -Force
        }
    }

    if (-not (Test-VCRedistributableInstalled)) {
        throw "Microsoft Visual C++ Redistributable installation could not be verified."
    }
}

function Ensure-FFmpeg {
    if (Get-Command ffmpeg -ErrorAction SilentlyContinue) {
        Write-Host "FFmpeg is already available on PATH." -ForegroundColor Green
        return
    }

    $ffmpegUrl = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
    $ffmpegArchive = Download-File -Uri $ffmpegUrl -Description "FFmpeg archive"
    $extractionDir = Join-Path ([System.IO.Path]::GetTempPath()) ([System.IO.Path]::GetRandomFileName())
    New-Item -ItemType Directory -Path $extractionDir | Out-Null

    try {
        Write-Section "Installing FFmpeg"
        Expand-Archive -Path $ffmpegArchive -DestinationPath $extractionDir -Force
        $extractedRoot = Get-ChildItem -Path $extractionDir -Directory | Select-Object -First 1
        if (-not $extractedRoot) {
            throw "FFmpeg archive extraction failed."
        }

        $programFiles = if ($env:ProgramFiles) { $env:ProgramFiles } else { "C:\\Program Files" }
        $installRoot = Join-Path $programFiles "ffmpeg"
        if (Test-Path $installRoot) {
            Remove-Item $installRoot -Recurse -Force
        }

        Move-Item -Path $extractedRoot.FullName -Destination $installRoot
        $ffmpegBin = Join-Path $installRoot "bin"
        if (-not (Test-Path (Join-Path $ffmpegBin "ffmpeg.exe"))) {
            throw "FFmpeg binary not found at $ffmpegBin after installation."
        }

        Add-ToSystemPath -Directory $ffmpegBin
        Write-Host "FFmpeg installed to $ffmpegBin" -ForegroundColor Green
    } finally {
        if (Test-Path $ffmpegArchive) {
            Remove-Item $ffmpegArchive -Force
        }
        if (Test-Path $extractionDir) {
            Remove-Item $extractionDir -Recurse -Force
        }
    }
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

Ensure-VCRedistributable
Ensure-FFmpeg

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
$repoRoot = (Get-Location).Path
$requirementsPath = Join-Path $repoRoot "requirements.txt"
Invoke-Pip -PythonExe $venvPython -Arguments @("install", "-r", $requirementsPath)

Write-Section "Validating base installation"
try {
    Invoke-Pip -PythonExe $venvPython -Arguments @("check")
} catch {
    Write-Warning "Base dependency validation reported issues: $($_.Exception.Message)"
}

$fishRequirementsPath = Join-Path $repoRoot "requirements-fish-speech.txt"
if (Test-Path $fishRequirementsPath) {
    Write-Section "Installing Fish-Speech runtime (ignoring upstream dependency pins)"
    Invoke-Pip -PythonExe $venvPython -Arguments @("install", "--no-deps", "-r", $fishRequirementsPath)
    Write-Warning "Fish-Speech is installed without its declared dependency constraints. Upstream metadata currently conflicts with FastRTC's requirements (e.g. numpy/gradio versions)."
}

Write-Section "Setup complete"
Write-Host "Activate the environment with:`n    $((Resolve-Path $VenvDir).Path)\\Scripts\\Activate.ps1" -ForegroundColor Green
Write-Host "Then launch the app with:`n    python app.py" -ForegroundColor Green
