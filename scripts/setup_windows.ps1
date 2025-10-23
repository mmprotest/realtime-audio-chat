[CmdletBinding()]
param(
    [switch]$SkipVenv
)

$ErrorActionPreference = 'Stop'

function Write-Section {
    param([string]$Message)
    Write-Host "`n=== $Message ===" -ForegroundColor Cyan
}

function Get-PythonExecutable {
    $candidates = @('python', 'py')
    foreach ($candidate in $candidates) {
        try {
            $command = Get-Command $candidate -ErrorAction Stop
            if ($command.Name -eq 'py') {
                $version = & $command.Source -3 --version 2>$null
                if ($LASTEXITCODE -eq 0) {
                    return "$($command.Source) -3"
                }
            } else {
                return $command.Source
            }
        } catch {
            continue
        }
    }
    throw 'Python 3.10+ is required but was not found on PATH. Install it from https://www.python.org/downloads/windows/ and re-run the script.'
}

function Get-PythonVersion {
    param([string]$PythonExe)
    $version = & $PythonExe -c "import sys; print('.'.join(map(str, sys.version_info[:3])))"
    return [version]$version
}

function Ensure-Python {
    $python = Get-PythonExecutable
    $version = Get-PythonVersion -PythonExe $python
    if ($version -lt [version]'3.10.0') {
        throw "Python 3.10 or newer is required. Found version $version."
    }
    return $python
}

function Ensure-Venv {
    param(
        [string]$PythonExe,
        [string]$RepoRoot
    )

    $venvPath = Join-Path $RepoRoot '.venv'
    $venvPython = Join-Path $venvPath 'Scripts\\python.exe'

    if (-not (Test-Path $venvPath)) {
        Write-Section "Creating virtual environment at $venvPath"
        & $PythonExe -m venv $venvPath
    } else {
        Write-Host "Virtual environment already exists at $venvPath" -ForegroundColor Yellow
    }

    if (-not (Test-Path $venvPython)) {
        throw "Virtual environment activation binary not found at $venvPython"
    }

    return $venvPython
}

function Install-PythonDependencies {
    param(
        [string]$VenvPython,
        [string]$RepoRoot
    )

    Write-Section 'Upgrading pip'
    & $VenvPython -m pip install --upgrade pip

    Write-Section 'Installing core requirements'
    & $VenvPython -m pip install -r (Join-Path $RepoRoot 'requirements.txt')

    Write-Section 'Installing optional STT/TTS extras'
    Push-Location $RepoRoot
    try {
        & $VenvPython -m pip install -e ".[stt_server,tts_server]"
    } finally {
        Pop-Location
    }
}

function Ensure-FFmpeg {
    param(
        [string]$RepoRoot
    )

    Write-Section 'Checking for FFmpeg'
    try {
        $ffmpegCmd = Get-Command ffmpeg -ErrorAction Stop
        $versionOutput = & $ffmpegCmd.Source -version 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "FFmpeg already available at $($ffmpegCmd.Source)" -ForegroundColor Green
            return
        }
    } catch {
        Write-Host 'FFmpeg not found on PATH. Installing bundled copy...' -ForegroundColor Yellow
    }

    $toolsDir = Join-Path $RepoRoot 'tools'
    if (-not (Test-Path $toolsDir)) {
        New-Item -ItemType Directory -Path $toolsDir | Out-Null
    }
    $ffmpegRoot = Join-Path $toolsDir 'ffmpeg'
    $ffmpegBin = Join-Path $ffmpegRoot 'bin'

    $downloadUrl = 'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip'
    $tempZip = Join-Path ([IO.Path]::GetTempPath()) ('ffmpeg-' + [guid]::NewGuid().ToString() + '.zip')
    $tempExtract = Join-Path ([IO.Path]::GetTempPath()) ('ffmpeg-' + [guid]::NewGuid().ToString())

    Write-Host "Downloading FFmpeg from $downloadUrl" -ForegroundColor Cyan
    Invoke-WebRequest -Uri $downloadUrl -OutFile $tempZip

    Write-Host 'Extracting FFmpeg archive' -ForegroundColor Cyan
    Expand-Archive -Path $tempZip -DestinationPath $tempExtract -Force
    $extracted = Get-ChildItem -Path $tempExtract | Where-Object { $_.PSIsContainer } | Select-Object -First 1
    if (-not $extracted) {
        throw 'FFmpeg archive structure unexpected; no directory found after extraction.'
    }

    if (Test-Path $ffmpegRoot) {
        Write-Host 'Removing existing FFmpeg directory' -ForegroundColor Yellow
        Remove-Item $ffmpegRoot -Recurse -Force
    }

    Move-Item $extracted.FullName $ffmpegRoot

    Remove-Item $tempZip -Force
    Remove-Item $tempExtract -Recurse -Force

    if (-not (Test-Path (Join-Path $ffmpegBin 'ffmpeg.exe'))) {
        throw "FFmpeg executable not found after extraction at $ffmpegBin"
    }

    Write-Host "FFmpeg installed to $ffmpegBin" -ForegroundColor Green

    $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    if (-not $userPath) { $userPath = '' }
    $pathEntries = $userPath -split ';' | Where-Object { $_ }
    if ($pathEntries -notcontains $ffmpegBin) {
        $newPath = if ($userPath -and $userPath.Trim()) { $userPath.TrimEnd(';') + ';' + $ffmpegBin } else { $ffmpegBin }
        [Environment]::SetEnvironmentVariable('Path', $newPath, 'User')
        Write-Host "Added $ffmpegBin to the user PATH. You may need to restart your terminal for changes to take effect." -ForegroundColor Yellow
    } else {
        Write-Host 'FFmpeg bin directory already present in user PATH.' -ForegroundColor Green
    }
}

function Show-NextSteps {
    param([string]$RepoRoot)

    Write-Section 'Setup complete'

    Write-Host "To start using the virtual environment, run:" -ForegroundColor Cyan
    Write-Host '    ".\.venv\Scripts\Activate.ps1"' -ForegroundColor Yellow
    Write-Host 'After activation you can launch the app with:' -ForegroundColor Cyan
    Write-Host '    python -m src.app' -ForegroundColor Yellow
    Write-Host 'If PATH was updated for FFmpeg, open a new terminal session to pick up the change.' -ForegroundColor Cyan
}

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptRoot '..')
Set-Location $repoRoot

$pythonExe = Ensure-Python
$venvPython = $pythonExe

if (-not $SkipVenv) {
    $venvPython = Ensure-Venv -PythonExe $pythonExe -RepoRoot $repoRoot
} else {
    Write-Host 'Skipping virtual environment creation as requested.' -ForegroundColor Yellow
    $venvPython = $pythonExe
}

Install-PythonDependencies -VenvPython $venvPython -RepoRoot $repoRoot
Ensure-FFmpeg -RepoRoot $repoRoot
Show-NextSteps -RepoRoot $repoRoot
