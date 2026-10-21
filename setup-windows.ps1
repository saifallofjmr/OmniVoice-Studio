[CmdletBinding()]
param(
    [switch]$SkipVoiceMeeter,
    [switch]$SkipModelDownloads
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$Arguments
    )

    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$FilePath failed with exit code $LASTEXITCODE"
    }
}

function Ensure-WingetPackage {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Id
    )

    winget install `
        --id $Id `
        --exact `
        --accept-package-agreements `
        --accept-source-agreements `
        --silent `
        --disable-interactivity

    if ($LASTEXITCODE -ne 0) {
        throw "Could not install $Id with winget."
    }
}

if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
    throw "Windows Package Manager (winget) is required."
}

$python312 = Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe"
$python310 = Join-Path $env:LOCALAPPDATA "Programs\Python\Python310\python.exe"

if (-not (Test-Path $python312)) {
    Ensure-WingetPackage "Python.Python.3.12"
}
if (-not (Test-Path $python310)) {
    Ensure-WingetPackage "Python.Python.3.10"
}
if (-not (Test-Path $python312) -or -not (Test-Path $python310)) {
    throw "Python installation completed but the expected executables were not found."
}

$git = (Get-Command git -ErrorAction SilentlyContinue).Source
if (-not $git) {
    Ensure-WingetPackage "Git.Git"
    $git = "C:\Program Files\Git\cmd\git.exe"
}
if (-not (Test-Path $git)) {
    throw "Git is required but was not found."
}

Ensure-WingetPackage "Gyan.FFmpeg"
if (-not $SkipVoiceMeeter) {
    Ensure-WingetPackage "VB-Audio.Voicemeeter"
}

Write-Host "Creating OmniVoice environment..."
Invoke-Checked $python312 -m venv .venv
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
Invoke-Checked $python -m pip install --upgrade pip
Invoke-Checked $python -m pip install `
    "torch==2.8.0+cu128" `
    "torchaudio==2.8.0+cu128" `
    --extra-index-url "https://download.pytorch.org/whl/cu128"
Invoke-Checked $python -m pip install -e ".[capture]"
Invoke-Checked $python -m pip check

if (-not $SkipModelDownloads) {
    Write-Host "Downloading the public OmniVoice checkpoint..."
    $downloadModel = @'
from huggingface_hub import snapshot_download
snapshot_download(repo_id="k2-fsa/OmniVoice", local_dir="local/OmniVoice")
'@
    $downloadModel | & $python -
    if ($LASTEXITCODE -ne 0) {
        throw "OmniVoice model download failed."
    }
}

$seedVcDirectory = Join-Path $env:USERPROFILE "SeedVC"
$seedVcCommit = "58d0ca372cc76f60a798d9f84611f053f9029f3a"

Write-Host "Installing isolated Seed-VC realtime engine..."
if (-not (Test-Path (Join-Path $seedVcDirectory ".git"))) {
    Invoke-Checked $git clone `
        "https://github.com/jiaheguo521/seed-vc-realtime.git" `
        $seedVcDirectory
}
Invoke-Checked $git -C $seedVcDirectory fetch origin
Invoke-Checked $git -C $seedVcDirectory checkout --force $seedVcCommit
$seedVcPatch = Join-Path $PSScriptRoot "seed-vc-realtime.patch"
if (Test-Path $seedVcPatch) {
    Invoke-Checked $git -C $seedVcDirectory apply --check $seedVcPatch
    Invoke-Checked $git -C $seedVcDirectory apply $seedVcPatch
}

$seedPython = Join-Path $seedVcDirectory ".venv\Scripts\python.exe"
Invoke-Checked $python310 -m venv (Join-Path $seedVcDirectory ".venv")
Invoke-Checked $seedPython -m pip install --upgrade pip
Invoke-Checked $seedPython -m pip install `
    "torch==2.8.0+cu128" `
    "torchvision==0.23.0+cu128" `
    "torchaudio==2.8.0+cu128" `
    --extra-index-url "https://download.pytorch.org/whl/cu128"

$filteredRequirements = Join-Path $env:TEMP "seed-vc-requirements.txt"
Get-Content (Join-Path $seedVcDirectory "requirements.txt") |
    Where-Object { $_ -notmatch "^torch(vision|audio)?==" } |
    Set-Content -LiteralPath $filteredRequirements -Encoding utf8
Invoke-Checked $seedPython -m pip install -r $filteredRequirements
Invoke-Checked $seedPython -m pip install "triton-windows==3.4.0.post20"
Invoke-Checked $seedPython -m pip check

New-Item `
    -ItemType Directory `
    -Path (Join-Path $seedVcDirectory "references") `
    -Force | Out-Null

if (-not $SkipModelDownloads) {
    Write-Host "Downloading Seed-VC realtime model files..."
    $initializeSeedVc = @'
from types import SimpleNamespace
from realtime_vc_engine import RealtimeVCEngine

RealtimeVCEngine(
    SimpleNamespace(
        checkpoint_path=None,
        config_path=None,
        gpu=0,
        fp16=True,
    )
)
print("Seed-VC models are ready.")
'@
    Push-Location $seedVcDirectory
    try {
        $initializeSeedVc | & $seedPython -
        if ($LASTEXITCODE -ne 0) {
            throw "Seed-VC model initialization failed."
        }
    }
    finally {
        Pop-Location
    }
}

Write-Host ""
Write-Host "Setup complete."
Write-Host "If VoiceMeeter was installed for the first time, restart Windows before use."
Write-Host "Launch with: .\launch-studio.cmd"
