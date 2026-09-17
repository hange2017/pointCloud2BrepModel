# =====================================================================
#  p2b environment bootstrap (Windows / conda)
#  Reproduces the verified V1+CUDA dev environment.
#
#  Verified on: Windows 10 19045, GTX 1080 (sm_61), driver 560.94
#  Result      : Python 3.11.16, torch 2.5.1+cu121, CUDA 12.1 runtime
#
#  NOTE (ASCII only on purpose): keep this file ASCII. Windows
#  PowerShell 5.1 reads .ps1 as GBK when there is no BOM, so non-ASCII
#  characters here would break parsing.
#
#  Usage:
#     powershell -ExecutionPolicy Bypass -File envs\setup-p2b.ps1
#  Options:
#     -SkipTorch   install the pure-CPU geometry stack only
#     -EnvName X   use another conda env name (default: p2b)
# =====================================================================
param(
    [switch]$SkipTorch,
    [string]$EnvName = "p2b"
)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"

# --- Tunables ---------------------------------------------------------
$PyVersion   = "3.11"
$TunaIndex   = "http://pypi.tuna.tsinghua.edu.cn/simple"
$TorchProxy  = "http://127.0.0.1:51370"      # Lantern HTTP proxy (see below)
$TorchMirror = "https://mirror.sjtu.edu.cn/pytorch-wheels/cu121"
$WheelDir    = "E:\pip-cache\wheels"
$TorchVer    = "torch-2.5.1%2Bcu121-cp311-cp311-win_amd64.whl"
$TvVer       = "torchvision-0.20.1%2Bcu121-cp311-cp311-win_amd64.whl"
# ----------------------------------------------------------------------

function Section($t) { Write-Host "`n=== $t ===" -ForegroundColor Cyan }

Section "0) Preconditions"
$conda = Get-Command conda -ErrorAction SilentlyContinue
if (-not $conda) { throw "conda not found in PATH. Install Anaconda/Miniconda first." }
Write-Host "conda: $($conda.Source)"

# Hard check: this machine MUST stay on PyTorch <= 2.5.1 for Pascal.
# torch 2.8+ (cu128) dropped sm_60/sm_61, so newer wheels will NOT run here.
Write-Host "NOTE: Pascal (sm_61) support ends at torch 2.5.1. Do not upgrade." -ForegroundColor Yellow

Section "1) Create conda env '$EnvName' (python $PyVersion)"
$exists = (conda env list) -match "^$EnvName\s"
if ($exists) {
    Write-Host "env '$EnvName' already exists, reusing."
} else {
    conda create -y -n $EnvName "python=$PyVersion"
}

Section "2) Base numeric + geometry stack (Tsinghua mirror, no proxy)"
$req = @("numpy","scipy","scikit-learn","networkx","h5py","pytest","matplotlib","pandas")
conda run --no-capture-output -n $EnvName python -m pip install -U pip setuptools wheel
conda run --no-capture-output -n $EnvName python -m pip install @req -i $TunaIndex

Section "3) Open3D + CadQuery/OCCT"
conda run --no-capture-output -n $EnvName python -m pip install "open3d==0.20.0" -i $TunaIndex
conda run --no-capture-output -n $EnvName python -m pip install "cadquery==2.8.0" -i $TunaIndex

if (-not $SkipTorch) {
    Section "4) PyTorch cu121 for Pascal (download from SJTU mirror, then install offline)"
    # Why download-then-install instead of plain pip install:
    #   - download.pytorch.org through the VPN proxy measured ~0.08 MB/s (stalled),
    #     while the SJTU mirror measured ~38-70 MB/s for the same 2.4 GB wheel.
    #   - A local wheel makes the install reproducible and re-runnable offline.
    New-Item -ItemType Directory -Force -Path $WheelDir | Out-Null
    foreach ($f in @($TorchVer, $TvVer)) {
        $out = Join-Path $WheelDir ($f -replace "%2B", "+")
        if ((Test-Path $out) -and ((Get-Item $out).Length -gt 100MB)) {
            Write-Host "already downloaded: $out"
            continue
        }
        Write-Host "downloading $f ..."
        curl.exe -L --noproxy "*" --retry 3 --retry-delay 2 -C - -o $out "$TorchMirror/$f"
    }
    # Verify the wheels are intact before installing.
    conda run --no-capture-output -n $EnvName python -c "import zipfile,sys;[print(p,'OK' if zipfile.is_zipfile(p) else 'CORRUPT') for p in sys.argv[1:]]" `
        (Join-Path $WheelDir ($TorchVer -replace "%2B","+")) `
        (Join-Path $WheelDir ($TvVer -replace "%2B","+"))

    conda run --no-capture-output -n $EnvName python -m pip install --no-index --no-deps `
        (Join-Path $WheelDir ($TorchVer -replace "%2B","+")) `
        (Join-Path $WheelDir ($TvVer -replace "%2B","+"))

    Section "5) torch runtime deps (pinned: torch needs sympy==1.13.1)"
    conda run --no-capture-output -n $EnvName python -m pip install `
        "sympy==1.13.1" networkx jinja2 fsspec "typing-extensions" filelock -i $TunaIndex
}

Section "6) Verify"
conda run --no-capture-output -n $EnvName python envs\verify_env.py
Write-Host "`nDone. Activate with:  conda activate $EnvName" -ForegroundColor Green
