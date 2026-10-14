# Create the FunCAD-Harness conda environments (funcad-cad, funcad-chrono) on Windows and
# record their interpreters in .env for scripts\funcad-gui.ps1.
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$env:PYTHONNOUSERSITE = '1'

$Tool = @('micromamba', 'mamba', 'conda') | Where-Object { Get-Command $_ -ErrorAction SilentlyContinue } | Select-Object -First 1
if (-not $Tool) { throw 'No conda installation found. Install Miniforge (https://conda-forge.org/download/) or micromamba first.' }

$existing = & $Tool env list | ForEach-Object { ($_ -split '\s+')[0] }
foreach ($name in 'cad', 'chrono') {
    $envName = "funcad-$name"
    $file = Join-Path $Root "environment-$name.yml"
    if ($existing -contains $envName) { Write-Host "Environment $envName already exists; remove it to recreate it." }
    elseif ($Tool -eq 'micromamba') { & micromamba create -y -f $file }
    else { & $Tool env create -f $file }
}

function Get-Interpreter($envName) { (& $Tool run -n $envName python -c "import sys; print(sys.executable)").Trim() }
$cad = Get-Interpreter 'funcad-cad'
$chrono = Get-Interpreter 'funcad-chrono'
@(
    '# Written by scripts/setup_conda.ps1'
    "SIMREADY_CAD_PYTHON=$cad"
    "SIMREADY_PYCHRONO_EXECUTABLE=$chrono"
    'PYTHONNOUSERSITE=1'
    'SIMREADY_RENDER_LANG=en'
) | Set-Content -Encoding utf8 (Join-Path $Root '.env')

& $cad -c "import build123d, OCP, trimesh, coacd, fcl; print('CAD environment ok')"
& $chrono -c "import pychrono; print('PyChrono environment ok')"
Write-Host "Wrote $Root\.env. Start the GUI with: scripts\funcad-gui.ps1"
