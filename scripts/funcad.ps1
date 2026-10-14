# Run the FunCAD-Harness command line on Windows with the environments from .env.
#   scripts\funcad.ps1 create --mode free --project folding_stool --task examples\tasks\folding_stool.md
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$envFile = Join-Path $Root '.env'
if (-not (Test-Path $envFile)) { throw 'Run scripts\setup_conda.ps1 first (or use Docker, see README).' }
Get-Content $envFile | Where-Object { $_ -match '^[A-Za-z_][A-Za-z0-9_]*=' } | ForEach-Object {
    $name, $value = $_ -split '=', 2
    Set-Item -Path "env:$name" -Value $value
}
$env:PATH = (Split-Path -Parent $env:SIMREADY_CAD_PYTHON) + ';' + $env:PATH
Set-Location $Root
& $env:SIMREADY_CAD_PYTHON -X utf8 -m simulation_ready_cad_workflow.validation_pipeline.slim_harness @args
exit $LASTEXITCODE
