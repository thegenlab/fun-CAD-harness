#!/usr/bin/env bash
# Create the FunCAD-Harness conda environments (funcad-cad, funcad-chrono) and
# record their interpreters in .env for scripts/funcad-gui.
# Works with micromamba, mamba or conda.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Packages in ~/.local would otherwise satisfy pip requirements and leak into the environments.
export PYTHONNOUSERSITE=1

if command -v micromamba >/dev/null 2>&1; then TOOL=micromamba
elif command -v mamba >/dev/null 2>&1; then TOOL=mamba
elif command -v conda >/dev/null 2>&1; then TOOL=conda
else
  echo "No conda installation found. Install Miniforge (https://conda-forge.org/download/) or micromamba first." >&2
  exit 1
fi

exists() { "$TOOL" env list | awk '{print $1}' | grep -qx "$1"; }
for name in cad chrono; do
  env="funcad-$name"
  if exists "$env"; then
    echo "Environment $env already exists; remove it to recreate it."
  elif [ "$TOOL" = micromamba ]; then
    micromamba create -y -f "$ROOT/environment-$name.yml"
  else
    "$TOOL" env create -f "$ROOT/environment-$name.yml"
  fi
done

interpreter() { "$TOOL" run -n "$1" python -c "import sys; print(sys.executable)"; }
CAD_PYTHON="$(interpreter funcad-cad)"
CHRONO_PYTHON="$(interpreter funcad-chrono)"
CAD_LIB="$(dirname "$(dirname "$CAD_PYTHON")")/lib"

{
  echo "# Written by scripts/setup_conda.sh"
  printf 'SIMREADY_CAD_PYTHON=%q\n' "$CAD_PYTHON"
  printf 'SIMREADY_PYCHRONO_EXECUTABLE=%q\n' "$CHRONO_PYTHON"
  echo "PYTHONNOUSERSITE=1"
  echo "SIMREADY_RENDER_LANG=en"
  if [ "$(uname -s)" = Linux ]; then
    # The OCP wheel must load the environment's libexpat, not an older system copy.
    printf 'LD_LIBRARY_PATH=%q\n' "$CAD_LIB"
  fi
} > "$ROOT/.env"

LD_LIBRARY_PATH="$CAD_LIB" "$CAD_PYTHON" -c "import build123d, OCP, trimesh, coacd, fcl; print('CAD environment ok')"
"$CHRONO_PYTHON" -c "import pychrono; print('PyChrono environment ok')"
echo "Wrote $ROOT/.env. Start the GUI with: scripts/funcad-gui"
