"""Per-machine CAD/physics launch recipe.

The locked sibling environments are the same on every computer: CAD from
``environment-cad-linux64.yml`` and physics from
``environment-pychrono-linux64.yml``. Absolute prefixes differ per machine.
This module discovers or creates those two environments, probes the documented
health imports, and writes ``.runtime_runs/machine_launch.json`` so later
Harness starts do not guess an interpreter from ``PATH``.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Callable, Mapping

from ..mechanism.io import read_json, write_json_atomic
from ..mechanism.launcher import PyChronoLauncher, resolve_pychrono_launcher


MACHINE_LAUNCH_SCHEMA = "simulation_ready_cad_machine_launch_v1"
RECIPE_RELATIVE = ".runtime_runs/machine_launch.json"
MAMBA_ROOT_ENV = "SIMREADY_MAMBA_ROOT"
CAD_ENV_NAME = "sr-cad-py312"
CHRONO_ENV_NAME = "sr-chrono-py312"
CAD_IMPORTS = ("build123d", "OCP", "fcl", "matplotlib", "psutil", "trimesh")
CHRONO_IMPORTS = ("pychrono",)

_PIPELINE_ROOT = Path(__file__).resolve().parents[1]
CAD_LOCK = _PIPELINE_ROOT / "environment-cad-linux64.yml"
CHRONO_LOCK = _PIPELINE_ROOT / "environment-pychrono-linux64.yml"


class MachineLaunchError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


def recipe_path(workspace_root: str | Path) -> Path:
    return Path(workspace_root).resolve() / RECIPE_RELATIVE


def load_machine_launch(workspace_root: str | Path) -> dict[str, Any] | None:
    path = recipe_path(workspace_root)
    if not path.is_file():
        return None
    return _validate_recipe(read_json(path, max_bytes=64 * 1024), require_files=True)


def pychrono_launcher_from_recipe(recipe: Mapping[str, Any]) -> PyChronoLauncher:
    environment = recipe.get("pychrono_environment")
    if environment is not None:
        environment = str(environment)
    return resolve_pychrono_launcher(
        str(recipe["pychrono_launcher"]),
        executable=str(recipe["pychrono_executable"]),
        environment=environment,
    )


def kernel_spawn_environment(
    recipe: Mapping[str, Any],
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    env = dict(os.environ if environ is None else environ)
    lib = str(recipe["cad_library_directory"])
    existing = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = lib if not existing else f"{lib}{os.pathsep}{existing}"
    env["PYTHONNOUSERSITE"] = "1"
    return env


def ensure_machine_launch(
    *,
    workspace_root: str | Path,
    install_if_missing: bool = True,
    environ: Mapping[str, str] | None = None,
    mamba_root: str | Path | None = None,
    probe_imports: Callable[..., str | None] | None = None,
    install_lock: Callable[..., None] | None = None,
) -> dict[str, Any]:
    """Return the workspace recipe, creating the locked envs on first use."""

    workspace = Path(workspace_root).resolve()
    values = os.environ if environ is None else environ
    probe = probe_imports or _probe_imports
    installer = install_lock or _install_lock
    existing = None
    path = recipe_path(workspace)
    if path.is_file():
        try:
            existing = _validate_recipe(
                read_json(path, max_bytes=64 * 1024), require_files=True
            )
        except MachineLaunchError:
            existing = None
        else:
            cad_error = probe(
                Path(existing["cad_python"]),
                modules=CAD_IMPORTS,
                ld_library_path=str(existing["cad_library_directory"]),
            )
            chrono_error = probe(
                Path(existing["pychrono_executable"]),
                modules=CHRONO_IMPORTS,
                ld_library_path=None,
            )
            if cad_error is None and chrono_error is None:
                return existing
            if not install_if_missing:
                raise MachineLaunchError(
                    "MACHINE_LAUNCH_PROBE_FAILED",
                    _probe_message(cad_error, chrono_error),
                )
            installer(
                mamba_root=Path(existing["mamba_root"]),
                cad_prefix=Path(existing["cad_environment"]),
                chrono_prefix=Path(existing["pychrono_environment_prefix"]),
                cad_exists=True,
                chrono_exists=True,
            )
            _require_healthy(existing, probe=probe)
            write_json_atomic(path, existing)
            return existing

    root = _mamba_root(values, mamba_root)
    cad_prefix = root / "envs" / CAD_ENV_NAME
    chrono_prefix = root / "envs" / CHRONO_ENV_NAME
    cad_exists = _python_in(cad_prefix) is not None
    chrono_exists = _python_in(chrono_prefix) is not None
    if not cad_exists or not chrono_exists:
        if not install_if_missing:
            raise MachineLaunchError(
                "MACHINE_LAUNCH_ENV_MISSING",
                "this computer has no frozen CAD/physics launch recipe and "
                f"the locked environments are not at {root}",
            )
        installer(
            mamba_root=root,
            cad_prefix=cad_prefix,
            chrono_prefix=chrono_prefix,
            cad_exists=cad_exists,
            chrono_exists=chrono_exists,
        )
    recipe = _recipe_from_prefixes(root, cad_prefix, chrono_prefix)
    _require_healthy(recipe, probe=probe)
    write_json_atomic(path, recipe)
    return recipe


def _mamba_root(
    values: Mapping[str, str], override: str | Path | None
) -> Path:
    if override is not None:
        return Path(override).expanduser().resolve()
    configured = values.get(MAMBA_ROOT_ENV)
    if configured and str(configured).strip():
        return Path(configured).expanduser().resolve()
    return (Path.home() / ".micromamba-simulation-ready").resolve()


def _python_in(prefix: Path) -> Path | None:
    for candidate in (
        prefix / "bin" / "python",
        prefix / "python.exe",
        prefix / "Scripts" / "python.exe",
    ):
        if candidate.is_file():
            return candidate
    return None


def _recipe_from_prefixes(
    mamba_root: Path, cad_prefix: Path, chrono_prefix: Path
) -> dict[str, Any]:
    cad_python = _python_in(cad_prefix)
    chrono_python = _python_in(chrono_prefix)
    if cad_python is None or chrono_python is None:
        raise MachineLaunchError(
            "MACHINE_LAUNCH_ENV_MISSING",
            "the locked CAD or physics environment has no Python interpreter",
        )
    cad_lib = cad_prefix / "lib"
    return {
        "schema_version": MACHINE_LAUNCH_SCHEMA,
        "mamba_root": str(mamba_root),
        "cad_environment": str(cad_prefix),
        "cad_python": str(cad_python),
        "cad_library_directory": str(cad_lib),
        "pychrono_launcher": "python",
        "pychrono_executable": str(chrono_python),
        "pychrono_environment": None,
        "pychrono_environment_prefix": str(chrono_prefix),
        "cad_lock": CAD_LOCK.name,
        "chrono_lock": CHRONO_LOCK.name,
    }


def _validate_recipe(
    value: Mapping[str, Any], *, require_files: bool
) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema_version") != MACHINE_LAUNCH_SCHEMA:
        raise MachineLaunchError(
            "MACHINE_LAUNCH_RECIPE_INVALID",
            "machine launch recipe is missing or has the wrong schema",
        )
    required_paths = (
        "mamba_root",
        "cad_environment",
        "cad_python",
        "cad_library_directory",
        "pychrono_executable",
        "pychrono_environment_prefix",
    )
    recipe = {key: value.get(key) for key in required_paths}
    recipe["schema_version"] = MACHINE_LAUNCH_SCHEMA
    recipe["pychrono_launcher"] = value.get("pychrono_launcher")
    recipe["pychrono_environment"] = value.get("pychrono_environment")
    recipe["cad_lock"] = value.get("cad_lock", CAD_LOCK.name)
    recipe["chrono_lock"] = value.get("chrono_lock", CHRONO_LOCK.name)
    for key in required_paths:
        if not isinstance(recipe[key], str) or not str(recipe[key]).strip():
            raise MachineLaunchError(
                "MACHINE_LAUNCH_RECIPE_INVALID",
                f"machine launch recipe field {key} must be a path string",
            )
        if not Path(str(recipe[key])).is_absolute():
            raise MachineLaunchError(
                "MACHINE_LAUNCH_RECIPE_INVALID",
                f"machine launch recipe field {key} must be absolute",
            )
    if recipe["pychrono_launcher"] != "python":
        raise MachineLaunchError(
            "MACHINE_LAUNCH_RECIPE_INVALID",
            "machine launch recipe must freeze the python physics launcher",
        )
    if require_files:
        for key in (
            "cad_python",
            "pychrono_executable",
            "cad_library_directory",
            "cad_environment",
            "pychrono_environment_prefix",
        ):
            path = Path(str(recipe[key]))
            if key.endswith("python") or key == "pychrono_executable":
                if not path.is_file():
                    raise MachineLaunchError(
                        "MACHINE_LAUNCH_RECIPE_INVALID",
                        f"machine launch recipe {key} is not a file",
                    )
            elif not path.is_dir():
                raise MachineLaunchError(
                    "MACHINE_LAUNCH_RECIPE_INVALID",
                    f"machine launch recipe {key} is not a directory",
                )
    return recipe


def _require_healthy(
    recipe: Mapping[str, Any],
    *,
    probe: Callable[..., str | None],
) -> None:
    cad_error = probe(
        Path(recipe["cad_python"]),
        modules=CAD_IMPORTS,
        ld_library_path=str(recipe["cad_library_directory"]),
    )
    chrono_error = probe(
        Path(recipe["pychrono_executable"]),
        modules=CHRONO_IMPORTS,
        ld_library_path=None,
    )
    if cad_error is not None or chrono_error is not None:
        raise MachineLaunchError(
            "MACHINE_LAUNCH_PROBE_FAILED",
            _probe_message(cad_error, chrono_error),
        )


def _probe_message(cad_error: str | None, chrono_error: str | None) -> str:
    parts = []
    if cad_error:
        parts.append(f"CAD interpreter failed: {cad_error}")
    if chrono_error:
        parts.append(f"physics interpreter failed: {chrono_error}")
    return " ".join(parts) or "locked environment probe failed"


def _probe_imports(
    python: Path,
    *,
    modules: tuple[str, ...],
    ld_library_path: str | None,
) -> str | None:
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    if ld_library_path:
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = (
            ld_library_path
            if not existing
            else f"{ld_library_path}{os.pathsep}{existing}"
        )
    statement = "import " + ", ".join(modules)
    try:
        completed = subprocess.run(
            (str(python), "-c", statement),
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"{type(exc).__name__}: {exc}"
    if completed.returncode == 0:
        return None
    detail = completed.stderr.decode("utf-8", "replace").strip() or (
        completed.stdout.decode("utf-8", "replace").strip()
    )
    return detail[:2000] or f"exit {completed.returncode}"


def _install_lock(
    *,
    mamba_root: Path,
    cad_prefix: Path,
    chrono_prefix: Path,
    cad_exists: bool,
    chrono_exists: bool,
) -> None:
    if os.name != "posix":
        raise MachineLaunchError(
            "MACHINE_LAUNCH_UNSUPPORTED_PLATFORM",
            "automatic sibling-environment install is POSIX-only; write a "
            "machine launch recipe by hand on this platform",
        )
    if not CAD_LOCK.is_file() or not CHRONO_LOCK.is_file():
        raise MachineLaunchError(
            "MACHINE_LAUNCH_ENV_MISSING",
            "locked environment YAML files are missing from this checkout",
        )
    micromamba = shutil.which("micromamba")
    if micromamba is None:
        raise MachineLaunchError(
            "MACHINE_LAUNCH_MICROMAMBA_REQUIRED",
            "micromamba is required to create the locked CAD/physics environments",
        )
    commands: list[tuple[str, ...]] = []
    if not cad_exists:
        commands.append(
            (
                micromamba,
                "create",
                "-y",
                "-r",
                str(mamba_root),
                "-f",
                str(CAD_LOCK),
            )
        )
    else:
        commands.append(
            (
                micromamba,
                "env",
                "update",
                "-y",
                "-p",
                str(cad_prefix),
                "-f",
                str(CAD_LOCK),
            )
        )
    if not chrono_exists:
        commands.append(
            (
                micromamba,
                "create",
                "-y",
                "-r",
                str(mamba_root),
                "-f",
                str(CHRONO_LOCK),
            )
        )
    else:
        commands.append(
            (
                micromamba,
                "env",
                "update",
                "-y",
                "-p",
                str(chrono_prefix),
                "-f",
                str(CHRONO_LOCK),
            )
        )
    for command in commands:
        try:
            completed = subprocess.run(
                command,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                check=False,
            )
        except OSError as exc:
            raise MachineLaunchError(
                "MACHINE_LAUNCH_INSTALL_FAILED",
                f"{type(exc).__name__}: {exc}",
            ) from exc
        if completed.returncode != 0:
            detail = completed.stderr.decode("utf-8", "replace").strip()[:2000]
            raise MachineLaunchError(
                "MACHINE_LAUNCH_INSTALL_FAILED",
                detail or f"{command[0]} exited {completed.returncode}",
            )
