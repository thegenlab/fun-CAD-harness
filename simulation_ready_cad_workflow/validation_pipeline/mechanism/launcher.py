from __future__ import annotations

import os
import re
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterator, Mapping

from .models import MechanismValidationError


DEFAULT_PYCHRONO_ENVIRONMENT = "chrono_py312_projectchrono"
PYCHRONO_LAUNCHER_ENV = "SIMREADY_PYCHRONO_LAUNCHER"
PYCHRONO_EXECUTABLE_ENV = "SIMREADY_PYCHRONO_EXECUTABLE"
PYCHRONO_ENVIRONMENT_ENV = "SIMREADY_PYCHRONO_ENVIRONMENT"

_ENVIRONMENT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SHELL_WRAPPERS = {
    "bash",
    "cmd",
    "cmd.exe",
    "powershell",
    "powershell.exe",
    "pwsh",
    "pwsh.exe",
    "sh",
}


def _fail(
    message: str,
    *,
    code: str = "PYCHRONO_LAUNCHER_INVALID",
) -> MechanismValidationError:
    return MechanismValidationError(code, message)


class PyChronoLauncherKind(str, Enum):
    PYTHON = "python"
    CONDA = "conda"
    MICROMAMBA = "micromamba"


def _nonempty_token(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or "\0" in value
        or "\r" in value
        or "\n" in value
    ):
        raise _fail(f"{label} must be a non-empty single-line string")
    return value


def _environment_selector(value: str) -> tuple[str, str]:
    environment = _nonempty_token(value, "PyChrono environment")
    if _ENVIRONMENT_NAME.fullmatch(environment) is not None:
        return "-n", environment
    path = Path(environment)
    if not path.is_absolute():
        raise _fail(
            "PyChrono environment must be a validated environment name or "
            "an absolute environment prefix"
        )
    return "-p", environment


@dataclass(frozen=True)
class PyChronoLauncher:
    """Shell-free command prefix for one trusted PyChrono interpreter."""

    kind: PyChronoLauncherKind
    executable: str
    environment: str | None = None

    def __post_init__(self) -> None:
        try:
            kind = PyChronoLauncherKind(self.kind)
        except ValueError as exc:
            raise _fail(f"unsupported PyChrono launcher: {self.kind!r}") from exc
        object.__setattr__(self, "kind", kind)
        executable = _nonempty_token(
            self.executable, "PyChrono launcher executable"
        )
        if Path(executable).name.casefold() in _SHELL_WRAPPERS:
            raise _fail("PyChrono launcher cannot be a shell wrapper")
        if kind is PyChronoLauncherKind.PYTHON:
            if self.environment is not None:
                raise _fail(
                    "the python launcher cannot select a Conda or micromamba "
                    "environment"
                )
        else:
            if self.environment is None:
                raise _fail(
                    f"the {kind.value} launcher requires an environment"
                )
            _environment_selector(self.environment)

    def python_command(self, *arguments: str) -> tuple[str, ...]:
        validated = tuple(
            _nonempty_token(argument, "PyChrono Python argument")
            for argument in arguments
        )
        if self.kind is PyChronoLauncherKind.PYTHON:
            return (self.executable, *validated)
        assert self.environment is not None
        selector, environment = _environment_selector(self.environment)
        return (
            self.executable,
            "run",
            selector,
            environment,
            "python",
            *validated,
        )

    def to_dict(self) -> dict[str, str | None]:
        return {
            "kind": self.kind.value,
            "executable": self.executable,
            "environment": self.environment,
        }


_ACTIVE_PYCHRONO_LAUNCHER: ContextVar[PyChronoLauncher | None] = ContextVar(
    "active_pychrono_launcher",
    default=None,
)


def resolve_pychrono_launcher(
    launcher: PyChronoLauncher | PyChronoLauncherKind | str | None = None,
    *,
    executable: str | None = None,
    environment: str | None = None,
    environ: Mapping[str, str] | None = None,
    platform_name: str | None = None,
    python_executable: str | None = None,
) -> PyChronoLauncher:
    """Resolve explicit/operator configuration without guessing an env path.

    Windows retains the historical Conda default. POSIX is fail-closed because
    the harness CAD interpreter and PyChrono interpreter are intentionally
    isolated: operators must explicitly select ``python``, ``micromamba``, or
    ``conda``. Environment launchers also require an environment name or
    absolute prefix.
    """

    if isinstance(launcher, PyChronoLauncher):
        if executable is not None or environment is not None:
            raise _fail(
                "a resolved PyChrono launcher cannot be combined with "
                "executable or environment overrides"
            )
        return launcher

    values = os.environ if environ is None else environ
    raw_kind = (
        launcher
        if launcher is not None
        else values.get(PYCHRONO_LAUNCHER_ENV)
    )
    effective_platform = os.name if platform_name is None else platform_name
    if raw_kind is None:
        if effective_platform != "nt":
            raise _fail(
                "POSIX requires an explicit PyChrono launcher; select the "
                "verified physics interpreter with "
                f"{PYCHRONO_LAUNCHER_ENV}=python, conda, or micromamba",
                code="PYCHRONO_LAUNCHER_REQUIRED",
            )
        raw_kind = PyChronoLauncherKind.CONDA
    try:
        kind = PyChronoLauncherKind(raw_kind)
    except ValueError as exc:
        raise _fail(f"unsupported PyChrono launcher: {raw_kind!r}") from exc

    configured_executable = (
        executable
        if executable is not None
        else values.get(PYCHRONO_EXECUTABLE_ENV)
    )
    configured_environment = (
        environment
        if environment is not None
        else values.get(PYCHRONO_ENVIRONMENT_ENV)
    )
    if kind is PyChronoLauncherKind.PYTHON:
        if configured_environment is not None:
            raise _fail(
                "a PyChrono environment was supplied for the python launcher; "
                "select conda or micromamba instead"
            )
        return PyChronoLauncher(
            kind=kind,
            executable=(
                configured_executable
                if configured_executable is not None
                else python_executable or sys.executable
            ),
        )

    if configured_executable is None:
        configured_executable = (
            "conda.exe"
            if kind is PyChronoLauncherKind.CONDA
            and effective_platform == "nt"
            else kind.value
        )
    if configured_environment is None:
        configured_environment = DEFAULT_PYCHRONO_ENVIRONMENT
    return PyChronoLauncher(
        kind=kind,
        executable=configured_executable,
        environment=configured_environment,
    )


def active_pychrono_launcher() -> PyChronoLauncher:
    configured = _ACTIVE_PYCHRONO_LAUNCHER.get()
    return configured if configured is not None else resolve_pychrono_launcher()


def select_pychrono_launcher(
    launcher: PyChronoLauncher | None,
    *,
    legacy_conda_executable: str | None = None,
    legacy_conda_environment: str | None = None,
) -> PyChronoLauncher:
    """Select a launcher while retaining the old programmatic Conda knobs."""

    if launcher is not None and (
        legacy_conda_executable is not None
        or legacy_conda_environment is not None
    ):
        raise _fail(
            "pychrono_launcher cannot be combined with legacy Conda overrides"
        )
    if (
        legacy_conda_executable is not None
        or legacy_conda_environment is not None
    ):
        return resolve_pychrono_launcher(
            PyChronoLauncherKind.CONDA,
            executable=legacy_conda_executable,
            environment=legacy_conda_environment,
        )
    return launcher if launcher is not None else active_pychrono_launcher()


@contextmanager
def use_pychrono_launcher(
    launcher: PyChronoLauncher | None,
) -> Iterator[None]:
    """Bind one launcher across dispatch, finalize, and whole-machine stages."""

    if launcher is None:
        yield
        return
    token = _ACTIVE_PYCHRONO_LAUNCHER.set(launcher)
    try:
        yield
    finally:
        _ACTIVE_PYCHRONO_LAUNCHER.reset(token)
