from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


GIB = 1024**3
MAX_MANIFEST_BYTES = 2 * 1024 * 1024
MAX_MANIFEST_NESTING = 64
MAX_JOBS = 256
MAX_COMMAND_ARGUMENTS = 512
MAX_EXPECTED_OUTPUTS = 2_000
MAX_INPUT_FILES = 2_000
MAX_ENVIRONMENT_VARIABLES = 512
_JOB_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_WINDOWS_RESERVED_NAMES = {
    "AUX",
    "CLOCK$",
    "CON",
    "CONIN$",
    "CONOUT$",
    "NUL",
    "PRN",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}
_WINDOWS_FORBIDDEN_PATH_CHARACTERS = set('<>:"|?*')
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


class ManifestError(ValueError):
    """A queue manifest is malformed or unsafe to execute."""


class JobKind(str, Enum):
    GENERIC = "generic"
    PYCHRONO = "pychrono"
    RENDER = "render"


class JobStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    TIMED_OUT = "TIMED_OUT"
    RESOURCE_BLOCKED = "RESOURCE_BLOCKED"
    RESOURCE_ABORTED = "RESOURCE_ABORTED"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    INTERRUPTED = "INTERRUPTED"
    CANCELLED = "CANCELLED"


@dataclass(frozen=True)
class ResourcePolicy:
    preflight_cpu_percent: float = 85.0
    preflight_memory_percent: float = 88.0
    min_available_memory_gib: float = 3.0
    min_free_disk_gib: float = 10.0
    critical_memory_percent: float = 93.0
    critical_available_memory_gib: float = 1.5
    critical_free_disk_gib: float = 5.0
    preflight_sample_seconds: float = 1.0
    resource_poll_seconds: float = 1.0
    max_resource_wait_seconds: float = 30.0
    critical_pressure_samples: int = 3
    cooldown_seconds: float = 5.0
    pychrono_cpu_threads: int = 2
    generic_cpu_threads: int = 2
    terminate_grace_seconds: float = 5.0
    max_log_bytes_per_stream: int = 16 * 1024 * 1024
    lower_process_priority: bool = True

    @classmethod
    def from_dict(cls, value: object) -> ResourcePolicy:
        data = _expect_mapping(value, "resource_policy")
        _reject_unknown_keys(data, set(cls.__dataclass_fields__), "resource_policy")
        policy = cls(**data)
        policy.validate()
        return policy

    def validate(self) -> None:
        for name in (
            "preflight_cpu_percent",
            "preflight_memory_percent",
            "critical_memory_percent",
        ):
            number = _finite_number(getattr(self, name), f"resource_policy.{name}")
            if not 0 < number <= 100:
                raise ManifestError(f"resource_policy.{name} must be in (0, 100]")
        if self.preflight_cpu_percent > 95:
            raise ManifestError(
                "preflight_cpu_percent cannot exceed the hard safety "
                "ceiling of 95"
            )
        if self.preflight_memory_percent > 95:
            raise ManifestError(
                "preflight_memory_percent cannot exceed the hard safety "
                "ceiling of 95"
            )
        if self.critical_memory_percent > 98:
            raise ManifestError(
                "critical_memory_percent cannot exceed the hard safety "
                "ceiling of 98"
            )
        if self.critical_memory_percent < self.preflight_memory_percent:
            raise ManifestError(
                "critical_memory_percent cannot be lower than "
                "preflight_memory_percent"
            )
        for name in (
            "min_available_memory_gib",
            "min_free_disk_gib",
            "critical_available_memory_gib",
            "critical_free_disk_gib",
            "preflight_sample_seconds",
            "resource_poll_seconds",
            "max_resource_wait_seconds",
            "cooldown_seconds",
            "terminate_grace_seconds",
        ):
            number = _finite_number(getattr(self, name), f"resource_policy.{name}")
            if number < 0:
                raise ManifestError(f"resource_policy.{name} cannot be negative")
        if self.min_available_memory_gib < 0.5:
            raise ManifestError(
                "min_available_memory_gib cannot be below the hard safety "
                "reserve of 0.5"
            )
        if self.critical_available_memory_gib < 0.25:
            raise ManifestError(
                "critical_available_memory_gib cannot be below the hard "
                "safety reserve of 0.25"
            )
        if self.critical_available_memory_gib > self.min_available_memory_gib:
            raise ManifestError(
                "critical_available_memory_gib cannot exceed "
                "min_available_memory_gib"
            )
        if self.min_free_disk_gib < 1.0:
            raise ManifestError(
                "min_free_disk_gib cannot be below the hard safety "
                "reserve of 1.0"
            )
        if self.critical_free_disk_gib < 0.5:
            raise ManifestError(
                "critical_free_disk_gib cannot be below the hard "
                "safety reserve of 0.5"
            )
        if self.critical_free_disk_gib > self.min_free_disk_gib:
            raise ManifestError(
                "critical_free_disk_gib cannot exceed "
                "min_free_disk_gib"
            )
        if not 0 < self.resource_poll_seconds <= 5:
            raise ManifestError(
                "resource_poll_seconds must be in (0, 5]"
            )
        if self.preflight_sample_seconds > 10:
            raise ManifestError(
                "preflight_sample_seconds cannot exceed 10"
            )
        if self.terminate_grace_seconds > 30:
            raise ManifestError(
                "terminate_grace_seconds cannot exceed 30"
            )
        if not isinstance(self.critical_pressure_samples, int) or isinstance(
            self.critical_pressure_samples, bool
        ):
            raise ManifestError("critical_pressure_samples must be an integer")
        if self.critical_pressure_samples < 1:
            raise ManifestError("critical_pressure_samples must be at least 1")
        if (
            self.critical_pressure_samples * self.resource_poll_seconds
            > 15
        ):
            raise ManifestError(
                "critical pressure detection window cannot exceed 15 seconds"
            )
        for name in ("pychrono_cpu_threads", "generic_cpu_threads"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ManifestError(f"resource_policy.{name} must be an integer")
            if not 1 <= value <= 4:
                raise ManifestError(f"resource_policy.{name} must be in [1, 4]")
        if (
            not isinstance(self.max_log_bytes_per_stream, int)
            or isinstance(self.max_log_bytes_per_stream, bool)
            or not 1024 <= self.max_log_bytes_per_stream <= 256 * 1024 * 1024
        ):
            raise ManifestError(
                "resource_policy.max_log_bytes_per_stream must be an integer "
                "in [1024, 268435456]"
            )
        combined_log_buffer_gib = (
            2.0 * float(self.max_log_bytes_per_stream) / GIB
        )
        available_reserve_gap_gib = (
            float(self.min_available_memory_gib)
            - float(self.critical_available_memory_gib)
        )
        if combined_log_buffer_gib > available_reserve_gap_gib:
            raise ManifestError(
                "combined stdout/stderr buffers "
                f"({combined_log_buffer_gib:.3f} GiB) must fit within "
                "min_available_memory_gib minus "
                "critical_available_memory_gib "
                f"({available_reserve_gap_gib:.3f} GiB)"
            )
        if not isinstance(self.lower_process_priority, bool):
            raise ManifestError("lower_process_priority must be boolean")


@dataclass(frozen=True)
class QueuePolicy:
    max_wall_time_seconds: float = 25200.0
    stop_on_failure: bool = True
    reuse_successful: bool = False
    max_timeout_resume_attempts: int = 0
    max_parallel_jobs: int = 1

    @classmethod
    def from_dict(cls, value: object) -> QueuePolicy:
        data = _expect_mapping(value, "queue_policy")
        _reject_unknown_keys(data, set(cls.__dataclass_fields__), "queue_policy")
        policy = cls(**data)
        policy.validate()
        return policy

    def validate(self) -> None:
        duration = _finite_number(
            self.max_wall_time_seconds,
            "queue_policy.max_wall_time_seconds",
        )
        if not 0 < duration <= 86400:
            raise ManifestError(
                "queue_policy.max_wall_time_seconds must be in (0, 86400]"
            )
        for name in ("stop_on_failure", "reuse_successful"):
            if not isinstance(getattr(self, name), bool):
                raise ManifestError(f"queue_policy.{name} must be boolean")
        if (
            isinstance(self.max_timeout_resume_attempts, bool)
            or not isinstance(self.max_timeout_resume_attempts, int)
            or not 0 <= self.max_timeout_resume_attempts <= 16
        ):
            raise ManifestError(
                "queue_policy.max_timeout_resume_attempts must be an integer "
                "in [0, 16]"
            )
        if (
            isinstance(self.max_parallel_jobs, bool)
            or not isinstance(self.max_parallel_jobs, int)
            or not 1 <= self.max_parallel_jobs <= MAX_JOBS
        ):
            raise ManifestError(
                "queue_policy.max_parallel_jobs must be an integer in "
                f"[1, {MAX_JOBS}]"
            )


@dataclass(frozen=True)
class JobSpec:
    job_id: str
    command: tuple[str, ...]
    timeout_seconds: float
    kind: JobKind = JobKind.GENERIC
    working_directory: str = "."
    expected_outputs: tuple[str, ...] = ()
    input_files: dict[str, str] = field(default_factory=dict)
    environment: dict[str, str] = field(default_factory=dict)
    inherit_parent_environment: bool = True
    estimated_peak_memory_gib: float | None = None
    max_process_tree_memory_gib: float | None = None
    estimated_wall_time_seconds: float | None = None
    estimated_simulation_steps: int | None = None
    cpu_threads: int | None = None

    @classmethod
    def from_dict(cls, value: object, *, index: int) -> JobSpec:
        field_name = f"jobs[{index}]"
        data = _expect_mapping(value, field_name)
        allowed = {
            "job_id",
            "command",
            "timeout_seconds",
            "kind",
            "working_directory",
            "expected_outputs",
            "input_files",
            "environment",
            "inherit_parent_environment",
            "estimated_peak_memory_gib",
            "max_process_tree_memory_gib",
            "estimated_wall_time_seconds",
            "estimated_simulation_steps",
            "cpu_threads",
        }
        _reject_unknown_keys(data, allowed, field_name)
        required = {"job_id", "command", "timeout_seconds"}
        missing = sorted(required - data.keys())
        if missing:
            raise ManifestError(f"{field_name} is missing: {', '.join(missing)}")

        command_value = data["command"]
        if not isinstance(command_value, list) or not command_value:
            raise ManifestError(f"{field_name}.command must be a non-empty array")
        if len(command_value) > MAX_COMMAND_ARGUMENTS:
            raise ManifestError(
                f"{field_name}.command must contain at most "
                f"{MAX_COMMAND_ARGUMENTS} arguments"
            )
        command: list[str] = []
        for command_index, item in enumerate(command_value):
            if not isinstance(item, str) or not item or "\x00" in item:
                raise ManifestError(
                    f"{field_name}.command[{command_index}] must be a "
                    "non-empty string without NUL"
                )
            command.append(item)
        executable = Path(command[0]).name.lower()
        if executable in _SHELL_WRAPPERS:
            raise ManifestError(
                f"{field_name}.command uses shell wrapper {command[0]!r}; "
                "provide the executable and arguments directly"
            )

        kind_value = data.get("kind", JobKind.GENERIC.value)
        try:
            kind = JobKind(kind_value)
        except (TypeError, ValueError) as exc:
            raise ManifestError(
                f"{field_name}.kind must be one of "
                f"{', '.join(item.value for item in JobKind)}"
            ) from exc

        outputs_value = data.get("expected_outputs", [])
        if not isinstance(outputs_value, list):
            raise ManifestError(f"{field_name}.expected_outputs must be an array")
        if len(outputs_value) > MAX_EXPECTED_OUTPUTS:
            raise ManifestError(
                f"{field_name}.expected_outputs must contain at most "
                f"{MAX_EXPECTED_OUTPUTS} paths"
            )
        outputs = tuple(
            _relative_path(item, f"{field_name}.expected_outputs[{output_index}]")
            for output_index, item in enumerate(outputs_value)
        )
        if len({output.casefold() for output in outputs}) != len(outputs):
            raise ManifestError(
                f"{field_name}.expected_outputs contains duplicate or "
                "case-colliding paths"
            )

        inputs_value = data.get("input_files", {})
        inputs_data = _expect_mapping(
            inputs_value,
            f"{field_name}.input_files",
        )
        if len(inputs_data) > MAX_INPUT_FILES:
            raise ManifestError(
                f"{field_name}.input_files must contain at most "
                f"{MAX_INPUT_FILES} paths"
            )
        input_files: dict[str, str] = {}
        folded_input_paths: set[str] = set()
        for raw_path, raw_digest in inputs_data.items():
            input_path = _relative_path(
                raw_path,
                f"{field_name}.input_files path",
            )
            if input_path.casefold() in folded_input_paths:
                raise ManifestError(
                    f"{field_name}.input_files contains duplicate or "
                    f"case-colliding path {input_path!r}"
                )
            folded_input_paths.add(input_path.casefold())
            if not isinstance(raw_digest, str) or not _SHA256_RE.fullmatch(
                raw_digest
            ):
                raise ManifestError(
                    f"{field_name}.input_files[{input_path!r}] must be a "
                    "lowercase SHA-256 digest"
                )
            input_files[input_path] = raw_digest
        output_paths = {item.casefold() for item in outputs}
        overlapping_paths = sorted(
            path for path in input_files if path.casefold() in output_paths
        )
        if overlapping_paths:
            raise ManifestError(
                f"{field_name} cannot declare the same path as input and "
                "expected output: " + ", ".join(overlapping_paths)
            )

        environment_value = data.get("environment", {})
        environment_data = _expect_mapping(
            environment_value,
            f"{field_name}.environment",
        )
        if len(environment_data) > MAX_ENVIRONMENT_VARIABLES:
            raise ManifestError(
                f"{field_name}.environment must contain at most "
                f"{MAX_ENVIRONMENT_VARIABLES} variables"
            )
        environment: dict[str, str] = {}
        folded_environment_names: set[str] = set()
        for key, item in environment_data.items():
            if not isinstance(key, str) or not _ENV_NAME_RE.fullmatch(key):
                raise ManifestError(
                    f"{field_name}.environment has invalid variable name {key!r}"
                )
            folded_key = key.casefold()
            if folded_key in folded_environment_names:
                raise ManifestError(
                    f"{field_name}.environment contains case-colliding "
                    f"variable name {key!r}"
                )
            folded_environment_names.add(folded_key)
            if not isinstance(item, str) or "\x00" in item:
                raise ManifestError(
                    f"{field_name}.environment.{key} must be a string without NUL"
                )
            environment[key] = item

        job = cls(
            job_id=data["job_id"],
            command=tuple(command),
            timeout_seconds=data["timeout_seconds"],
            kind=kind,
            working_directory=_relative_path(
                data.get("working_directory", "."),
                f"{field_name}.working_directory",
                allow_dot=True,
            ),
            expected_outputs=outputs,
            input_files=input_files,
            environment=environment,
            inherit_parent_environment=data.get(
                "inherit_parent_environment",
                True,
            ),
            estimated_peak_memory_gib=data.get("estimated_peak_memory_gib"),
            max_process_tree_memory_gib=data.get(
                "max_process_tree_memory_gib"
            ),
            estimated_wall_time_seconds=data.get(
                "estimated_wall_time_seconds"
            ),
            estimated_simulation_steps=data.get(
                "estimated_simulation_steps"
            ),
            cpu_threads=data.get("cpu_threads"),
        )
        job.validate(field_name)
        return job

    def validate(self, field_name: str = "job") -> None:
        if not isinstance(self.kind, JobKind):
            raise ManifestError(
                f"{field_name}.kind must be a JobKind value"
            )
        if (
            not isinstance(self.command, tuple)
            or not self.command
            or len(self.command) > MAX_COMMAND_ARGUMENTS
        ):
            raise ManifestError(
                f"{field_name}.command must be a non-empty tuple with at most "
                f"{MAX_COMMAND_ARGUMENTS} arguments"
            )
        for index, item in enumerate(self.command):
            if not isinstance(item, str) or not item or "\x00" in item:
                raise ManifestError(
                    f"{field_name}.command[{index}] must be a non-empty "
                    "string without NUL"
                )
        executable = Path(self.command[0]).name.lower()
        if executable in _SHELL_WRAPPERS:
            raise ManifestError(
                f"{field_name}.command uses shell wrapper "
                f"{self.command[0]!r}"
            )
        _relative_path(
            self.working_directory,
            f"{field_name}.working_directory",
            allow_dot=True,
        )
        if (
            not isinstance(self.expected_outputs, tuple)
            or len(self.expected_outputs) > MAX_EXPECTED_OUTPUTS
        ):
            raise ManifestError(
                f"{field_name}.expected_outputs must be a tuple with at most "
                f"{MAX_EXPECTED_OUTPUTS} paths"
            )
        for index, output in enumerate(self.expected_outputs):
            _relative_path(
                output,
                f"{field_name}.expected_outputs[{index}]",
            )
        if (
            len({output.casefold() for output in self.expected_outputs})
            != len(self.expected_outputs)
        ):
            raise ManifestError(
                f"{field_name}.expected_outputs contains duplicate or "
                "case-colliding paths"
            )
        if not isinstance(self.input_files, dict):
            raise ManifestError(f"{field_name}.input_files must be an object")
        if len(self.input_files) > MAX_INPUT_FILES:
            raise ManifestError(
                f"{field_name}.input_files must contain at most "
                f"{MAX_INPUT_FILES} paths"
            )
        folded_input_paths: set[str] = set()
        for path, digest in self.input_files.items():
            normalized = _relative_path(
                path,
                f"{field_name}.input_files path",
            )
            if normalized != path:
                raise ManifestError(
                    f"{field_name}.input_files path is not normalized: {path!r}"
                )
            if path.casefold() in folded_input_paths:
                raise ManifestError(
                    f"{field_name}.input_files contains duplicate or "
                    f"case-colliding path {path!r}"
                )
            folded_input_paths.add(path.casefold())
            if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
                raise ManifestError(
                    f"{field_name}.input_files[{path!r}] must be a lowercase "
                    "SHA-256 digest"
                )
        overlapping_paths = sorted(
            path
            for path in self.input_files
            if path.casefold()
            in {output.casefold() for output in self.expected_outputs}
        )
        if overlapping_paths:
            raise ManifestError(
                f"{field_name} cannot declare the same path as input and "
                "expected output: " + ", ".join(overlapping_paths)
            )
        if not isinstance(self.environment, dict):
            raise ManifestError(f"{field_name}.environment must be an object")
        if not isinstance(self.inherit_parent_environment, bool):
            raise ManifestError(
                f"{field_name}.inherit_parent_environment must be boolean"
            )
        if len(self.environment) > MAX_ENVIRONMENT_VARIABLES:
            raise ManifestError(
                f"{field_name}.environment must contain at most "
                f"{MAX_ENVIRONMENT_VARIABLES} variables"
            )
        folded_environment_names: set[str] = set()
        for key, value in self.environment.items():
            if not isinstance(key, str) or not _ENV_NAME_RE.fullmatch(key):
                raise ManifestError(
                    f"{field_name}.environment has invalid variable name "
                    f"{key!r}"
                )
            if key.casefold() in folded_environment_names:
                raise ManifestError(
                    f"{field_name}.environment contains case-colliding "
                    f"variable name {key!r}"
                )
            folded_environment_names.add(key.casefold())
            if not isinstance(value, str) or "\x00" in value:
                raise ManifestError(
                    f"{field_name}.environment.{key} must be a string "
                    "without NUL"
                )
        if not isinstance(self.job_id, str) or not _JOB_ID_RE.fullmatch(self.job_id):
            raise ManifestError(
                f"{field_name}.job_id must match {_JOB_ID_RE.pattern}"
            )
        if _is_windows_reserved_name(self.job_id):
            raise ManifestError(
                f"{field_name}.job_id cannot use a Windows device name"
            )
        if self.kind in {JobKind.PYCHRONO, JobKind.RENDER}:
            required_budget_fields = {
                "estimated_peak_memory_gib": self.estimated_peak_memory_gib,
                "max_process_tree_memory_gib": (
                    self.max_process_tree_memory_gib
                ),
                "estimated_wall_time_seconds": (
                    self.estimated_wall_time_seconds
                ),
            }
            missing_budget_fields = [
                name
                for name, value in required_budget_fields.items()
                if value is None
            ]
            if missing_budget_fields:
                raise ManifestError(
                    f"{field_name} {self.kind.value} job is missing resource "
                    "budget field(s): "
                    + ", ".join(missing_budget_fields)
                )
        if (
            self.kind is JobKind.PYCHRONO
            and self.estimated_simulation_steps is None
        ):
            raise ManifestError(
                f"{field_name} pychrono job must declare "
                "estimated_simulation_steps"
            )
        timeout = _finite_number(
            self.timeout_seconds,
            f"{field_name}.timeout_seconds",
        )
        if not 0 < timeout <= 21600:
            raise ManifestError(
                f"{field_name}.timeout_seconds must be in (0, 21600]"
            )
        if self.estimated_peak_memory_gib is not None:
            estimate = _finite_number(
                self.estimated_peak_memory_gib,
                f"{field_name}.estimated_peak_memory_gib",
            )
            if not 0 < estimate <= 128:
                raise ManifestError(
                    f"{field_name}.estimated_peak_memory_gib must be in (0, 128]"
                )
        if self.max_process_tree_memory_gib is not None:
            limit = _finite_number(
                self.max_process_tree_memory_gib,
                f"{field_name}.max_process_tree_memory_gib",
            )
            if not 0 < limit <= 128:
                raise ManifestError(
                    f"{field_name}.max_process_tree_memory_gib must be "
                    "in (0, 128]"
                )
            if (
                self.estimated_peak_memory_gib is not None
                and limit < self.estimated_peak_memory_gib
            ):
                raise ManifestError(
                    f"{field_name}.max_process_tree_memory_gib cannot be "
                    "lower than estimated_peak_memory_gib"
                )
        if self.estimated_wall_time_seconds is not None:
            estimate = _finite_number(
                self.estimated_wall_time_seconds,
                f"{field_name}.estimated_wall_time_seconds",
            )
            if not 0 < estimate <= self.timeout_seconds:
                raise ManifestError(
                    f"{field_name}.estimated_wall_time_seconds must be "
                    "positive and no greater than timeout_seconds"
                )
        if self.estimated_simulation_steps is not None:
            steps = self.estimated_simulation_steps
            if not isinstance(steps, int) or isinstance(steps, bool):
                raise ManifestError(
                    f"{field_name}.estimated_simulation_steps must be an integer"
                )
            if not 1 <= steps <= 1_000_000_000:
                raise ManifestError(
                    f"{field_name}.estimated_simulation_steps must be in "
                    "[1, 1000000000]"
                )
        if self.cpu_threads is not None:
            if not isinstance(self.cpu_threads, int) or isinstance(
                self.cpu_threads, bool
            ):
                raise ManifestError(f"{field_name}.cpu_threads must be an integer")
            if not 1 <= self.cpu_threads <= 4:
                raise ManifestError(f"{field_name}.cpu_threads must be in [1, 4]")
        if _job_text_char_count(self) > MAX_MANIFEST_BYTES:
            raise ManifestError(
                f"{field_name} text exceeds the bounded manifest budget"
            )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "job_id": self.job_id,
            "kind": self.kind.value,
            "command": list(self.command),
            "working_directory": self.working_directory,
            "timeout_seconds": self.timeout_seconds,
            "expected_outputs": list(self.expected_outputs),
            "environment": dict(sorted(self.environment.items())),
            "estimated_peak_memory_gib": self.estimated_peak_memory_gib,
            "max_process_tree_memory_gib": (
                self.max_process_tree_memory_gib
            ),
            "estimated_wall_time_seconds": self.estimated_wall_time_seconds,
            "estimated_simulation_steps": self.estimated_simulation_steps,
            "cpu_threads": self.cpu_threads,
        }
        # Preserve legacy fingerprints for manifests that predate explicit
        # content-bound inputs. New jobs include the map in their fingerprint.
        if self.input_files:
            payload["input_files"] = dict(sorted(self.input_files.items()))
        # Preserve legacy fingerprints while allowing security-sensitive jobs
        # to opt into a closed, manifest-bound process environment.
        if not self.inherit_parent_environment:
            payload["inherit_parent_environment"] = False
        return payload

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict())


def _job_text_char_count(job: JobSpec) -> int:
    return (
        len(job.job_id)
        + sum(len(item) for item in job.command)
        + len(job.working_directory)
        + sum(len(item) for item in job.expected_outputs)
        + sum(len(path) + len(digest) for path, digest in job.input_files.items())
        + sum(len(key) + len(value) for key, value in job.environment.items())
    )


@dataclass(frozen=True)
class QueueManifest:
    run_name: str
    jobs: tuple[JobSpec, ...]
    resource_policy: ResourcePolicy = field(default_factory=ResourcePolicy)
    queue_policy: QueuePolicy = field(default_factory=QueuePolicy)
    schema_version: str = "0.1"

    @classmethod
    def from_file(cls, path: str | Path) -> QueueManifest:
        manifest_path = Path(path)
        try:
            before = manifest_path.stat()
        except OSError as exc:
            raise ManifestError(
                f"cannot inspect manifest {manifest_path}: {exc}"
            ) from exc
        if before.st_size > MAX_MANIFEST_BYTES:
            raise ManifestError(
                f"manifest exceeds {MAX_MANIFEST_BYTES} byte limit"
            )
        try:
            with manifest_path.open("rb") as stream:
                raw = stream.read(MAX_MANIFEST_BYTES + 1)
            after = manifest_path.stat()
        except OSError as exc:
            raise ManifestError(
                f"cannot read manifest {manifest_path}: {exc}"
            ) from exc
        if (
            after.st_size > MAX_MANIFEST_BYTES
            or len(raw) > MAX_MANIFEST_BYTES
        ):
            raise ManifestError(
                f"manifest exceeds {MAX_MANIFEST_BYTES} byte limit"
            )
        before_identity = (
            int(before.st_size),
            int(before.st_mtime_ns),
            getattr(before, "st_ino", None),
        )
        after_identity = (
            int(after.st_size),
            int(after.st_mtime_ns),
            getattr(after, "st_ino", None),
        )
        if before_identity != after_identity or len(raw) != after.st_size:
            raise ManifestError(
                f"manifest changed while being read: {manifest_path}"
            )
        try:
            text = raw.decode("utf-8")
        except UnicodeError as exc:
            raise ManifestError(
                f"cannot read manifest {manifest_path} as UTF-8: {exc}"
            ) from exc
        nesting = _max_json_nesting(text)
        if nesting > MAX_MANIFEST_NESTING:
            raise ManifestError(
                f"manifest nesting {nesting} exceeds "
                f"{MAX_MANIFEST_NESTING} level limit"
            )
        data = _strict_json_object(text, source=str(manifest_path))
        allowed = {
            "schema_version",
            "run_name",
            "jobs",
            "resource_policy",
            "queue_policy",
        }
        _reject_unknown_keys(data, allowed, "manifest")
        if data.get("schema_version") != "0.1":
            raise ManifestError("manifest.schema_version must equal '0.1'")
        run_name = data.get("run_name")
        if not isinstance(run_name, str) or not _JOB_ID_RE.fullmatch(run_name):
            raise ManifestError(
                f"manifest.run_name must match {_JOB_ID_RE.pattern}"
            )
        if _is_windows_reserved_name(run_name):
            raise ManifestError(
                "manifest.run_name cannot use a Windows device name"
            )
        jobs_value = data.get("jobs")
        if not isinstance(jobs_value, list) or not jobs_value:
            raise ManifestError("manifest.jobs must be a non-empty array")
        if len(jobs_value) > MAX_JOBS:
            raise ManifestError(
                f"manifest.jobs must contain at most {MAX_JOBS} jobs"
            )
        jobs = tuple(
            JobSpec.from_dict(item, index=index)
            for index, item in enumerate(jobs_value)
        )
        job_ids = [job.job_id for job in jobs]
        if len({job_id.casefold() for job_id in job_ids}) != len(job_ids):
            raise ManifestError(
                "manifest.jobs contains duplicate or case-colliding job_id values"
            )
        resource_policy = ResourcePolicy.from_dict(
            data.get("resource_policy", {})
        )
        queue_policy = QueuePolicy.from_dict(data.get("queue_policy", {}))
        manifest = cls(
            schema_version="0.1",
            run_name=run_name,
            jobs=jobs,
            resource_policy=resource_policy,
            queue_policy=queue_policy,
        )
        manifest.validate()
        # Preserve the exact, stable bytes that were parsed.  These private
        # source bindings are deliberately excluded from ``to_dict`` and the
        # semantic manifest fingerprint; the safe runner uses them to prove
        # which on-disk queue bytes authorized an execution.
        object.__setattr__(
            manifest,
            "_source_path",
            str(manifest_path.resolve()),
        )
        object.__setattr__(
            manifest,
            "_source_sha256",
            hashlib.sha256(raw).hexdigest(),
        )
        object.__setattr__(
            manifest,
            "_source_size_bytes",
            len(raw),
        )
        return manifest

    def validate(self) -> None:
        if self.schema_version != "0.1":
            raise ManifestError("manifest.schema_version must equal '0.1'")
        if (
            not isinstance(self.run_name, str)
            or not _JOB_ID_RE.fullmatch(self.run_name)
        ):
            raise ManifestError(
                f"manifest.run_name must match {_JOB_ID_RE.pattern}"
            )
        if _is_windows_reserved_name(self.run_name):
            raise ManifestError(
                "manifest.run_name cannot use a Windows device name"
            )
        if (
            not isinstance(self.jobs, tuple)
            or not self.jobs
            or len(self.jobs) > MAX_JOBS
        ):
            raise ManifestError(
                f"manifest.jobs must be a non-empty tuple with at most "
                f"{MAX_JOBS} jobs"
            )
        for index, job in enumerate(self.jobs):
            if not isinstance(job, JobSpec):
                raise ManifestError(
                    f"manifest.jobs[{index}] must be a JobSpec"
                )
            job.validate(f"jobs[{index}]")
        job_ids = [job.job_id for job in self.jobs]
        if len({job_id.casefold() for job_id in job_ids}) != len(job_ids):
            raise ManifestError(
                "manifest.jobs contains duplicate or case-colliding job_id values"
            )
        if not isinstance(self.resource_policy, ResourcePolicy):
            raise ManifestError(
                "manifest.resource_policy must be a ResourcePolicy"
            )
        if not isinstance(self.queue_policy, QueuePolicy):
            raise ManifestError(
                "manifest.queue_policy must be a QueuePolicy"
            )
        self.resource_policy.validate()
        self.queue_policy.validate()
        if self.queue_policy.max_parallel_jobs > 1:
            if not self.queue_policy.stop_on_failure:
                raise ManifestError(
                    "parallel queues require queue_policy.stop_on_failure=true"
                )
            seen_outputs: dict[str, str] = {}
            all_inputs = {
                path.casefold(): job.job_id
                for job in self.jobs
                for path in job.input_files
            }
            for job in self.jobs:
                for output in job.expected_outputs:
                    folded = output.casefold()
                    prior = seen_outputs.get(folded)
                    if prior is not None:
                        raise ManifestError(
                            "parallel jobs must have disjoint expected outputs: "
                            f"{output!r} is shared by {prior!r} and "
                            f"{job.job_id!r}"
                        )
                    if folded in all_inputs:
                        raise ManifestError(
                            "parallel job output cannot be another job's "
                            f"declared input: {output!r}"
                        )
                    seen_outputs[folded] = job.job_id
        text_char_count = len(self.run_name)
        for job in self.jobs:
            text_char_count += _job_text_char_count(job)
            if text_char_count > MAX_MANIFEST_BYTES:
                raise ManifestError(
                    "programmatic manifest text exceeds the bounded size budget"
                )
        serialized = json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        if len(serialized) > MAX_MANIFEST_BYTES:
            raise ManifestError(
                f"programmatic manifest exceeds {MAX_MANIFEST_BYTES} byte limit"
            )

    def to_dict(self) -> dict[str, Any]:
        queue_policy = asdict(self.queue_policy)
        # Keep legacy serial manifests byte/fingerprint compatible.  Parallel
        # execution is an explicit opt-in and is serialized only when > 1.
        if self.queue_policy.max_parallel_jobs == 1:
            queue_policy.pop("max_parallel_jobs", None)
        return {
            "schema_version": self.schema_version,
            "run_name": self.run_name,
            "resource_policy": asdict(self.resource_policy),
            "queue_policy": queue_policy,
            "jobs": [job.to_dict() for job in self.jobs],
        }

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict())

    @property
    def source_path(self) -> Path | None:
        value = getattr(self, "_source_path", None)
        return Path(value) if isinstance(value, str) and value else None

    @property
    def source_sha256(self) -> str | None:
        value = getattr(self, "_source_sha256", None)
        return value if isinstance(value, str) else None

    @property
    def source_size_bytes(self) -> int | None:
        value = getattr(self, "_source_size_bytes", None)
        return value if isinstance(value, int) else None


def _relative_path(value: object, field_name: str, *, allow_dot: bool = False) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ManifestError(f"{field_name} must be a non-empty path string")
    path = Path(value)
    if path.is_absolute() or path.drive or ".." in path.parts:
        raise ManifestError(f"{field_name} must stay inside the workspace")
    for part in path.parts:
        if part in {"", "."}:
            continue
        if (
            any(
                character in _WINDOWS_FORBIDDEN_PATH_CHARACTERS
                for character in part
            )
            or part.endswith((" ", "."))
            or _is_windows_reserved_name(part)
        ):
            raise ManifestError(
                f"{field_name} contains a Windows-unsafe path component "
                f"{part!r}"
            )
    normalized = path.as_posix()
    if normalized in ("", ".") and not allow_dot:
        raise ManifestError(f"{field_name} must identify a file")
    return normalized or "."


def _is_windows_reserved_name(value: str) -> bool:
    stem = value.rstrip(" .").split(".", 1)[0].upper()
    return stem in _WINDOWS_RESERVED_NAMES


def _expect_mapping(value: object, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ManifestError(f"{field_name} must be an object")
    return value


def _reject_unknown_keys(
    value: dict[str, Any],
    allowed: set[str],
    field_name: str,
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ManifestError(
            f"{field_name} has unknown field(s): {', '.join(unknown)}"
        )


def _finite_number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ManifestError(f"{field_name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ManifestError(f"{field_name} must be finite")
    return number


def _strict_json_object(text: str, *, source: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ManifestError(f"{source} contains non-finite JSON number {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ManifestError(f"{source} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            text,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except ManifestError:
        raise
    except (
        ValueError,
        json.JSONDecodeError,
        RecursionError,
        UnicodeError,
    ) as exc:
        raise ManifestError(f"{source} is not valid bounded JSON: {exc}") from exc
    _reject_nonfinite(value, source)
    if not isinstance(value, dict):
        raise ManifestError(f"{source} must contain a JSON object")
    return value


def _reject_nonfinite(value: object, source: str) -> None:
    stack = [value]
    while stack:
        current = stack.pop()
        if isinstance(current, float) and not math.isfinite(current):
            raise ManifestError(f"{source} contains an overflowed JSON number")
        if isinstance(current, list):
            stack.extend(current)
        elif isinstance(current, dict):
            stack.extend(current.values())


def _max_json_nesting(text: str) -> int:
    depth = 0
    maximum = 0
    in_string = False
    escaped = False
    for character in text:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in "[{":
            depth += 1
            maximum = max(maximum, depth)
        elif character in "]}" and depth:
            depth -= 1
    return maximum


def _fingerprint(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
