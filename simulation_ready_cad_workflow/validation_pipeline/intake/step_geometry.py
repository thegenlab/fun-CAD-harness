from __future__ import annotations

import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ..runtime.bounded_capture import BoundedPipeCapture
from ..runtime.child_environment import (
    MAX_WINDOWS_ENVIRONMENT_UTF16_UNITS,
    update_environment_case_insensitively,
    windows_environment_utf16_units,
)
from ..runtime.process_containment import (
    WINDOWS_CREATE_SUSPENDED,
    ProcessContainment,
)


MAX_STEP_SOLIDS = 20_000
MAX_ISOLATED_STEP_BYTES = 512 * 1024 * 1024
MAX_WORKER_RESULT_BYTES = 16 * 1024 * 1024
MAX_WORKER_RESULT_NESTING = 64
MAX_WORKER_LOG_BYTES = 8 * 1024 * 1024
MAX_KERNEL_TIMEOUT_SECONDS = 900.0
MAX_KERNEL_RSS_GIB = 6.0
MAX_PREFLIGHT_CPU_PERCENT = 95.0
MAX_PREFLIGHT_MEMORY_PERCENT = 92.0
MAX_CRITICAL_MEMORY_PERCENT = 96.0
MIN_PREFLIGHT_AVAILABLE_GIB = 1.0
MIN_CRITICAL_AVAILABLE_GIB = 0.5
MIN_PREFLIGHT_FREE_DISK_GIB = 1.0
MIN_CRITICAL_FREE_DISK_GIB = 0.5
MAX_CRITICAL_SAMPLES = 30
DEFAULT_KERNEL_TIMEOUT_SECONDS = 120.0
DEFAULT_MAX_KERNEL_RSS_GIB = 1.0
DEFAULT_PREFLIGHT_CPU_PERCENT = 85.0
DEFAULT_PREFLIGHT_MEMORY_PERCENT = 88.0
DEFAULT_PREFLIGHT_AVAILABLE_GIB = 1.0
DEFAULT_CRITICAL_MEMORY_PERCENT = 93.0
DEFAULT_CRITICAL_AVAILABLE_GIB = 0.5
DEFAULT_PREFLIGHT_FREE_DISK_GIB = 5.0
DEFAULT_CRITICAL_FREE_DISK_GIB = 2.0
DEFAULT_CRITICAL_SAMPLES = 3


def _is_windows_platform() -> bool:
    return os.name == "nt"


@dataclass(frozen=True)
class StepSolid:
    solid_id: str
    valid_brep: bool
    bounds_mm: dict[str, list[float]]
    size_mm: list[float]

    def to_dict(self) -> dict[str, object]:
        return {
            "solid_id": self.solid_id,
            "valid_brep": self.valid_brep,
            "bounds_mm": self.bounds_mm,
            "size_mm": self.size_mm,
        }

    @classmethod
    def from_dict(cls, value: object) -> StepSolid:
        if not isinstance(value, dict):
            raise ValueError("solid result must be an object")
        unexpected = set(value) - {
            "solid_id",
            "valid_brep",
            "bounds_mm",
            "size_mm",
        }
        if unexpected:
            raise ValueError(
                "solid result contains unexpected fields: "
                + ", ".join(sorted(str(item) for item in unexpected))
            )
        solid_id = value.get("solid_id")
        valid_brep = value.get("valid_brep")
        bounds = value.get("bounds_mm")
        size = value.get("size_mm")
        if (
            not isinstance(solid_id, str)
            or re.fullmatch(r"solid_[1-9][0-9]*", solid_id) is None
        ):
            raise ValueError("solid_id is invalid")
        if not isinstance(valid_brep, bool):
            raise ValueError("valid_brep is invalid")
        if not isinstance(bounds, dict):
            raise ValueError("bounds_mm is invalid")
        minimum = _finite_vector(bounds.get("min"), "bounds_mm.min")
        maximum = _finite_vector(bounds.get("max"), "bounds_mm.max")
        size_vector = _finite_vector(size, "size_mm")
        expected_size = [
            maximum[index] - minimum[index] for index in range(3)
        ]
        if any(item < 0 for item in expected_size):
            raise ValueError("bounds_mm maximum must not be below minimum")
        for index, (actual, expected) in enumerate(
            zip(size_vector, expected_size, strict=True)
        ):
            tolerance = max(1e-9, abs(expected) * 1e-9)
            if actual < 0 or abs(actual - expected) > tolerance:
                raise ValueError(
                    f"size_mm[{index}] is inconsistent with bounds_mm"
                )
        return cls(
            solid_id=solid_id,
            valid_brep=valid_brep,
            bounds_mm={"min": minimum, "max": maximum},
            size_mm=size_vector,
        )


@dataclass(frozen=True)
class StepGeometryInspection:
    kernel_available: bool
    transferred: bool
    solids: tuple[StepSolid, ...] = ()
    error: str | None = None
    failure_category: str | None = None
    source_sha256: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "kernel_available": self.kernel_available,
            "transferred": self.transferred,
            "solids": [solid.to_dict() for solid in self.solids],
            "error": self.error,
            "failure_category": self.failure_category,
            "source_sha256": self.source_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> StepGeometryInspection:
        if not isinstance(value, dict):
            raise ValueError("inspection result must be an object")
        unexpected = set(value) - {
            "kernel_available",
            "transferred",
            "solids",
            "error",
            "failure_category",
            "source_sha256",
        }
        if unexpected:
            raise ValueError(
                "inspection result contains unexpected fields: "
                + ", ".join(sorted(str(item) for item in unexpected))
            )
        kernel_available = value.get("kernel_available")
        transferred = value.get("transferred")
        solids = value.get("solids", [])
        error = value.get("error")
        failure_category = value.get("failure_category")
        source_sha256 = value.get("source_sha256")
        if not isinstance(kernel_available, bool):
            raise ValueError("kernel_available is invalid")
        if not isinstance(transferred, bool):
            raise ValueError("transferred is invalid")
        if not isinstance(solids, list):
            raise ValueError("solids is invalid")
        if len(solids) > MAX_STEP_SOLIDS:
            raise ValueError("worker returned too many solids")
        if error is not None and not isinstance(error, str):
            raise ValueError("error is invalid")
        if failure_category not in {None, "infrastructure"}:
            raise ValueError("failure_category is invalid")
        if (
            source_sha256 is not None
            and (
                not isinstance(source_sha256, str)
                or re.fullmatch(r"[a-f0-9]{64}", source_sha256) is None
            )
        ):
            raise ValueError("source_sha256 is invalid")
        parsed_solids = tuple(StepSolid.from_dict(item) for item in solids)
        expected_ids = [
            f"solid_{index}"
            for index in range(1, len(parsed_solids) + 1)
        ]
        actual_ids = [solid.solid_id for solid in parsed_solids]
        if actual_ids != expected_ids:
            raise ValueError(
                "solid IDs must be unique and sequential from solid_1"
            )
        if transferred and not kernel_available:
            raise ValueError(
                "transferred geometry requires an available kernel"
            )
        if parsed_solids and not transferred:
            raise ValueError(
                "solid inventory requires transferred geometry"
            )
        if failure_category is not None and transferred:
            raise ValueError(
                "infrastructure failure cannot claim transferred geometry"
            )
        if transferred and error is not None:
            raise ValueError(
                "transferred geometry cannot include an error"
            )
        if not transferred and (error is None or not error.strip()):
            raise ValueError(
                "failed geometry inspection must include an error"
            )
        if failure_category is not None and not error:
            raise ValueError(
                "infrastructure failure must include an error"
            )
        return cls(
            kernel_available=kernel_available,
            transferred=transferred,
            solids=parsed_solids,
            error=error,
            failure_category=failure_category,
            source_sha256=source_sha256,
        )


def inspect_step_geometry(path: Path) -> StepGeometryInspection:
    """Import STEP with OCCT and inventory solids in deterministic order."""
    try:
        from OCP.BRepBndLib import BRepBndLib
        from OCP.BRepCheck import BRepCheck_Analyzer
        from OCP.Bnd import Bnd_Box
        from OCP.IFSelect import IFSelect_RetDone
        from OCP.Interface import Interface_Static
        from OCP.STEPControl import STEPControl_Reader
        from OCP.TopAbs import TopAbs_SOLID
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopoDS import TopoDS
    except (ImportError, ModuleNotFoundError) as exc:
        return StepGeometryInspection(
            kernel_available=False,
            transferred=False,
            error=f"{type(exc).__name__}: {exc}",
        )

    try:
        Interface_Static.SetCVal_s("xstep.cascade.unit", "MM")
        reader = STEPControl_Reader()
        status = reader.ReadFile(str(path))
        if int(status) != int(IFSelect_RetDone):
            return StepGeometryInspection(
                kernel_available=True,
                transferred=False,
                error=f"STEP reader returned status {int(status)}",
            )
        transferred_roots = int(reader.TransferRoots())
        shape = reader.OneShape()
        if transferred_roots <= 0 or shape.IsNull():
            return StepGeometryInspection(
                kernel_available=True,
                transferred=False,
                error="STEP transfer produced no shape",
            )

        solids: list[StepSolid] = []
        explorer = TopExp_Explorer(shape, TopAbs_SOLID)
        while explorer.More():
            if len(solids) >= MAX_STEP_SOLIDS:
                return StepGeometryInspection(
                    kernel_available=True,
                    transferred=False,
                    error=(
                        f"STEP contains more than {MAX_STEP_SOLIDS} solids; "
                        "split the simulation-focused export"
                    ),
                )
            solid = TopoDS.Solid_s(explorer.Current())
            box = Bnd_Box()
            BRepBndLib.AddOptimal_s(solid, box, False, False)
            if box.IsVoid():
                minimum = [0.0, 0.0, 0.0]
                maximum = [0.0, 0.0, 0.0]
            else:
                min_x, min_y, min_z, max_x, max_y, max_z = box.Get()
                minimum = [float(min_x), float(min_y), float(min_z)]
                maximum = [float(max_x), float(max_y), float(max_z)]
            solids.append(
                StepSolid(
                    solid_id=f"solid_{len(solids) + 1}",
                    valid_brep=bool(BRepCheck_Analyzer(solid).IsValid()),
                    bounds_mm={"min": minimum, "max": maximum},
                    size_mm=[
                        maximum[index] - minimum[index] for index in range(3)
                    ],
                )
            )
            explorer.Next()
        return StepGeometryInspection(
            kernel_available=True,
            transferred=True,
            solids=tuple(solids),
        )
    except Exception as exc:
        return StepGeometryInspection(
            kernel_available=True,
            transferred=False,
            error=f"{type(exc).__name__}: {exc}",
        )


def _snapshot_step_source(
    source: Path,
    destination: Path,
    *,
    deadline: float,
) -> tuple[str | None, str | None]:
    try:
        before = source.stat()
    except OSError as exc:
        return None, f"could not inspect STEP source: {type(exc).__name__}: {exc}"
    if before.st_size > MAX_ISOLATED_STEP_BYTES:
        return None, (
            "STEP source exceeds isolated snapshot limit: "
            f"{before.st_size} > {MAX_ISOLATED_STEP_BYTES} bytes"
        )

    digest = hashlib.sha256()
    copied = 0
    try:
        with source.open("rb") as input_stream, destination.open("xb") as output_stream:
            while True:
                if time.monotonic() >= deadline:
                    return None, "STEP snapshot copy timed out within the inspection budget"
                chunk = input_stream.read(1024 * 1024)
                if not chunk:
                    break
                copied += len(chunk)
                if copied > MAX_ISOLATED_STEP_BYTES:
                    return None, (
                        "STEP source grew beyond isolated snapshot limit: "
                        f"more than {MAX_ISOLATED_STEP_BYTES} bytes"
                    )
                digest.update(chunk)
                output_stream.write(chunk)
            output_stream.flush()
            os.fsync(output_stream.fileno())
        after = source.stat()
        snapshot_size = destination.stat().st_size
    except OSError as exc:
        return None, f"could not snapshot STEP source: {type(exc).__name__}: {exc}"

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
    if (
        before_identity != after_identity
        or copied != after.st_size
        or snapshot_size != copied
    ):
        return None, (
            "STEP source changed while the private OCCT snapshot was created"
        )
    return digest.hexdigest(), None


def inspect_step_geometry_isolated(
    path: Path,
    *,
    timeout_seconds: float = DEFAULT_KERNEL_TIMEOUT_SECONDS,
    max_process_tree_rss_gib: float = DEFAULT_MAX_KERNEL_RSS_GIB,
    preflight_cpu_percent: float = DEFAULT_PREFLIGHT_CPU_PERCENT,
    preflight_memory_percent: float = DEFAULT_PREFLIGHT_MEMORY_PERCENT,
    preflight_available_memory_gib: float = DEFAULT_PREFLIGHT_AVAILABLE_GIB,
    critical_memory_percent: float = DEFAULT_CRITICAL_MEMORY_PERCENT,
    critical_available_memory_gib: float = DEFAULT_CRITICAL_AVAILABLE_GIB,
    preflight_free_disk_gib: float = DEFAULT_PREFLIGHT_FREE_DISK_GIB,
    critical_free_disk_gib: float = DEFAULT_CRITICAL_FREE_DISK_GIB,
    critical_samples: int = DEFAULT_CRITICAL_SAMPLES,
) -> StepGeometryInspection:
    """Run OCCT in a monitored subprocess so kernel faults cannot kill intake."""
    numeric_limits = {
        "timeout_seconds": timeout_seconds,
        "max_process_tree_rss_gib": max_process_tree_rss_gib,
        "preflight_cpu_percent": preflight_cpu_percent,
        "preflight_memory_percent": preflight_memory_percent,
        "preflight_available_memory_gib": preflight_available_memory_gib,
        "critical_memory_percent": critical_memory_percent,
        "critical_available_memory_gib": critical_available_memory_gib,
        "preflight_free_disk_gib": preflight_free_disk_gib,
        "critical_free_disk_gib": critical_free_disk_gib,
    }
    for name, value in numeric_limits.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0
        ):
            raise ValueError(f"{name} must be a positive finite number")
    if preflight_cpu_percent > MAX_PREFLIGHT_CPU_PERCENT:
        raise ValueError(
            "preflight_cpu_percent exceeds the hard safety envelope"
        )
    if not 0 < preflight_memory_percent < critical_memory_percent:
        raise ValueError(
            "memory percentages must satisfy 0 < preflight < critical"
        )
    if preflight_memory_percent > MAX_PREFLIGHT_MEMORY_PERCENT:
        raise ValueError(
            "preflight_memory_percent exceeds the hard safety envelope"
        )
    if critical_memory_percent > MAX_CRITICAL_MEMORY_PERCENT:
        raise ValueError(
            "critical_memory_percent exceeds the hard safety envelope"
        )
    if (
        critical_available_memory_gib
        > preflight_available_memory_gib
    ):
        raise ValueError(
            "critical available memory cannot exceed the preflight reserve"
        )
    if preflight_available_memory_gib < MIN_PREFLIGHT_AVAILABLE_GIB:
        raise ValueError(
            "preflight available-memory reserve is below the hard minimum"
        )
    if critical_available_memory_gib < MIN_CRITICAL_AVAILABLE_GIB:
        raise ValueError(
            "critical available-memory reserve is below the hard minimum"
        )
    if critical_free_disk_gib > preflight_free_disk_gib:
        raise ValueError(
            "critical free disk cannot exceed the preflight reserve"
        )
    if preflight_free_disk_gib < MIN_PREFLIGHT_FREE_DISK_GIB:
        raise ValueError(
            "preflight free-disk reserve is below the hard minimum"
        )
    if critical_free_disk_gib < MIN_CRITICAL_FREE_DISK_GIB:
        raise ValueError(
            "critical free-disk reserve is below the hard minimum"
        )
    if timeout_seconds > MAX_KERNEL_TIMEOUT_SECONDS:
        raise ValueError("timeout_seconds exceeds the hard safety envelope")
    if max_process_tree_rss_gib > MAX_KERNEL_RSS_GIB:
        raise ValueError(
            "max_process_tree_rss_gib exceeds the hard safety envelope"
        )
    if (
        not isinstance(critical_samples, int)
        or isinstance(critical_samples, bool)
        or not 1 <= critical_samples <= MAX_CRITICAL_SAMPLES
    ):
        raise ValueError(
            f"critical_samples must be an integer from 1 to "
            f"{MAX_CRITICAL_SAMPLES}"
        )

    input_path = Path(path).resolve()
    try:
        from ..runtime.system_probe import SystemProbe

        worker_temp_root = Path(tempfile.gettempdir()).resolve()
        probe = SystemProbe(disk_path=worker_temp_root)
    except (ImportError, OSError, RuntimeError) as exc:
        return StepGeometryInspection(
            kernel_available=False,
            transferred=False,
            error=f"resource monitor unavailable: {type(exc).__name__}: {exc}",
            failure_category="infrastructure",
        )

    try:
        preflight = probe.sample(0.25)
    except Exception as exc:
        return StepGeometryInspection(
            kernel_available=False,
            transferred=False,
            error=(
                "OCCT resource preflight failed: "
                f"{type(exc).__name__}: {exc}"
            ),
            failure_category="infrastructure",
        )
    parent_log_buffer_gib = 2.0 * MAX_WORKER_LOG_BYTES / 1024**3
    required_available_memory_gib = max(
        float(preflight_available_memory_gib),
        float(max_process_tree_rss_gib)
        + float(critical_available_memory_gib)
        + parent_log_buffer_gib,
    )
    if (
        preflight.cpu_percent > preflight_cpu_percent
        or preflight.memory_percent >= preflight_memory_percent
        or preflight.available_memory_gib
        < required_available_memory_gib
        or preflight.free_disk_gib < preflight_free_disk_gib
    ):
        return StepGeometryInspection(
            kernel_available=False,
            transferred=False,
            error=(
                "OCCT inspection deferred by resource preflight: "
                f"{preflight.cpu_percent:.1f}% CPU, "
                f"{preflight.memory_percent:.1f}% memory used, "
                f"{preflight.available_memory_gib:.2f} GiB available; "
                f"requires CPU at most {preflight_cpu_percent:.1f}%, "
                f"less than {preflight_memory_percent:.1f}% memory used "
                f"and at least {required_available_memory_gib:.2f} GiB "
                "memory available, plus at least "
                f"{preflight_free_disk_gib:.2f} GiB worker-temp disk free; "
                f"observed {preflight.free_disk_gib:.2f} GiB disk free"
            ),
            failure_category="infrastructure",
        )

    try:
        temporary_context = tempfile.TemporaryDirectory(
            prefix="cad-intake-occt-",
            dir=worker_temp_root,
        )
    except OSError as exc:
        return StepGeometryInspection(
            kernel_available=False,
            transferred=False,
            error=(
                "could not create OCCT worker directory: "
                f"{type(exc).__name__}: {exc}"
            ),
            failure_category="infrastructure",
        )
    with temporary_context as directory:
        temporary = Path(directory)
        result_path = temporary / "inspection.json"
        stdout_path = temporary / "worker.stdout.log"
        stderr_path = temporary / "worker.stderr.log"
        snapshot_path = temporary / "source.step"
        deadline = time.monotonic() + timeout_seconds
        source_sha256, snapshot_error = _snapshot_step_source(
            input_path,
            snapshot_path,
            deadline=deadline,
        )
        if snapshot_error is not None or source_sha256 is None:
            return StepGeometryInspection(
                kernel_available=False,
                transferred=False,
                error=snapshot_error or "STEP snapshot failed without a reason",
                failure_category="infrastructure",
            )
        environment = os.environ.copy()
        package_root = str(Path(__file__).resolve().parents[3])
        existing_pythonpath = next(
            (
                value
                for name, value in environment.items()
                if name.casefold() == "pythonpath"
            ),
            None,
        )
        update_environment_case_insensitively(
            environment,
            {
                "OMP_NUM_THREADS": "1",
                "OMP_THREAD_LIMIT": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
                "PYTHONUNBUFFERED": "1",
                "TEMP": str(temporary),
                "TMP": str(temporary),
                "TMPDIR": str(temporary),
                "PYTHONPATH": (
                    package_root
                    if not existing_pythonpath
                    else package_root + os.pathsep + existing_pythonpath
                ),
            },
        )
        if _is_windows_platform():
            environment_utf16_units = windows_environment_utf16_units(
                environment
            )
            if (
                environment_utf16_units
                > MAX_WINDOWS_ENVIRONMENT_UTF16_UNITS
            ):
                return StepGeometryInspection(
                    kernel_available=False,
                    transferred=False,
                    error=(
                        "Windows OCCT worker environment block exceeds "
                        f"{MAX_WINDOWS_ENVIRONMENT_UTF16_UNITS} UTF-16 units: "
                        f"{environment_utf16_units}"
                    ),
                    failure_category="infrastructure",
                )
        command = [
            sys.executable,
            "-m",
            (
                "simulation_ready_cad_workflow.validation_pipeline.intake."
                "step_geometry_worker"
            ),
            "--input",
            str(snapshot_path),
            "--output",
            str(result_path),
        ]
        creationflags = 0
        if os.name == "nt":
            creationflags = (
                subprocess.CREATE_NEW_PROCESS_GROUP
                | subprocess.CREATE_NO_WINDOW
                | WINDOWS_CREATE_SUSPENDED
            )
        process: subprocess.Popen[bytes] | None = None
        captures: list[BoundedPipeCapture] = []
        containment = ProcessContainment()
        try:
            with containment:
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=False,
                    env=environment,
                    shell=False,
                    creationflags=creationflags,
                    start_new_session=os.name != "nt",
                )
                if process.stdout is None or process.stderr is None:
                    raise OSError(
                        "could not capture OCCT worker stdout and stderr"
                    )
                captures = [
                    BoundedPipeCapture(
                        "stdout",
                        process.stdout,
                        stdout_path,
                        MAX_WORKER_LOG_BYTES,
                    ),
                    BoundedPipeCapture(
                        "stderr",
                        process.stderr,
                        stderr_path,
                        MAX_WORKER_LOG_BYTES,
                    ),
                ]
                for capture in captures:
                    capture.start()
                probe.lower_process_tree_priority(process.pid)
                containment.attach_and_resume(
                    process,
                    max_job_memory_bytes=int(
                        max_process_tree_rss_gib * 1024**3
                    ),
                )
                pressure_count = 0
                rss_measurement_failures = 0
                stop_reason: str | None = None
                while process.poll() is None:
                    now = time.monotonic()
                    if now >= deadline:
                        stop_reason = (
                            "OCCT inspection timed out after "
                            f"{timeout_seconds:.1f} seconds"
                        )
                        break
                    snapshot = probe.sample(0.0)
                    tree_rss_bytes = probe.process_tree_rss_bytes(
                        process.pid
                    )
                    if tree_rss_bytes <= 0:
                        rss_measurement_failures += 1
                    else:
                        rss_measurement_failures = 0
                    tree_rss_gib = tree_rss_bytes / 1024**3
                    critical = (
                        snapshot.memory_percent >= critical_memory_percent
                        or snapshot.available_memory_gib
                        <= critical_available_memory_gib
                        or snapshot.free_disk_gib
                        <= critical_free_disk_gib
                    )
                    pressure_count = pressure_count + 1 if critical else 0
                    if rss_measurement_failures >= critical_samples:
                        stop_reason = (
                            "OCCT process-tree RSS could not be measured for "
                            f"{rss_measurement_failures} consecutive samples"
                        )
                        break
                    if tree_rss_gib > max_process_tree_rss_gib:
                        stop_reason = (
                            "OCCT process tree exceeded memory limit: "
                            f"{tree_rss_gib:.2f} GiB > "
                            f"{max_process_tree_rss_gib:.2f} GiB"
                        )
                        break
                    oversized_capture = next(
                        (
                            capture
                            for capture in captures
                            if capture.exceeds_limit()
                        ),
                        None,
                    )
                    if oversized_capture is not None:
                        stop_reason = (
                            f"OCCT worker {oversized_capture.name} exceeded "
                            f"the {MAX_WORKER_LOG_BYTES}-byte log limit"
                        )
                        break
                    if pressure_count >= critical_samples:
                        stop_reason = (
                            "OCCT inspection stopped under critical system "
                            f"resource pressure: {snapshot.memory_percent:.1f}% "
                            f"memory used, {snapshot.available_memory_gib:.2f} "
                            "GiB memory available, "
                            f"{snapshot.free_disk_gib:.2f} GiB disk free"
                        )
                        break
                    probe.lower_process_tree_priority(process.pid)
                    time.sleep(min(0.5, max(0.0, deadline - now)))
                if stop_reason is not None:
                    _terminate_process_tree(process)
                    containment.close()
                    capture_failure = _finish_worker_captures(captures)
                    if capture_failure is not None:
                        stop_reason = f"{stop_reason}; {capture_failure}"
                    return StepGeometryInspection(
                        kernel_available=True,
                        transferred=False,
                        error=stop_reason,
                        failure_category="infrastructure",
                    )
                return_code = process.wait()
        except OSError as exc:
            cleanup_error: str | None = None
            if process is not None:
                try:
                    _terminate_process_tree(process)
                except (OSError, RuntimeError) as cleanup_exc:
                    cleanup_error = (
                        f"; cleanup also failed: {type(cleanup_exc).__name__}: "
                        f"{cleanup_exc}"
                    )
            capture_failure = _finish_worker_captures(captures)
            if capture_failure is not None:
                cleanup_error = (
                    f"{cleanup_error or ''}; {capture_failure}"
                )
            return StepGeometryInspection(
                kernel_available=False,
                transferred=False,
                error=(
                    "could not start or finalize OCCT worker: "
                    f"{type(exc).__name__}: {exc}{cleanup_error or ''}"
                ),
                failure_category="infrastructure",
            )
        except Exception as exc:
            cleanup_error = None
            if process is not None:
                try:
                    _terminate_process_tree(process)
                except (OSError, RuntimeError) as cleanup_exc:
                    cleanup_error = (
                        f"; cleanup also failed: {type(cleanup_exc).__name__}: "
                        f"{cleanup_exc}"
                    )
            capture_failure = _finish_worker_captures(captures)
            if capture_failure is not None:
                cleanup_error = (
                    f"{cleanup_error or ''}; {capture_failure}"
                )
            return StepGeometryInspection(
                kernel_available=True,
                transferred=False,
                error=(
                    "OCCT worker monitoring failed: "
                    f"{type(exc).__name__}: {exc}{cleanup_error or ''}"
                ),
                failure_category="infrastructure",
            )
        except BaseException:
            # KeyboardInterrupt and SystemExit must propagate, but the POSIX
            # containment object does not kill the child for us.
            if process is not None and process.poll() is None:
                try:
                    _terminate_process_tree(process)
                except (OSError, RuntimeError):
                    pass
            _finish_worker_captures(captures)
            raise

        try:
            if process is not None:
                _terminate_process_tree(process)
        except (OSError, RuntimeError) as exc:
            capture_failure = _finish_worker_captures(captures)
            detail = f"{type(exc).__name__}: {exc}"
            if capture_failure is not None:
                detail = f"{detail}; {capture_failure}"
            return StepGeometryInspection(
                kernel_available=True,
                transferred=False,
                error=f"OCCT worker process-tree cleanup failed: {detail}",
                failure_category="infrastructure",
            )

        capture_failure = _finish_worker_captures(captures)
        if capture_failure is not None:
            return StepGeometryInspection(
                kernel_available=True,
                transferred=False,
                error=capture_failure,
                failure_category="infrastructure",
            )
        if return_code != 0:
            return StepGeometryInspection(
                kernel_available=True,
                transferred=False,
                error=(
                    f"OCCT worker exited with code {return_code}: "
                    f"{_tail_text(stderr_path, 4096)}"
                ),
                failure_category="infrastructure",
            )
        try:
            raw_result = _read_bounded_worker_result(result_path)
            payload = _strict_worker_result_json(
                raw_result.decode("utf-8")
            )
            inspection = StepGeometryInspection.from_dict(payload)
            return replace(inspection, source_sha256=source_sha256)
        except (
            OSError,
            RecursionError,
            UnicodeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            return StepGeometryInspection(
                kernel_available=True,
                transferred=False,
                error=f"invalid OCCT worker result: {type(exc).__name__}: {exc}",
                failure_category="infrastructure",
            )


def _finish_worker_captures(
    captures: list[BoundedPipeCapture],
) -> str | None:
    deadline = time.monotonic() + 5.0
    problems: list[str] = []
    for capture in captures:
        metrics = capture.finish(
            timeout_seconds=max(0.0, deadline - time.monotonic())
        )
        if metrics["truncated"]:
            problems.append(
                f"OCCT worker {capture.name} exceeded the "
                f"{capture.max_bytes}-byte log limit"
            )
        if metrics["error"] is not None:
            problems.append(
                f"OCCT worker {capture.name} capture failed: "
                f"{metrics['error']}"
            )
    return "; ".join(problems) or None

def _read_bounded_worker_result(path: Path) -> bytes:
    before = path.stat()
    if before.st_size > MAX_WORKER_RESULT_BYTES:
        raise ValueError("worker result exceeds size limit")
    with path.open("rb") as stream:
        raw = stream.read(MAX_WORKER_RESULT_BYTES + 1)
    after = path.stat()
    if (
        after.st_size > MAX_WORKER_RESULT_BYTES
        or len(raw) > MAX_WORKER_RESULT_BYTES
    ):
        raise ValueError("worker result exceeds size limit")
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
        raise ValueError("worker result changed while being read")
    return raw


def _worker_result_json_nesting(text: str) -> int:
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


def _strict_worker_result_json(text: str) -> dict[str, Any]:
    nesting = _worker_result_json_nesting(text)
    if nesting > MAX_WORKER_RESULT_NESTING:
        raise ValueError(
            f"OCCT worker result nesting {nesting} exceeds "
            f"{MAX_WORKER_RESULT_NESTING} level limit"
        )
    def reject_duplicates(
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(
                    f"duplicate OCCT worker result key {key!r}"
                )
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(
            f"non-finite OCCT worker result number {value}"
        )

    def parse_integer(value: str) -> int:
        if len(value.lstrip("-")) > 309:
            raise ValueError("OCCT worker result integer is too large")
        return int(value)

    value = json.loads(
        text,
        object_pairs_hook=reject_duplicates,
        parse_constant=reject_constant,
        parse_int=parse_integer,
    )
    if not isinstance(value, dict):
        raise ValueError("OCCT worker result root must be an object")
    return value


def _terminate_process_tree(process: subprocess.Popen[Any]) -> None:
    from ..runtime.system_probe import process_exists, terminate_process_tree

    grace = 5.0
    deadline = time.monotonic() + grace
    psutil_budget = max(0.0, (deadline - time.monotonic()) * 0.75)
    survivors = terminate_process_tree(process.pid, psutil_budget)
    if os.name == "nt" and (process.poll() is None or survivors):
        for pid in sorted({process.pid, *survivors}):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                    timeout=remaining,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
            except (OSError, subprocess.TimeoutExpired):
                continue
    elif os.name != "nt":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    if process.poll() is None:
        remaining = deadline - time.monotonic()
        if remaining > 0:
            try:
                process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                process.kill()
        else:
            process.kill()
    if process.poll() is None:
        remaining = deadline - time.monotonic()
        if remaining > 0:
            try:
                process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                pass
    live_survivors = tuple(
        pid for pid in survivors if process_exists(pid)
    )
    if process.poll() is None or live_survivors:
        raise RuntimeError(
            "OCCT worker process tree could not be confirmed stopped within "
            f"5 seconds; surviving PIDs: {list(live_survivors)}"
        )


def _tail_text(path: Path, limit: int) -> str:
    read_limit = max(0, int(limit))
    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - read_limit), os.SEEK_SET)
            data = stream.read(read_limit)
    except OSError as exc:
        return f"stderr unavailable: {exc}"
    return data.decode("utf-8", errors="replace").strip()


def _finite_vector(value: object, label: str) -> list[float]:
    if (
        not isinstance(value, list)
        or len(value) != 3
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            for item in value
        )
    ):
        raise ValueError(f"{label} must contain three finite numbers")
    return [float(item) for item in value]
