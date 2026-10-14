"""Isolate project adapter import and callbacks in a bounded child process."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

from ..runtime.bounded_subprocess import (
    BoundedSubprocessError,
    run_bounded_subprocess,
)
from .adapters import AdapterSpec, adapter_from_dict, adapter_to_dict
from .family_plugins import GeneralizedCoordinateSpec, MechanismAdapter
from .io import validator_import_root
from .models import LoadCase, MechanismPlan, MechanismValidationError


GIB = 1024**3
DEFAULT_PROBE_SECONDS = 30.0
DEFAULT_CALL_SECONDS = 60.0
DEFAULT_LANE_SECONDS = 1800.0
DEFAULT_RSS_BYTES = 1 * GIB
_ISOLATE_ENV = "SIMREADY_PROJECT_ADAPTER_ISOLATE"
_LANE_METHODS = frozenset(
    {"materialize", "simulate", "simulate_whole_machine"}
)


def isolation_enabled() -> bool:
    return os.environ.get(_ISOLATE_ENV, "1") != "0"


def adapter_call_timeout_seconds(
    method: str, default_call_seconds: float = DEFAULT_CALL_SECONDS
) -> float:
    """Probe-like callbacks stay short; CAD lanes use the queue-scale budget."""

    call_budget = default_call_seconds
    raw_call = os.environ.get("SIMREADY_PROJECT_ADAPTER_CALL_SECONDS")
    if raw_call is not None and str(raw_call).strip():
        try:
            call_budget = float(raw_call)
        except ValueError:
            call_budget = default_call_seconds
    if method not in _LANE_METHODS:
        return max(0.1, call_budget)
    raw_lane = os.environ.get("SIMREADY_PROJECT_ADAPTER_LANE_SECONDS")
    if raw_lane is not None and str(raw_lane).strip():
        try:
            return max(0.1, float(raw_lane))
        except ValueError:
            pass
    return max(0.1, call_budget, DEFAULT_LANE_SECONDS)


def project_adapter_descriptor(plugin: Any) -> dict[str, Any]:
    coordinate = plugin.coordinate
    return {
        "plugin_id": plugin.plugin_id,
        "families": sorted(plugin.families),
        "coordinate": {
            "position_field": coordinate.position_field,
            "velocity_field": coordinate.velocity_field,
            "position_unit": coordinate.position_unit,
            "velocity_unit": coordinate.velocity_unit,
            "input_wrench_type": coordinate.input_wrench_type,
            "input_unit": coordinate.input_unit,
            "component_count": coordinate.component_count,
        },
        "worker_backend": plugin.worker_backend,
        "whole_machine_backend": plugin.whole_machine_backend,
        "wrench_policy_profiles": sorted(plugin.wrench_policy_profiles),
        "implementation_capability_ids": sorted(
            plugin.implementation_capability_ids
        ),
        "model_boundary_ids": sorted(plugin.model_boundary_ids),
        "geometry_proof_kinds": sorted(plugin.geometry_proof_kinds),
        "initial_geometry_evidence_keys": sorted(
            plugin.initial_geometry_evidence_keys
        ),
        "whole_machine_artifact_branch": plugin.whole_machine_artifact_branch,
        "extra_family_spec_fields": sorted(plugin.extra_family_spec_fields),
        "required_trace_audit_sections": sorted(
            plugin.required_trace_audit_sections
        ),
        "task_input_actuation_spans_all_inputs": bool(
            plugin.task_input_actuation_spans_all_inputs
        ),
        "lane_fingerprint_dropped_keys": list(
            plugin.lane_fingerprint_dropped_keys
        ),
    }


def encode_worker_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return {"__type__": "Path", "value": str(value)}
    if isinstance(value, MechanismPlan):
        return {"__type__": "MechanismPlan", "value": value.to_dict()}
    if isinstance(value, AdapterSpec):
        return {"__type__": "AdapterSpec", "value": adapter_to_dict(value)}
    if isinstance(value, LoadCase):
        return {"__type__": "LoadCase", "value": value.to_dict()}
    if isinstance(value, SimpleNamespace):
        return {
            "__type__": "SimpleNamespace",
            "value": encode_worker_value(vars(value)),
        }
    if isinstance(value, tuple):
        return {
            "__type__": "tuple",
            "value": [encode_worker_value(item) for item in value],
        }
    if isinstance(value, frozenset):
        return {
            "__type__": "frozenset",
            "value": [encode_worker_value(item) for item in value],
        }
    if isinstance(value, set):
        return {
            "__type__": "set",
            "value": [encode_worker_value(item) for item in value],
        }
    if isinstance(value, list):
        return [encode_worker_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): encode_worker_value(item) for key, item in value.items()
        }
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return {
            "__type__": type(value).__name__,
            "value": encode_worker_value(value.to_dict()),
        }
    request_file = getattr(value, "request_file", None)
    plan = getattr(value, "plan", None)
    if request_file is not None and isinstance(plan, MechanismPlan):
        return {
            "__type__": "ExecutionRequest",
            "value": {
                "request_file": str(value.request_file),
                "output_file": str(value.output_file),
                "branch": value.branch,
                "artifact_branch": value.artifact_branch,
                "adapter": adapter_to_dict(value.adapter),
                "plan": value.plan.to_dict(),
                "load_case": value.load_case.to_dict(),
                "reference_trace_file": (
                    None
                    if value.reference_trace_file is None
                    else str(value.reference_trace_file)
                ),
                "functional_geometry_evidence": value.functional_geometry_evidence,
            },
        }
    raise MechanismValidationError(
        "PROJECT_ADAPTER_RESULT_NOT_DATA",
        f"project adapter returned a non-data object: {type(value).__name__}",
    )


def decode_worker_value(value: Any, *, plugin: Any | None = None) -> Any:
    if isinstance(value, list):
        return [decode_worker_value(item, plugin=plugin) for item in value]
    if not isinstance(value, dict) or "__type__" not in value:
        if isinstance(value, dict):
            return {
                key: decode_worker_value(item, plugin=plugin)
                for key, item in value.items()
            }
        return value
    kind = value["__type__"]
    inner = value.get("value")
    if kind == "Path":
        return Path(str(inner))
    if kind == "MechanismPlan":
        return MechanismPlan.from_dict(inner)
    if kind == "AdapterSpec":
        return adapter_from_dict(inner)
    if kind == "LoadCase":
        return LoadCase.from_dict(inner)
    if kind == "SimpleNamespace":
        decoded = decode_worker_value(inner, plugin=plugin)
        return SimpleNamespace(**decoded)
    if kind == "tuple":
        return tuple(decode_worker_value(item, plugin=plugin) for item in inner)
    if kind == "frozenset":
        return frozenset(
            decode_worker_value(item, plugin=plugin) for item in inner
        )
    if kind == "set":
        return {
            decode_worker_value(item, plugin=plugin) for item in inner
        }
    if kind == "IsolatedIdealModel":
        return IsolatedIdealModelMarker()
    if kind == "ExecutionRequest":
        from .worker import ExecutionRequest

        payload = decode_worker_value(inner, plugin=plugin)
        return ExecutionRequest(
            request_file=Path(payload["request_file"]),
            output_file=Path(payload["output_file"]),
            branch=payload["branch"],
            artifact_branch=payload["artifact_branch"],
            adapter=payload["adapter"],
            mechanism_plugin=plugin,
            plan=payload["plan"],
            load_case=payload["load_case"],
            reference_trace_file=(
                None
                if payload.get("reference_trace_file") is None
                else Path(payload["reference_trace_file"])
            ),
            functional_geometry_evidence=payload.get(
                "functional_geometry_evidence"
            ),
        )
    return decode_worker_value(inner, plugin=plugin)


class IsolatedIdealModelMarker:
    """Decoded marker: the live ideal model stays in the child process."""


class IsolatedIdealMechanismModel:
    """Parent-side proxy for ``ideal_model().evaluate_samples``."""

    model_id = "isolated.project.ideal_model"

    def __init__(
        self,
        adapter: "IsolatedProjectAdapter",
        plan: MechanismPlan,
        family_spec: Mapping[str, Any] | None,
    ) -> None:
        self._adapter = adapter
        self._plan = plan
        self._family_spec = family_spec

    def evaluate_samples(self, plan: MechanismPlan, samples: Any) -> tuple[Any, Any]:
        result = self._adapter._call(
            "ideal_model_evaluate_samples",
            plan,
            self._family_spec,
            list(samples),
        )
        if not isinstance(result, Mapping):
            raise TypeError(
                "isolated ideal_model.evaluate_samples must return sample data",
            )
        evaluated = result.get("samples")
        domain = result.get("domain")
        return evaluated, domain


class IsolatedParsedFamilySpec:
    """Parent-side family spec with ``.bodies`` and ``to_dict()`` after JSON."""

    def __init__(self, payload: Mapping[str, Any]) -> None:
        self._payload = {
            str(key): value for key, value in dict(payload).items()
        }
        bodies: list[Any] = []
        for item in self._payload.get("bodies") or ():
            if isinstance(item, Mapping):
                bodies.append(
                    SimpleNamespace(
                        fixed=bool(item.get("fixed")),
                        group_ids=tuple(
                            str(group) for group in (item.get("group_ids") or ())
                        ),
                    )
                )
                continue
            if hasattr(item, "fixed") and hasattr(item, "group_ids"):
                bodies.append(item)
                continue
            raise TypeError(
                "parse_family_spec bodies must expose fixed and group_ids"
            )
        self.bodies = tuple(bodies)

    def to_dict(self) -> dict[str, Any]:
        return dict(self._payload)


def reconstruct_parsed_family_spec(result: Any) -> Any:
    if hasattr(result, "bodies") and callable(getattr(result, "to_dict", None)):
        return result
    if isinstance(result, Mapping):
        return IsolatedParsedFamilySpec(result)
    raise TypeError(
        "parse_family_spec must return family-spec data with bodies"
    )


def load_isolated_project_adapter(
    package_dir: str | Path,
    relative: str,
    expected_sha256: str,
    *,
    probe_timeout_seconds: float = DEFAULT_PROBE_SECONDS,
    call_timeout_seconds: float = DEFAULT_CALL_SECONDS,
    max_rss_bytes: int = DEFAULT_RSS_BYTES,
) -> Any:
    descriptor = _run_adapter_worker(
        package_dir,
        relative,
        expected_sha256,
        action="probe",
        timeout_seconds=probe_timeout_seconds,
        max_rss_bytes=max_rss_bytes,
    )["descriptor"]
    return IsolatedProjectAdapter(
        package_dir=package_dir,
        relative=relative,
        expected_sha256=expected_sha256,
        descriptor=descriptor,
        call_timeout_seconds=call_timeout_seconds,
        max_rss_bytes=max_rss_bytes,
    )


def invoke_isolated_project_adapter(
    *,
    package_dir: str | Path,
    relative: str,
    expected_sha256: str,
    method: str,
    args: tuple[Any, ...] = (),
    kwargs: Mapping[str, Any] | None = None,
    timeout_seconds: float = DEFAULT_CALL_SECONDS,
    max_rss_bytes: int = DEFAULT_RSS_BYTES,
) -> Any:
    payload = _run_adapter_worker(
        package_dir,
        relative,
        expected_sha256,
        action="call",
        method=method,
        request={
            "method": method,
            "args": [encode_worker_value(item) for item in args],
            "kwargs": {
                str(key): encode_worker_value(value)
                for key, value in dict(kwargs or {}).items()
            },
        },
        timeout_seconds=timeout_seconds,
        max_rss_bytes=max_rss_bytes,
    )
    return decode_worker_value(payload.get("result"))


def _run_adapter_worker(
    package_dir: str | Path,
    relative: str,
    expected_sha256: str,
    *,
    action: str,
    method: str = "",
    request: Mapping[str, Any] | None = None,
    timeout_seconds: float,
    max_rss_bytes: int,
) -> dict[str, Any]:
    import_root = str(validator_import_root())
    environment = os.environ.copy()
    current = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = (
        import_root if not current else import_root + os.pathsep + current
    )
    environment[_ISOLATE_ENV] = "0"
    command = [
        sys.executable,
        "-m",
        "simulation_ready_cad_workflow.validation_pipeline.mechanism.project_adapter_worker",
        action,
        "--package",
        str(Path(package_dir).resolve()),
        "--relative",
        relative,
        "--sha256",
        expected_sha256,
    ]
    if method:
        command.extend(["--method", method])
    try:
        completed = run_bounded_subprocess(
            command,
            timeout_seconds=timeout_seconds,
            max_rss_bytes=max_rss_bytes,
            env=environment,
            input_bytes=(
                b""
                if request is None
                else (
                    json.dumps(
                        request,
                        ensure_ascii=True,
                        allow_nan=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                ).encode("utf-8")
            ),
        )
    except BoundedSubprocessError as exc:
        raise MechanismValidationError(exc.code, exc.message) from exc
    try:
        payload = json.loads(completed.stdout.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        detail = completed.stderr.decode("utf-8", errors="replace")[:2048]
        raise MechanismValidationError(
            "PROJECT_MECHANISM_ADAPTER_LOAD_FAILED",
            f"isolated adapter worker returned invalid JSON: {detail}",
        ) from exc
    if not isinstance(payload, dict) or payload.get("ok") is not True:
        failure = (
            payload.get("failure")
            if isinstance(payload, Mapping)
            and isinstance(payload.get("failure"), Mapping)
            else None
        )
        details = (
            decode_worker_value(payload.get("details"))
            if isinstance(payload, Mapping)
            and payload.get("details") is not None
            else None
        )
        raise MechanismValidationError(
            str((payload or {}).get("code") or "PROJECT_MECHANISM_ADAPTER_LOAD_FAILED"),
            str((payload or {}).get("message") or "isolated adapter worker failed"),
            details=(details if isinstance(details, Mapping) else None),
            failure=failure,
        )
    return payload


class IsolatedProjectAdapter(MechanismAdapter):
    """Parent-side proxy: every project callback runs in a bounded child."""

    plugin_id = "project.isolated"
    families = frozenset()
    coordinate = GeneralizedCoordinateSpec(
        "position", "velocity", "m", "m/s", "force", "N"
    )
    worker_backend = "isolated_project_adapter"
    whole_machine_backend = "isolated_project_adapter"

    def __init__(
        self,
        *,
        package_dir: str | Path,
        relative: str,
        expected_sha256: str,
        descriptor: Mapping[str, Any],
        call_timeout_seconds: float,
        max_rss_bytes: int,
    ) -> None:
        self._package_dir = Path(package_dir)
        self._relative = relative
        self._expected_sha256 = expected_sha256
        self._descriptor = dict(descriptor)
        self._call_timeout_seconds = call_timeout_seconds
        self._max_rss_bytes = max_rss_bytes
        self.plugin_id = str(descriptor["plugin_id"])
        self.families = frozenset(str(item) for item in descriptor["families"])
        coordinate = descriptor["coordinate"]
        self.coordinate = GeneralizedCoordinateSpec(
            position_field=str(coordinate["position_field"]),
            velocity_field=str(coordinate["velocity_field"]),
            position_unit=str(coordinate["position_unit"]),
            velocity_unit=str(coordinate["velocity_unit"]),
            input_wrench_type=str(coordinate["input_wrench_type"]),
            input_unit=str(coordinate["input_unit"]),
            component_count=int(coordinate.get("component_count") or 1),
        )
        self.worker_backend = str(descriptor["worker_backend"])
        self.whole_machine_backend = str(descriptor["whole_machine_backend"])
        self.wrench_policy_profiles = frozenset(
            str(item) for item in descriptor.get("wrench_policy_profiles") or ()
        )
        self.implementation_capability_ids = frozenset(
            str(item)
            for item in descriptor.get("implementation_capability_ids") or ()
        )
        self.model_boundary_ids = frozenset(
            str(item) for item in descriptor.get("model_boundary_ids") or ()
        )
        self.geometry_proof_kinds = frozenset(
            str(item)
            for item in descriptor.get("geometry_proof_kinds") or ()
        )
        self.initial_geometry_evidence_keys = frozenset(
            str(item)
            for item in descriptor.get("initial_geometry_evidence_keys") or ()
        )
        self.whole_machine_artifact_branch = str(
            descriptor.get("whole_machine_artifact_branch") or "reduced"
        )
        self.extra_family_spec_fields = frozenset(
            str(item)
            for item in descriptor.get("extra_family_spec_fields") or ()
        )
        self.required_trace_audit_sections = frozenset(
            str(item)
            for item in descriptor.get("required_trace_audit_sections") or ()
        )
        self.task_input_actuation_spans_all_inputs = bool(
            descriptor.get("task_input_actuation_spans_all_inputs")
        )
        self.lane_fingerprint_dropped_keys = tuple(
            str(item)
            for item in descriptor.get("lane_fingerprint_dropped_keys") or ()
        )

    def _call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        return invoke_isolated_project_adapter(
            package_dir=self._package_dir,
            relative=self._relative,
            expected_sha256=self._expected_sha256,
            method=method,
            args=args,
            kwargs=kwargs,
            timeout_seconds=adapter_call_timeout_seconds(
                method, self._call_timeout_seconds
            ),
            max_rss_bytes=self._max_rss_bytes,
        )

    def supports(self, plan: MechanismPlan, adapter: AdapterSpec) -> bool:
        result = self._call("supports", plan, adapter)
        if not isinstance(result, bool):
            raise TypeError("supports must return bool")
        return result

    def materialize(
        self,
        plan: MechanismPlan,
        adapter: AdapterSpec,
        *,
        workspace_root: str | Path,
        output_dir: str | Path,
    ) -> AdapterSpec:
        result = self._call(
            "materialize",
            plan,
            adapter,
            workspace_root=workspace_root,
            output_dir=output_dir,
        )
        if isinstance(result, AdapterSpec):
            return result
        if isinstance(result, Mapping):
            try:
                return adapter_from_dict(result)
            except (KeyError, TypeError, ValueError) as exc:
                raise TypeError("materialize returned invalid adapter data") from exc
        raise TypeError("materialize must return adapter data")

    def load_cases(self, plan: MechanismPlan) -> tuple[LoadCase, ...]:
        result = self._call("load_cases", plan)
        if result is None:
            raise TypeError("load_cases must return a tuple of LoadCase data")
        if not isinstance(result, tuple):
            raise TypeError("load_cases must return a tuple of LoadCase data")
        decoded: list[LoadCase] = []
        for item in result:
            if isinstance(item, LoadCase):
                decoded.append(item)
            elif isinstance(item, Mapping):
                try:
                    decoded.append(LoadCase.from_dict(dict(item)))
                except (KeyError, TypeError, ValueError) as exc:
                    raise TypeError("load_cases returned invalid LoadCase data") from exc
            else:
                raise TypeError("load_cases must return LoadCase data")
        return tuple(decoded)

    def expected_reduced_ideal_links(
        self,
        plan: MechanismPlan,
        adapter: AdapterSpec,
    ) -> tuple[str, ...]:
        result = self._call("expected_reduced_ideal_links", plan, adapter)
        if not isinstance(result, tuple):
            raise TypeError("expected_reduced_ideal_links must return a tuple")
        return result

    def physical_supports(self, plan: MechanismPlan, adapter: AdapterSpec) -> tuple:
        result = self._call("physical_supports", plan, adapter)
        if not isinstance(result, tuple):
            raise TypeError("physical_supports must return a tuple")
        return result

    def constitutive_supports(
        self, plan: MechanismPlan, adapter: AdapterSpec
    ) -> tuple:
        result = self._call("constitutive_supports", plan, adapter)
        if not isinstance(result, tuple):
            raise TypeError("constitutive_supports must return a tuple")
        return result

    def validate_family_spec(self, adapter: AdapterSpec) -> None:
        self._call("validate_family_spec", adapter)
        return None

    def simulate(self, execution: Any, workspace: Path) -> dict[str, Any]:
        result = self._call("simulate", execution, workspace)
        return dict(result) if isinstance(result, Mapping) else result

    def wrench_body_roles(self, family_spec: Mapping[str, Any]) -> Any:
        return self._call("wrench_body_roles", family_spec)

    def load_envelope(self, load_cases: tuple, comparisons: tuple) -> Any:
        return self._call("load_envelope", load_cases, comparisons)

    def certificate_model(self, adapter: AdapterSpec) -> Any:
        return self._call("certificate_model", adapter)

    def replacement_topology(self, adapter: AdapterSpec) -> Any:
        return self._call("replacement_topology", adapter)

    def build_whole_machine_execution_plan(self, **kwargs: Any) -> Any:
        return self._call("build_whole_machine_execution_plan", **kwargs)

    def load_whole_machine_plan(self, value: Mapping[str, Any]) -> Any:
        return self._call("load_whole_machine_plan", value)

    def parse_family_spec(self, raw: Mapping[str, Any]) -> Any:
        return reconstruct_parsed_family_spec(self._call("parse_family_spec", raw))

    def required_geometry_proofs(
        self,
        adapter: AdapterSpec,
        *,
        capsule_id: str,
    ) -> tuple[Any, ...]:
        from .joint_realization import RevoluteProofRequirement

        values = self._call(
            "required_geometry_proofs",
            adapter,
            capsule_id=capsule_id,
        )
        return tuple(
            RevoluteProofRequirement.from_dict(item)
            if isinstance(item, Mapping)
            and item.get("proof_kind") == "revolute.pin_bore.v1"
            else item
            for item in values or ()
        )

    def ideal_model(self, plan: MechanismPlan, family_spec: Mapping[str, Any] | None) -> Any:
        result = self._call("ideal_model", plan, family_spec)
        if result is None:
            return None
        return IsolatedIdealMechanismModel(self, plan, family_spec)

    def ideal_comparison_protocol(
        self,
        plan: MechanismPlan,
        family_spec: Mapping[str, Any] | None,
    ) -> Any:
        return self._call("ideal_comparison_protocol", plan, family_spec)

    def physical_behavior(
        self,
        plan: MechanismPlan,
        load_case: LoadCase,
        trace: dict[str, Any],
        family_spec: Mapping[str, Any] | None,
    ) -> Any:
        return self._call("physical_behavior", plan, load_case, trace, family_spec)

    def task_channel_layout(
        self,
        plan: MechanismPlan,
        family_spec: Mapping[str, Any],
        body_ids: tuple[str, ...],
    ) -> Any:
        return self._call("task_channel_layout", plan, family_spec, body_ids)

    def simulate_whole_machine(
        self, plan: Any, *, workspace_root: Path
    ) -> Any:
        return self._call("simulate_whole_machine", plan, workspace_root=workspace_root)
