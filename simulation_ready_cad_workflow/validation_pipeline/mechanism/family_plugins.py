from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from ..contracts.failure_envelope import (
    EvidenceValidity,
    FailureOwner,
    make_failure_envelope,
    make_plugin_failure_envelope,
)
from .adapters import (
    DERIVED_COMPOUND_ADAPTER_ID,
    DERIVED_PLAIN_PRISMATIC_ADAPTER_ID,
    DERIVED_PLAIN_REVOLUTE_ADAPTER_ID,
    AdapterSpec,
    adapter_from_dict,
    get_adapter,
)
from .io import ensure_inside, sha256_file
from .models import MechanismPlan, MechanismValidationError
from .plugin_boundary import invoke_plugin_callback, plugin_boundary_error
from .wrench_ledger import WrenchPolicy


if TYPE_CHECKING:
    from .models import LoadCase
    from .worker import ExecutionRequest


def geometry_proof_contract_error(
    plugin: "MechanismAdapter",
    code: str,
    message: str,
) -> MechanismValidationError:
    try:
        registration = plugin_registration_for(str(plugin.plugin_id))
    except MechanismValidationError:
        registration = None
    return MechanismValidationError(
        code,
        message,
        failure=make_plugin_failure_envelope(
            code,
            registration=registration,
            plugin_id=str(plugin.plugin_id),
            plugin_kind="mechanism",
            source_component="mechanism.geometry_proof_contract",
            message=message,
            evidence_validity=EvidenceValidity.NOT_APPLICABLE,
        ),
    )


@dataclass(frozen=True)
class GeneralizedCoordinateSpec:
    """Public kinematic contract owned by a mechanism-family plugin."""

    position_field: str
    velocity_field: str
    position_unit: str
    velocity_unit: str
    input_wrench_type: str
    input_unit: str
    component_count: int = 1


@dataclass(frozen=True)
class CompliantSupportSpec:
    """Explicit non-ideal physical support owned by a family plugin."""

    support_id: str
    chrono_type: str
    translational_stiffness_n_m: tuple[float, float, float]
    translational_damping_n_s_m: tuple[float, float, float]
    rotational_stiffness_n_m_rad: tuple[float, float, float]
    rotational_damping_n_m_s_rad: tuple[float, float, float]
    free_coordinate: str
    source: str
    preload_force_local_n: tuple[float, float, float] = (0.0, 0.0, 0.0)
    force_application_point: str = "joint_origin"

    def __post_init__(self) -> None:
        if self.force_application_point not in {"joint_origin", "body_center"}:
            raise ValueError(
                "force_application_point must be joint_origin or body_center"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "support_id": self.support_id,
            "chrono_type": self.chrono_type,
            "application_frame": (
                "moving_body_center"
                if self.force_application_point == "body_center"
                else "joint_origin_with_local_z_on_mechanism_axis"
            ),
            "translational_stiffness_n_m": list(
                self.translational_stiffness_n_m
            ),
            "translational_damping_n_s_m": list(
                self.translational_damping_n_s_m
            ),
            "rotational_stiffness_n_m_rad": list(
                self.rotational_stiffness_n_m_rad
            ),
            "rotational_damping_n_m_s_rad": list(
                self.rotational_damping_n_m_s_rad
            ),
            "preload_force_local_n": list(self.preload_force_local_n),
            "free_coordinate": self.free_coordinate,
            "ideal_constraint": False,
            "source": self.source,
        }


class MechanismAdapter(ABC):
    """Trusted execution plugin for one mechanism-family semantics.

    ``AdapterSpec`` remains immutable, hash-bound execution data.  This class
    owns behavior: derivation, validator load generation, branch semantics,
    and backend selection.  Keeping the two concepts separate prevents a
    design package or product-data adapter from acquiring verdict authority.
    """

    plugin_id: str
    families: frozenset[str]
    coordinate: GeneralizedCoordinateSpec
    worker_backend: str
    whole_machine_backend: str
    implementation_capability_ids: frozenset[str] = frozenset()
    model_boundary_ids: frozenset[str] = frozenset()
    geometry_proof_kinds: frozenset[str] = frozenset()
    # Evidence keys the initial functional-geometry report must carry as
    # objects before this family may enter physical execution.
    initial_geometry_evidence_keys: frozenset[str] = frozenset()
    #: Trace branch recorded when replaying the certified whole-machine task.
    whole_machine_artifact_branch: str = "reduced"
    #: ``mechanism_kind`` values in compound family specs this family owns.
    #: ``None`` inside the set claims specs without a ``mechanism_kind``.
    family_spec_kinds: frozenset[Any] = frozenset()
    #: Raw family-spec fields tolerated beyond the parsed template fields.
    extra_family_spec_fields: frozenset[str] = frozenset()
    # Physical-trace audit sections that must additionally be authoritative
    # and passing before this family may enter certificate verdicts.
    required_trace_audit_sections: frozenset[str] = frozenset()
    #: Task-input channels actuate every declared input body at once
    #: (e.g. paired hydraulic cylinders) instead of only the port's body.
    task_input_actuation_spans_all_inputs: bool = False
    #: Reviewer-owned compound simulators with multiple task input bodies must
    #: name their explicit per-phase actuation backend.  The shared compound
    #: runtime intentionally owns only one task-wrench target.
    multi_input_task_actuation_backend: str | None = None
    #: Existing Harness wrench-policy profiles used by a project adapter.
    wrench_policy_profiles: frozenset[str] = frozenset()
    #: Plan keys this family's *lane execution path* provably never reads,
    #: excluded from the cross-run lane fingerprint so acceptance-side
    #: edits do not invalidate physically identical lanes.  The default is
    #: empty (everything fingerprinted) because some families do consume
    #: acceptance data inside their lane comparisons — e.g. the bicycle
    #: chain ``ideal_comparison_protocol`` derives thresholds from
    #: ``plan.acceptance_contract``.  Only declare a key here after
    #: auditing every module on the family's lane path.  The original
    #: eleven plugins were audited on 2026-08-12; the written per-family
    #: evidence lives in
    #: docs/final_refactor/evidence/lane_fingerprint_drop_audit.md.
    #: ``plain_revolute_v1`` and ``prismatic_detent_latch_v1`` were
    #: re-audited on 2026-08-26 and keep both typed-contract keys
    #: fingerprinted.  The nail-clipper plugin was admitted later and
    #: declares the same two-key drop as the 2026-08-12 catalog default.
    lane_fingerprint_dropped_keys: tuple[str, ...] = ()

    def supports(self, plan: MechanismPlan, adapter: AdapterSpec) -> bool:
        return plan.family in self.families and adapter.family == plan.family

    def materialize(
        self,
        plan: MechanismPlan,
        adapter: AdapterSpec,
        *,
        workspace_root: str | Path,
        output_dir: str | Path,
    ) -> AdapterSpec:
        return adapter

    def required_geometry_proofs(
        self,
        adapter: AdapterSpec,
        *,
        capsule_id: str,
    ) -> tuple[Any, ...]:
        """Return deterministic STEP proof requirements for physical joints."""

        del adapter, capsule_id
        return ()

    def validate_execution_geometry_proofs(
        self,
        execution: "ExecutionRequest",
    ) -> None:
        requirements = MechanismAdapter.validated_geometry_proof_requirements(
            self,
            execution.adapter,
            capsule_id=execution.plan.capsule_id,
        )

        for requirement in requirements:
            execution.require_geometry_proof(
                requirement.joint_id,
                requirement.proof_kind,
            )

    def validated_geometry_proof_requirements(
        self,
        adapter: AdapterSpec,
        *,
        capsule_id: str,
    ) -> tuple[Any, ...]:
        requirements = self.required_geometry_proofs(
            copy.deepcopy(adapter),
            capsule_id=capsule_id,
        )
        if not isinstance(requirements, tuple):
            raise geometry_proof_contract_error(
                self,
                "MECHANISM_ADAPTER_GEOMETRY_PROOFS_INVALID",
                f"adapter {self.plugin_id!r} required_geometry_proofs() must "
                "return a tuple",
            )
        requirement_kinds = {
            str(requirement.proof_kind) for requirement in requirements
        }
        if self.geometry_proof_kinds and not requirements:
            raise geometry_proof_contract_error(
                self,
                "MECHANISM_ADAPTER_GEOMETRY_PROOFS_EMPTY",
                f"adapter {self.plugin_id!r} declared geometry proof kinds but "
                "returned no requirements",
            )
        if not requirement_kinds <= self.geometry_proof_kinds:
            raise geometry_proof_contract_error(
                self,
                "MECHANISM_ADAPTER_GEOMETRY_PROOF_KIND_UNDECLARED",
                f"adapter {self.plugin_id!r} returned an undeclared geometry "
                "proof kind",
            )
        return requirements

    @abstractmethod
    def load_cases(self, plan: MechanismPlan) -> tuple[LoadCase, ...]:
        raise NotImplementedError

    @abstractmethod
    def expected_reduced_ideal_links(
        self,
        plan: MechanismPlan,
        adapter: AdapterSpec,
    ) -> tuple[str, ...]:
        raise NotImplementedError

    def physical_supports(
        self,
        plan: MechanismPlan,
        adapter: AdapterSpec,
    ) -> tuple[CompliantSupportSpec, ...]:
        return ()

    def constitutive_supports(
        self,
        plan: MechanismPlan,
        adapter: AdapterSpec,
    ) -> tuple[CompliantSupportSpec, ...]:
        """Real component forces that exist in physical and reduced branches."""

        return ()

    def validate_family_spec(self, adapter: AdapterSpec) -> None:
        """Parse the hash-bound family execution data; raise when invalid.

        The generic worker calls this before building an execution
        request instead of dispatching on backend names.
        """

        return None

    def simulate(
        self,
        execution: "ExecutionRequest",
        workspace: Path,
    ) -> dict[str, Any]:
        """Run this family's trusted physical/reduced branch simulation."""

        raise MechanismValidationError(
            "MECHANISM_WORKER_BACKEND_UNAVAILABLE",
            f"worker backend {self.worker_backend!r} is not implemented",
        )

    def wrench_body_roles(
        self,
        family_spec: Mapping[str, Any],
    ) -> tuple[frozenset[str], frozenset[str]]:
        """Declared (input, output) body IDs for wrench-ledger accounting."""

        inputs = {str(family_spec.get("input_body_id") or "")}
        outputs = {str(family_spec.get("output_body_id") or "")}
        return (
            frozenset(item for item in inputs if item),
            frozenset(item for item in outputs if item),
        )

    def load_envelope(
        self,
        load_cases: tuple["LoadCase", ...],
        comparisons: tuple[dict[str, Any], ...],
    ) -> dict[str, Any] | None:
        """Family-owned certificate load envelope; ``None`` keeps the generic one."""

        return None

    def certificate_model(
        self,
        adapter: AdapterSpec,
    ) -> dict[str, Any] | None:
        """Family-owned compound certificate model; ``None`` keeps the generic one."""

        return None

    def replacement_topology(
        self,
        adapter: AdapterSpec,
    ) -> dict[str, Any] | None:
        """Family-owned replacement topology; ``None`` keeps the generic one."""

        return None

    def build_whole_machine_execution_plan(
        self,
        *,
        plan: MechanismPlan,
        workspace: Path,
        output_dir: Path,
        certificate_path: Path,
        certificate_sha256: str,
        replacement_path: Path,
        replacement_sha256: str,
        certificate: Mapping[str, Any],
        replacement_model: Mapping[str, Any],
        adapter: AdapterSpec,
    ) -> Any | None:
        """Build the family's whole-machine execution plan.

        ``None`` means this family has no whole-machine backend; the
        generic dispatcher fails closed with
        ``WHOLE_MACHINE_BACKEND_UNAVAILABLE``.
        """

        return None

    def load_whole_machine_plan(self, value: Mapping[str, Any]) -> Any | None:
        """Parse a whole-machine plan document this family claims.

        Families claim their own ``coordinate_kind`` vocabulary and return
        the typed plan; ``None`` lets other families claim the document.
        """

        return None

    def parse_family_spec(self, raw: Mapping[str, Any]) -> Any:
        """Parse a raw family spec this family claimed via family_spec_kinds."""

        raise MechanismValidationError(
            "MECHANISM_KIND_UNSUPPORTED",
            f"plugin {self.plugin_id!r} does not parse family specs",
        )

    def ideal_model(
        self,
        plan: MechanismPlan,
        family_spec: Mapping[str, Any] | None,
    ) -> Any | None:
        """Return this family's deterministic ideal evaluator, if it claims
        the plan; ``None`` lets other families claim it."""

        return None

    def ideal_comparison_protocol(
        self,
        plan: MechanismPlan,
        family_spec: Mapping[str, Any] | None,
    ) -> tuple[tuple[str, ...], float, str, str] | None:
        """Saved-pose comparison coordinates: (fields, threshold, unit, source).

        ``None`` lets other families claim the plan; the generic comparator
        fails closed when nobody claims it.
        """

        return None

    def physical_behavior(
        self,
        plan: MechanismPlan,
        load_case: "LoadCase",
        trace: dict[str, Any],
        family_spec: Mapping[str, Any] | None,
    ) -> tuple[dict[str, Any], dict[str, Any], list[str]] | None:
        """Family-owned physical trace evaluation: (metrics, thresholds,
        failures); ``None`` lets other families claim the plan."""

        return None

    def task_channel_layout(
        self,
        plan: MechanismPlan,
        family_spec: Mapping[str, Any],
        body_ids: tuple[str, ...],
    ) -> tuple[
        set[str],
        set[str],
        dict[str, str],
        dict[str, tuple[float, float, float]],
    ] | None:
        """Exact runtime task-channel layout: (inputs, outputs, fields, axes).

        ``None`` means this family declares no exact layout; the task
        channel registry fails closed.
        """

        return None

    def simulate_whole_machine(
        self,
        plan: Any,
        *,
        workspace_root: Path,
    ) -> dict[str, Any] | None:
        """Run the family's whole-machine task simulation in the worker.

        ``None`` means unavailable; the worker fails closed.
        """

        return None


class PlainRevoluteAdapter(MechanismAdapter):
    plugin_id = "plain_revolute_v1"
    families = frozenset({"revolute"})
    # Audited 2026-08-26: the 2026-08-12 drop of acceptance_contract and
    # semantic_binding is withdrawn.  The three typed-contract gaps that
    # justified the drop are now closed — typed task_channel_layout on
    # this family (moving_group channel), strict tongs task-program
    # evidence, and tongs suite metrics in evaluate_task_acceptance —
    # so certificates consume the typed contract.  Both keys stay in
    # the lane fingerprint.  The 2026-08-12 catalog audit in
    # docs/final_refactor/evidence/lane_fingerprint_drop_audit.md is
    # superseded for this family.
    lane_fingerprint_dropped_keys = ()
    coordinate = GeneralizedCoordinateSpec(
        position_field="generalized_angle_rad",
        velocity_field="generalized_speed_rad_s",
        position_unit="rad",
        velocity_unit="rad/s",
        input_wrench_type="torque",
        input_unit="N*m",
    )
    worker_backend = "plain_revolute_v1"
    whole_machine_backend = "plain_revolute_v1"

    def materialize(
        self,
        plan: MechanismPlan,
        adapter: AdapterSpec,
        *,
        workspace_root: str | Path,
        output_dir: str | Path,
    ) -> AdapterSpec:
        if (
            adapter.adapter_id == DERIVED_PLAIN_REVOLUTE_ADAPTER_ID
            and adapter.geometry is None
        ):
            from .derived_adapter import materialize_plain_revolute_adapter

            return materialize_plain_revolute_adapter(
                plan,
                workspace_root=workspace_root,
                output_dir=output_dir,
                family_spec=adapter.family_spec,
            )
        return adapter

    def load_cases(self, plan: MechanismPlan) -> tuple[LoadCase, ...]:
        from .loads import default_revolute_load_cases

        return default_revolute_load_cases(plan)

    def expected_reduced_ideal_links(
        self,
        plan: MechanismPlan,
        adapter: AdapterSpec,
    ) -> tuple[str, ...]:
        return ("ChLinkLockRevolute",)

    def simulate(
        self,
        execution: "ExecutionRequest",
        workspace: Path,
    ) -> dict[str, Any]:
        from .worker import simulate_plain_revolute

        return simulate_plain_revolute(execution, workspace)

    def wrench_body_roles(
        self,
        family_spec: Mapping[str, Any],
    ) -> tuple[frozenset[str], frozenset[str]]:
        return (frozenset({"moving_group"}), frozenset({"moving_group"}))

    def physical_supports(
        self,
        plan: MechanismPlan,
        adapter: AdapterSpec,
    ) -> tuple[CompliantSupportSpec, ...]:
        # The physical branch still resolves the authored pin, barrel, thrust
        # faces, clearance, friction, and hard stops as contact. This finite
        # bearing support represents housing/pin elasticity and damping; its
        # local Z coordinate has zero stiffness, so it cannot impose the
        # claimed hinge angle as an ideal revolute constraint. Finite local-Z
        # damping represents hinge friction and dissipates impact energy
        # without prescribing a position.
        off_axis_stiffness = 10.0
        off_axis_damping = 0.1
        source = "provisional_small_machine_plain_bearing_default_v2"
        if (
            adapter.family_spec is not None
            and adapter.family_spec.get("mechanism_kind") == "axis_spring"
        ):
            from .axis_spring_spec import axis_spring_from_dict

            spring = axis_spring_from_dict(
                adapter.family_spec,
                expected_family="revolute",
            )
            off_axis_stiffness = spring.bearing_off_axis_stiffness_si
            off_axis_damping = spring.bearing_off_axis_damping_si
            source = "hash_bound_axis_spring_bearing_v1"
        return (
            CompliantSupportSpec(
                support_id="plain_revolute_bearing_v1",
                chrono_type="ExplicitChLoadBodyForceTorque",
                translational_stiffness_n_m=(110_000.0, 110_000.0, 110_000.0),
                translational_damping_n_s_m=(105.0, 105.0, 105.0),
                rotational_stiffness_n_m_rad=(
                    off_axis_stiffness, off_axis_stiffness, 0.0
                ),
                rotational_damping_n_m_s_rad=(
                    off_axis_damping, off_axis_damping, 0.35
                ),
                free_coordinate="joint_frame_rz",
                source=source,
            ),
        )

    def constitutive_supports(
        self,
        plan: MechanismPlan,
        adapter: AdapterSpec,
    ) -> tuple[CompliantSupportSpec, ...]:
        if adapter.family_spec is None:
            return ()
        from .axis_spring_spec import axis_spring_from_dict

        spring = axis_spring_from_dict(
            adapter.family_spec,
            expected_family="revolute",
        )
        return (
            CompliantSupportSpec(
                support_id="axis_torsion_spring_v1",
                chrono_type="ExplicitChLoadBodyForceTorque",
                translational_stiffness_n_m=(0.0, 0.0, 0.0),
                translational_damping_n_s_m=(0.0, 0.0, 0.0),
                rotational_stiffness_n_m_rad=(
                    0.0,
                    0.0,
                    spring.stiffness_si,
                ),
                rotational_damping_n_m_s_rad=(
                    0.0,
                    0.0,
                    spring.damping_si,
                ),
                free_coordinate="none",
                source="hash_bound_axis_spring_spec_v1",
            ),
        )


    def build_whole_machine_execution_plan(
        self,
        *,
        plan: MechanismPlan,
        workspace: Path,
        output_dir: Path,
        certificate_path: Path,
        certificate_sha256: str,
        replacement_path: Path,
        replacement_sha256: str,
        certificate: Mapping[str, Any],
        replacement_model: Mapping[str, Any],
        adapter: AdapterSpec,
    ) -> Any:
        from .whole_machine import build_plain_execution_plan

        return build_plain_execution_plan(
            plan=plan,
            workspace=workspace,
            certificate_path=certificate_path,
            certificate_sha256=certificate_sha256,
            replacement_path=replacement_path,
            replacement_sha256=replacement_sha256,
            certificate=certificate,
            replacement_model=replacement_model,
            adapter=adapter,
            plugin=self,
            kind='revolute',
        )

    def load_whole_machine_plan(self, value: Mapping[str, Any]) -> Any | None:
        if value.get("coordinate_kind") != 'rotation':
            return None
        from .whole_machine import WholeMachineRevoluteExecutionPlan

        return WholeMachineRevoluteExecutionPlan.from_dict(dict(value))

    def simulate_whole_machine(
        self,
        plan: Any,
        *,
        workspace_root: Path,
    ) -> dict[str, Any]:
        from .whole_machine_worker import (
            run_plain_whole_machine,
            simulate_plain_revolute_stages,
        )

        return run_plain_whole_machine(
            plan, lambda: simulate_plain_revolute_stages(plan)
        )


    def ideal_model(
        self,
        plan: MechanismPlan,
        family_spec: Mapping[str, Any] | None,
    ) -> Any | None:
        if plan.family not in {"revolute", "continuous_revolute"}:
            return None
        from .ideal_reference import PlainCoordinateModel

        return PlainCoordinateModel(
            model_id="plain_revolute_coordinate_v1",
            position_field="generalized_angle_rad",
            velocity_field="generalized_speed_rad_s",
            unit="rad",
        )


    def ideal_comparison_protocol(
        self,
        plan: MechanismPlan,
        family_spec: Mapping[str, Any] | None,
    ) -> tuple[tuple[str, ...], float, str, str] | None:
        if plan.family not in {"revolute", "continuous_revolute"}:
            return None
        return (
            ("generalized_angle_rad",),
            math.radians(3.0),
            "rad",
            "plain_revolute_protocol",
        )

    def physical_behavior(
        self,
        plan: MechanismPlan,
        load_case: "LoadCase",
        trace: dict[str, Any],
        family_spec: Mapping[str, Any] | None,
    ) -> tuple[dict[str, Any], dict[str, Any], list[str]] | None:
        if plan.family not in {"revolute", "continuous_revolute"}:
            return None
        from .comparison import (
            RevoluteKinematicThresholds,
            _physical_revolute_scope_evaluation,
            _revolute_behavior_failures,
            _revolute_metrics,
            _validate_revolute_trace,
        )

        thresholds = RevoluteKinematicThresholds()
        samples = _validate_revolute_trace(
            trace,
            branch="physical",
            plan=plan,
            load_case=load_case,
        )
        metrics = _revolute_metrics(samples, load_case, plan.axis_abs)
        failures = _revolute_behavior_failures(
            metrics,
            branch="physical",
            load_case=load_case,
            thresholds=thresholds,
        )
        scope_failures, _ = _physical_revolute_scope_evaluation(
            plan,
            metrics,
            load_case,
        )
        return metrics, thresholds.to_dict(), [*failures, *scope_failures]


    def task_channel_layout(
        self,
        plan: MechanismPlan,
        family_spec: Mapping[str, Any],
        body_ids: tuple[str, ...],
    ) -> tuple[
        set[str],
        set[str],
        dict[str, str],
        dict[str, tuple[float, float, float]],
    ] | None:
        # Single-hinge machines declare no family_spec.bodies; inventory
        # aliases every moving group onto this channel, matching the
        # no-bodies fallback. Axis is plan.axis_abs (family_spec has none).
        input_ids: set[str] = set()
        output_ids: set[str] = set()
        fields: dict[str, str] = {}
        axes: dict[str, tuple[float, float, float]] = {}
        body_id = "moving_group"
        input_ids.add(body_id)
        output_ids.add(body_id)
        fields[body_id] = self.coordinate.position_field
        axis = plan.axis_abs
        if isinstance(axis, (list, tuple)) and len(axis) == 3:
            axes[body_id] = tuple(float(item) for item in axis)
        return input_ids, output_ids, fields, axes


class RackPinionAdapter(MechanismAdapter):
    """Mixed revolute/prismatic mechanism coupled by a rack-pinion mesh."""

    plugin_id = "rack_pinion_v1"
    families = frozenset({"compound"})
    coordinate = GeneralizedCoordinateSpec(
        position_field="rack_progress_m",
        velocity_field="rack_speed_m_s",
        position_unit="m",
        velocity_unit="m/s",
        input_wrench_type="torque",
        input_unit="N*m",
        component_count=2,
    )
    worker_backend = "rack_pinion_v1"
    whole_machine_backend = "rack_pinion_v1"
    # Audited 2026-08-12: the rack-pinion lane path (worker request
    # loading, rack_pinion_runtime_v10, rack_pinion_loads_v3 via dispatch,
    # ideal_reference RackPinionModel, ideal_comparison,
    # physical_ideal_comparison, comparison) reads only ``family_spec``,
    # capsule geometry fields, and the load case.  Acceptance evaluation
    # consumes these sections in finalize, which always re-runs.
    lane_fingerprint_dropped_keys = (
        "acceptance_contract",
        "semantic_binding",
    )

    def supports(self, plan: MechanismPlan, adapter: AdapterSpec) -> bool:
        return (
            super().supports(plan, adapter)
            and isinstance(adapter.family_spec, dict)
            and adapter.family_spec.get("mechanism_kind") == "rack_pinion"
        )

    def materialize(
        self,
        plan: MechanismPlan,
        adapter: AdapterSpec,
        *,
        workspace_root: str | Path,
        output_dir: str | Path,
    ) -> AdapterSpec:
        from .family_collision import materialize_family_collision_adapter
        from .rack_pinion_spec import rack_pinion_from_dict

        spec = rack_pinion_from_dict(adapter.family_spec)
        return materialize_family_collision_adapter(
            plan,
            adapter,
            spec.bodies,
            workspace_root=workspace_root,
            output_dir=output_dir,
        )

    def load_cases(self, plan: MechanismPlan) -> tuple[LoadCase, ...]:
        from .rack_pinion_loads_v3 import default_rack_pinion_load_cases

        return default_rack_pinion_load_cases(plan)

    def expected_reduced_ideal_links(
        self,
        plan: MechanismPlan,
        adapter: AdapterSpec,
    ) -> tuple[str, ...]:
        from .rack_pinion_spec import rack_pinion_from_dict

        rack_pinion_from_dict(adapter.family_spec)
        return (
            "ChLinkLockRevolute",
            "ChLinkLockPrismatic",
            "ChLinkMateRackPinion",
        )

    family_spec_kinds = frozenset({'rack_pinion'})

    def parse_family_spec(self, raw: Mapping[str, Any]) -> Any:
        from .rack_pinion_spec import rack_pinion_from_dict

        return rack_pinion_from_dict(raw)

    def validate_family_spec(self, adapter: AdapterSpec) -> None:
        self.parse_family_spec(adapter.family_spec)

    def simulate(
        self,
        execution: "ExecutionRequest",
        workspace: Path,
    ) -> dict[str, Any]:
        from .rack_pinion_runtime_v10 import simulate_rack_pinion

        return simulate_rack_pinion(execution, workspace)

    def wrench_body_roles(
        self,
        family_spec: Mapping[str, Any],
    ) -> tuple[frozenset[str], frozenset[str]]:
        inputs = {str(family_spec.get("pinion_body_id") or "")}
        outputs = {str(family_spec.get("rack_body_id") or "")}
        return (
            frozenset(item for item in inputs if item),
            frozenset(item for item in outputs if item),
        )

    def load_envelope(
        self,
        load_cases: tuple["LoadCase", ...],
        comparisons: tuple[dict[str, Any], ...],
    ) -> dict[str, Any] | None:
        from .rack_pinion_certificate import rack_pinion_load_envelope

        return rack_pinion_load_envelope(load_cases, comparisons)

    def certificate_model(
        self,
        adapter: AdapterSpec,
    ) -> dict[str, Any] | None:
        from .rack_pinion_certificate import rack_pinion_certificate_model

        return rack_pinion_certificate_model(adapter)

    def replacement_topology(
        self,
        adapter: AdapterSpec,
    ) -> dict[str, Any] | None:
        from .rack_pinion_certificate import rack_pinion_replacement_topology

        return rack_pinion_replacement_topology(adapter)


    def build_whole_machine_execution_plan(
        self,
        *,
        plan: MechanismPlan,
        workspace: Path,
        output_dir: Path,
        certificate_path: Path,
        certificate_sha256: str,
        replacement_path: Path,
        replacement_sha256: str,
        certificate: Mapping[str, Any],
        replacement_model: Mapping[str, Any],
        adapter: AdapterSpec,
    ) -> Any:
        from .rack_pinion_whole_machine_v4 import build_rack_pinion_execution_plan

        return build_rack_pinion_execution_plan(
            plan=plan,
            workspace=workspace,
            certificate_path=certificate_path,
            certificate_sha256=certificate_sha256,
            replacement_path=replacement_path,
            replacement_sha256=replacement_sha256,
            certificate=certificate,
            replacement_model=replacement_model,
            adapter=adapter,
            plugin=self,
        )

    def load_whole_machine_plan(self, value: Mapping[str, Any]) -> Any | None:
        if value.get("coordinate_kind") != 'rack_pinion_progress':
            return None
        from .rack_pinion_whole_machine_v4 import WholeMachineRackPinionExecutionPlan

        return WholeMachineRackPinionExecutionPlan.from_dict(dict(value))

    def simulate_whole_machine(
        self,
        plan: Any,
        *,
        workspace_root: Path,
    ) -> dict[str, Any]:
        from .rack_pinion_whole_machine_v4 import (
            simulate_whole_machine_rack_pinion,
        )
        from .whole_machine_worker import run_family_single_case

        return run_family_single_case(
            plan, lambda: simulate_whole_machine_rack_pinion(plan)
        )


    def ideal_model(
        self,
        plan: MechanismPlan,
        family_spec: Mapping[str, Any] | None,
    ) -> Any | None:
        if (
            not isinstance(family_spec, Mapping)
            or family_spec.get("mechanism_kind") not in self.family_spec_kinds
        ):
            return None
        from .ideal_reference import RackPinionModel

        return RackPinionModel()


    def ideal_comparison_protocol(
        self,
        plan: MechanismPlan,
        family_spec: Mapping[str, Any] | None,
    ) -> tuple[tuple[str, ...], float, str, str] | None:
        if (
            not isinstance(family_spec, Mapping)
            or family_spec.get("mechanism_kind") not in self.family_spec_kinds
        ):
            return None
        from .rack_pinion_spec import rack_pinion_from_dict

        spec = rack_pinion_from_dict(family_spec)
        return (
            ("rack_progress_m",),
            spec.output_progress_tolerance_m,
            "m",
            "trusted_rack_pinion_family_spec",
        )

    def physical_behavior(
        self,
        plan: MechanismPlan,
        load_case: "LoadCase",
        trace: dict[str, Any],
        family_spec: Mapping[str, Any] | None,
    ) -> tuple[dict[str, Any], dict[str, Any], list[str]] | None:
        if (
            not isinstance(family_spec, Mapping)
            or family_spec.get("mechanism_kind") not in self.family_spec_kinds
        ):
            return None
        from .rack_pinion_comparison_v3 import (
            RackPinionThresholds,
            _adapter_spec,
            _behavior_failures,
            _metrics,
            _validate_trace,
        )

        spec = _adapter_spec(plan)
        thresholds = RackPinionThresholds.for_spec(spec)
        samples = _validate_trace(
            trace,
            branch="physical",
            plan=plan,
            load_case=load_case,
        )
        metrics = _metrics(samples, load_case, spec.pinion_axis_abs)
        failures = _behavior_failures(
            metrics,
            branch="physical",
            load_case=load_case,
            thresholds=thresholds,
        )
        return metrics, thresholds.to_dict(), failures


    def task_channel_layout(
        self,
        plan: MechanismPlan,
        family_spec: Mapping[str, Any],
        body_ids: tuple[str, ...],
    ) -> tuple[
        set[str],
        set[str],
        dict[str, str],
        dict[str, tuple[float, float, float]],
    ] | None:
        input_ids: set[str] = set()
        output_ids: set[str] = set()
        fields: dict[str, str] = {}
        axes: dict[str, tuple[float, float, float]] = {}
        input_id = str(family_spec.get("pinion_body_id") or "")
        output_id = str(family_spec.get("rack_body_id") or "")
        input_ids.add(input_id)
        output_ids.add(output_id)
        fields[input_id] = "pinion_angle_rad"
        fields[output_id] = "rack_progress_m"
        # The trusted load program raises the gate with a crank torque along
        # -pinion_axis_abs (see rack_pinion_loads_v3.outbound_axis), so that is
        # the exact actuation axis capsule input ports must declare.
        pinion_axis = tuple(family_spec.get("pinion_axis_abs") or ())
        axes[input_id] = tuple(-float(item) for item in pinion_axis)
        axes[output_id] = tuple(family_spec.get("rack_axis_abs") or ())
        return input_ids, output_ids, fields, axes


class SpurGearClockAdapter(MechanismAdapter):
    """Multi-shaft spur-gear train with a cam-driven alarm follower."""

    plugin_id = "spur_gear_clock_v1"
    families = frozenset({"compound"})
    # Audited 2026-08-12 (catalog-wide consumption audit; per-family evidence
    # in docs/final_refactor/evidence/lane_fingerprint_drop_audit.md): the
    # clock lane path (clock_gear_runtime/loads/comparison, ideal_reference,
    # ideal_comparison) never reads ``plan.acceptance_contract`` or
    # ``plan.semantic_binding``.
    lane_fingerprint_dropped_keys = (
        "acceptance_contract",
        "semantic_binding",
    )
    coordinate = GeneralizedCoordinateSpec(
        position_field="clock_output_angle_rad",
        velocity_field="clock_output_speed_rad_s",
        position_unit="rad",
        velocity_unit="rad/s",
        input_wrench_type="torque",
        input_unit="N*m",
        component_count=3,
    )
    worker_backend = "spur_gear_clock_v1"
    whole_machine_backend = "spur_gear_clock_v1"
    implementation_capability_ids = frozenset(
        {
            "clock.finite_gear_train",
            "clock.independent_alarm_power_paths",
        }
    )

    def supports(self, plan: MechanismPlan, adapter: AdapterSpec) -> bool:
        return (
            super().supports(plan, adapter)
            and isinstance(adapter.family_spec, dict)
            and adapter.family_spec.get("mechanism_kind") == "spur_gear_clock"
        )

    def materialize(
        self,
        plan: MechanismPlan,
        adapter: AdapterSpec,
        *,
        workspace_root: str | Path,
        output_dir: str | Path,
    ) -> AdapterSpec:
        from .clock_gear_train_spec import clock_gear_train_from_dict
        from .family_collision import materialize_family_collision_adapter

        spec = clock_gear_train_from_dict(adapter.family_spec)
        return materialize_family_collision_adapter(
            plan,
            adapter,
            spec.bodies,
            workspace_root=workspace_root,
            output_dir=output_dir,
        )

    def load_cases(self, plan: MechanismPlan) -> tuple[LoadCase, ...]:
        from .clock_gear_loads import default_clock_gear_load_cases

        return default_clock_gear_load_cases(plan)

    family_spec_kinds = frozenset({'spur_gear_clock'})

    def parse_family_spec(self, raw: Mapping[str, Any]) -> Any:
        from .clock_gear_train_spec import clock_gear_train_from_dict

        return clock_gear_train_from_dict(raw)

    def validate_family_spec(self, adapter: AdapterSpec) -> None:
        self.parse_family_spec(adapter.family_spec)

    def simulate(
        self,
        execution: "ExecutionRequest",
        workspace: Path,
    ) -> dict[str, Any]:
        from .clock_gear_runtime import simulate_clock_gear_train

        return simulate_clock_gear_train(execution, workspace)

    def load_envelope(
        self,
        load_cases: tuple["LoadCase", ...],
        comparisons: tuple[dict[str, Any], ...],
    ) -> dict[str, Any] | None:
        from .clock_gear_certificate import clock_gear_load_envelope

        return clock_gear_load_envelope(load_cases, comparisons)

    def certificate_model(
        self,
        adapter: AdapterSpec,
    ) -> dict[str, Any] | None:
        from .clock_gear_certificate import clock_gear_certificate_model

        return clock_gear_certificate_model(adapter)

    def replacement_topology(
        self,
        adapter: AdapterSpec,
    ) -> dict[str, Any] | None:
        from .clock_gear_certificate import clock_gear_replacement_topology

        return clock_gear_replacement_topology(adapter)

    def wrench_body_roles(
        self,
        family_spec: Mapping[str, Any],
    ) -> tuple[frozenset[str], frozenset[str]]:
        inputs, outputs = super().wrench_body_roles(family_spec)
        extra = {
            str(family_spec.get(key) or "")
            for key in ("setting_input_body_id", "alarm_input_body_id")
        }
        return (
            frozenset(inputs | {item for item in extra if item}),
            outputs,
        )

    def expected_reduced_ideal_links(
        self,
        plan: MechanismPlan,
        adapter: AdapterSpec,
    ) -> tuple[str, ...]:
        from .clock_gear_train_spec import clock_gear_train_from_dict

        spec = clock_gear_train_from_dict(adapter.family_spec)
        return (
            *("ChLinkLockRevolute" for _ in spec.shafts),
            *("ChLinkLockGear" for _ in spec.gear_meshes),
            *("AnalyticalCamFollower" for _ in spec.cam_followers),
            *("AnalyticalShaftTransfer" for _ in spec.shaft_transfers),
            *(("AnalyticalSettingClutch",) if spec.setting_clutch is not None else ()),
            *(
                ("AnalyticalAlarmSettingClutch",)
                if spec.alarm_setting_clutch is not None
                else ()
            ),
        )


# [kimi-scissor-lift] hydraulic scissor-lift family plugin.
    def build_whole_machine_execution_plan(
        self,
        *,
        plan: MechanismPlan,
        workspace: Path,
        output_dir: Path,
        certificate_path: Path,
        certificate_sha256: str,
        replacement_path: Path,
        replacement_sha256: str,
        certificate: Mapping[str, Any],
        replacement_model: Mapping[str, Any],
        adapter: AdapterSpec,
    ) -> Any:
        from .clock_gear_whole_machine import build_clock_execution_plan

        return build_clock_execution_plan(
            plan=plan,
            workspace=workspace,
            certificate_path=certificate_path,
            certificate_sha256=certificate_sha256,
            replacement_path=replacement_path,
            replacement_sha256=replacement_sha256,
            certificate=certificate,
            replacement_model=replacement_model,
            adapter=adapter,
            plugin=self,
        )

    def load_whole_machine_plan(self, value: Mapping[str, Any]) -> Any | None:
        if value.get("coordinate_kind") != 'clock_multifunction':
            return None
        from .whole_machine import WholeMachineClockExecutionPlan

        return WholeMachineClockExecutionPlan.from_dict(dict(value))

    def simulate_whole_machine(
        self,
        plan: Any,
        *,
        workspace_root: Path,
    ) -> dict[str, Any]:
        from .whole_machine_worker import run_clock_whole_machine

        return run_clock_whole_machine(plan)


    def ideal_model(
        self,
        plan: MechanismPlan,
        family_spec: Mapping[str, Any] | None,
    ) -> Any | None:
        if (
            not isinstance(family_spec, Mapping)
            or family_spec.get("mechanism_kind") not in self.family_spec_kinds
        ):
            return None
        from .ideal_reference import ClockGearModel

        return ClockGearModel()


    def ideal_comparison_protocol(
        self,
        plan: MechanismPlan,
        family_spec: Mapping[str, Any] | None,
    ) -> tuple[tuple[str, ...], float, str, str] | None:
        if (
            not isinstance(family_spec, Mapping)
            or family_spec.get("mechanism_kind") not in self.family_spec_kinds
        ):
            return None
        from .clock_gear_train_spec import clock_gear_train_from_dict

        spec = clock_gear_train_from_dict(family_spec)
        return (
            (
                "clock_output_angle_rad",
                "hour_hand_angle_rad",
                "second_hand_angle_rad",
                "alarm_angle_rad",
            ),
            spec.output_angle_tolerance_rad,
            "rad",
            "trusted_clock_gear_family_spec",
        )

    def physical_behavior(
        self,
        plan: MechanismPlan,
        load_case: "LoadCase",
        trace: dict[str, Any],
        family_spec: Mapping[str, Any] | None,
    ) -> tuple[dict[str, Any], dict[str, Any], list[str]] | None:
        if (
            not isinstance(family_spec, Mapping)
            or family_spec.get("mechanism_kind") not in self.family_spec_kinds
        ):
            return None
        from .clock_gear_comparison import (
            _behavior_failures,
            _metrics,
            _validate_trace,
        )
        from .clock_gear_train_spec import clock_gear_train_from_dict

        spec = clock_gear_train_from_dict(family_spec)
        samples = _validate_trace(
            trace,
            branch="physical",
            plan=plan,
            load_case=load_case,
            spec=spec,
        )
        metrics = _metrics(samples, load_case)
        failures = _behavior_failures(
            metrics,
            branch="physical",
            load_case=load_case,
            spec=spec,
        )
        return metrics, {
            "source": "trusted_clock_gear_family_spec",
            "output_angle_tolerance_rad": spec.output_angle_tolerance_rad,
            "max_support_drift_m": spec.max_support_drift_m,
            "max_off_axis_rotation_rad": spec.max_off_axis_rotation_rad,
            "max_mesh_error_rad": spec.max_mesh_error_rad,
            "minimum_alarm_swing_rad": spec.minimum_alarm_swing_rad,
        }, failures


    def task_channel_layout(
        self,
        plan: MechanismPlan,
        family_spec: Mapping[str, Any],
        body_ids: tuple[str, ...],
    ) -> tuple[
        set[str],
        set[str],
        dict[str, str],
        dict[str, tuple[float, float, float]],
    ] | None:
        input_ids: set[str] = set()
        output_ids: set[str] = set()
        fields: dict[str, str] = {}
        axes: dict[str, tuple[float, float, float]] = {}
        scheduled_alarm = family_spec.get("scheduled_alarm")
        alarm_target_body_id = (
            str(scheduled_alarm.get("setting_body_id") or "")
            if isinstance(scheduled_alarm, Mapping)
            else ""
        )
        input_ids.update(
            str(family_spec.get(key) or "")
            for key in (
                "input_body_id",
                "setting_input_body_id",
                "alarm_input_body_id",
                "hour_body_id",
            )
        )
        output_ids.update(
            str(family_spec.get(key) or "")
            for key in (
                "output_body_id",
                "hour_body_id",
                "second_body_id",
                "alarm_body_id",
            )
        )
        output_ids.add(alarm_target_body_id)
        for body_id in body_ids:
            fields[body_id] = f"{body_id}_angle_rad"
        fields[str(family_spec.get("input_body_id") or "")] = (
            "barrel_input_angle_rad"
        )
        fields[str(family_spec.get("setting_input_body_id") or "")] = (
            "setting_input_angle_rad"
        )
        fields[str(family_spec.get("alarm_input_body_id") or "")] = (
            "alarm_input_angle_rad"
        )
        fields[str(family_spec.get("output_body_id") or "")] = (
            "clock_output_angle_rad"
        )
        fields[str(family_spec.get("hour_body_id") or "")] = (
            "hour_hand_angle_rad"
        )
        fields[str(family_spec.get("second_body_id") or "")] = (
            "second_hand_angle_rad"
        )
        fields[str(family_spec.get("alarm_body_id") or "")] = (
            "alarm_angle_rad"
        )
        fields[alarm_target_body_id] = "alarm_setting_target_angle_rad"
        fields.pop("", None)
        for raw in family_spec.get("shafts", []):
            if isinstance(raw, Mapping):
                body_id = str(raw.get("body_id") or "")
                axis = raw.get("axis_abs")
                if (
                    body_id
                    and isinstance(axis, (list, tuple))
                    and len(axis) == 3
                ):
                    axes[body_id] = tuple(float(item) for item in axis)
        return input_ids, output_ids, fields, axes


class ScissorLiftAdapter(MechanismAdapter):
    """Single-tier hydraulic scissor lift driven by a cylinder force pair."""

    plugin_id = "hydraulic_scissor_lift_v1"
    families = frozenset({"compound"})
    # Audited 2026-08-12 (catalog-wide consumption audit; per-family evidence
    # in docs/final_refactor/evidence/lane_fingerprint_drop_audit.md): the
    # scissor-lift lane path (scissor_lift_runtime/loads/comparison,
    # ideal_reference, ideal_comparison) never reads
    # ``plan.acceptance_contract`` or ``plan.semantic_binding``.
    lane_fingerprint_dropped_keys = (
        "acceptance_contract",
        "semantic_binding",
    )
    # The paired cylinders push both scissor arms together, so a single
    # capsule input port actuates every declared input body.
    task_input_actuation_spans_all_inputs = True
    coordinate = GeneralizedCoordinateSpec(
        position_field="platform_height_m",
        velocity_field="platform_speed_m_s",
        position_unit="m",
        velocity_unit="m/s",
        input_wrench_type="force",
        input_unit="N",
    )
    worker_backend = "hydraulic_scissor_lift_v1"
    whole_machine_backend = "hydraulic_scissor_lift_v1"

    def supports(self, plan: MechanismPlan, adapter: AdapterSpec) -> bool:
        return (
            super().supports(plan, adapter)
            and isinstance(adapter.family_spec, dict)
            and adapter.family_spec.get("mechanism_kind") == "hydraulic_scissor_lift"
        )

    def materialize(
        self,
        plan: MechanismPlan,
        adapter: AdapterSpec,
        *,
        workspace_root: str | Path,
        output_dir: str | Path,
    ) -> AdapterSpec:
        from .family_collision import materialize_family_collision_adapter
        from .scissor_lift_spec import scissor_lift_from_dict

        spec = scissor_lift_from_dict(adapter.family_spec)
        return materialize_family_collision_adapter(
            plan,
            adapter,
            spec.bodies,
            workspace_root=workspace_root,
            output_dir=output_dir,
        )

    def load_cases(self, plan: MechanismPlan) -> tuple[LoadCase, ...]:
        from .scissor_lift_loads import default_scissor_lift_load_cases

        return default_scissor_lift_load_cases(plan)

    family_spec_kinds = frozenset({'hydraulic_scissor_lift'})

    def parse_family_spec(self, raw: Mapping[str, Any]) -> Any:
        from .scissor_lift_spec import scissor_lift_from_dict

        return scissor_lift_from_dict(raw)

    def validate_family_spec(self, adapter: AdapterSpec) -> None:
        self.parse_family_spec(adapter.family_spec)

    def simulate(
        self,
        execution: "ExecutionRequest",
        workspace: Path,
    ) -> dict[str, Any]:
        from .scissor_lift_runtime import simulate_scissor_lift

        return simulate_scissor_lift(execution, workspace)

    def load_envelope(
        self,
        load_cases: tuple["LoadCase", ...],
        comparisons: tuple[dict[str, Any], ...],
    ) -> dict[str, Any] | None:
        from .scissor_lift_certificate import scissor_lift_load_envelope

        return scissor_lift_load_envelope(load_cases, comparisons)

    def certificate_model(
        self,
        adapter: AdapterSpec,
    ) -> dict[str, Any] | None:
        from .scissor_lift_certificate import scissor_lift_certificate_model

        return scissor_lift_certificate_model(adapter)

    def replacement_topology(
        self,
        adapter: AdapterSpec,
    ) -> dict[str, Any] | None:
        from .scissor_lift_certificate import (
            scissor_lift_replacement_topology,
        )

        return scissor_lift_replacement_topology(adapter)

    def wrench_body_roles(
        self,
        family_spec: Mapping[str, Any],
    ) -> tuple[frozenset[str], frozenset[str]]:
        inputs, outputs = super().wrench_body_roles(family_spec)
        actuator = family_spec.get("actuator") or {}
        extra: set[str] = set()
        if isinstance(actuator, dict):
            extra = {
                str(actuator.get(key) or "")
                for key in ("barrel_body_id", "rod_body_id")
            }
        return (
            frozenset(inputs | {item for item in extra if item}),
            outputs,
        )

    def expected_reduced_ideal_links(
        self,
        plan: MechanismPlan,
        adapter: AdapterSpec,
    ) -> tuple[str, ...]:
        from .scissor_lift_spec import expected_ideal_links, scissor_lift_from_dict

        spec = scissor_lift_from_dict(adapter.family_spec)
        return expected_ideal_links(spec)


    def build_whole_machine_execution_plan(
        self,
        *,
        plan: MechanismPlan,
        workspace: Path,
        output_dir: Path,
        certificate_path: Path,
        certificate_sha256: str,
        replacement_path: Path,
        replacement_sha256: str,
        certificate: Mapping[str, Any],
        replacement_model: Mapping[str, Any],
        adapter: AdapterSpec,
    ) -> Any:
        from .scissor_lift_whole_machine import build_scissor_lift_execution_plan

        return build_scissor_lift_execution_plan(
            plan=plan,
            workspace=workspace,
            certificate_path=certificate_path,
            certificate_sha256=certificate_sha256,
            replacement_path=replacement_path,
            replacement_sha256=replacement_sha256,
            certificate=certificate,
            replacement_model=replacement_model,
            adapter=adapter,
            plugin=self,
            output_dir=output_dir,
        )

    def load_whole_machine_plan(self, value: Mapping[str, Any]) -> Any | None:
        if value.get("coordinate_kind") != 'scissor_lift_platform_progress':
            return None
        from .scissor_lift_whole_machine import WholeMachineScissorLiftExecutionPlan

        return WholeMachineScissorLiftExecutionPlan.from_dict(dict(value))

    def simulate_whole_machine(
        self,
        plan: Any,
        *,
        workspace_root: Path,
    ) -> dict[str, Any]:
        from .scissor_lift_whole_machine import (
            simulate_whole_machine_scissor_lift,
        )
        from .whole_machine_worker import run_family_single_case

        return run_family_single_case(
            plan, lambda: simulate_whole_machine_scissor_lift(plan)
        )


    def ideal_model(
        self,
        plan: MechanismPlan,
        family_spec: Mapping[str, Any] | None,
    ) -> Any | None:
        if (
            not isinstance(family_spec, Mapping)
            or family_spec.get("mechanism_kind") not in self.family_spec_kinds
        ):
            return None
        from .ideal_reference import ScissorLiftModel

        return ScissorLiftModel()


    def ideal_comparison_protocol(
        self,
        plan: MechanismPlan,
        family_spec: Mapping[str, Any] | None,
    ) -> tuple[tuple[str, ...], float, str, str] | None:
        if (
            not isinstance(family_spec, Mapping)
            or family_spec.get("mechanism_kind") not in self.family_spec_kinds
        ):
            return None
        from .scissor_lift_spec import scissor_lift_from_dict

        spec = scissor_lift_from_dict(family_spec)
        return (
            ("platform_progress_m",),
            spec.output_progress_tolerance_m,
            "m",
            "trusted_scissor_lift_family_spec",
        )

    def physical_behavior(
        self,
        plan: MechanismPlan,
        load_case: "LoadCase",
        trace: dict[str, Any],
        family_spec: Mapping[str, Any] | None,
    ) -> tuple[dict[str, Any], dict[str, Any], list[str]] | None:
        if (
            not isinstance(family_spec, Mapping)
            or family_spec.get("mechanism_kind") not in self.family_spec_kinds
        ):
            return None
        from .scissor_lift_comparison import (
            ScissorLiftThresholds,
            _adapter_spec,
            _behavior_failures,
            _metrics,
            _validate_trace,
        )
        from .scissor_lift_loads import cylinder_home_axis

        spec = _adapter_spec(plan)
        thresholds = ScissorLiftThresholds.for_spec(spec)
        samples = _validate_trace(
            trace,
            branch="physical",
            plan=plan,
            load_case=load_case,
            spec=spec,
        )
        metrics = _metrics(samples, load_case, cylinder_home_axis(spec))
        failures = _behavior_failures(
            metrics,
            branch="physical",
            load_case=load_case,
            thresholds=thresholds,
        )
        return metrics, thresholds.to_dict(), failures


    def task_channel_layout(
        self,
        plan: MechanismPlan,
        family_spec: Mapping[str, Any],
        body_ids: tuple[str, ...],
    ) -> tuple[
        set[str],
        set[str],
        dict[str, str],
        dict[str, tuple[float, float, float]],
    ] | None:
        input_ids: set[str] = set()
        output_ids: set[str] = set()
        fields: dict[str, str] = {}
        axes: dict[str, tuple[float, float, float]] = {}
        actuator = family_spec.get("actuator")
        if isinstance(actuator, Mapping):
            barrel_id = str(actuator.get("barrel_body_id") or "")
            rod_id = str(actuator.get("rod_body_id") or "")
            input_ids.update((barrel_id, rod_id))
            barrel_anchor = actuator.get("barrel_anchor_abs_m")
            rod_anchor = actuator.get("rod_anchor_abs_m")
            if (
                isinstance(barrel_anchor, (list, tuple))
                and isinstance(rod_anchor, (list, tuple))
                and len(barrel_anchor) == 3
                and len(rod_anchor) == 3
            ):
                raw_axis = tuple(
                    float(right) - float(left)
                    for left, right in zip(barrel_anchor, rod_anchor)
                )
                norm = math.sqrt(sum(item * item for item in raw_axis))
                if norm > 1.0e-12:
                    axis = tuple(item / norm for item in raw_axis)
                    axes[barrel_id] = axis
                    axes[rod_id] = axis
        output_id = str(family_spec.get("output_body_id") or "")
        output_ids.add(output_id)
        fields[output_id] = "platform_height_m"
        axes[output_id] = tuple(family_spec.get("output_axis_abs") or ())
        return input_ids, output_ids, fields, axes


class MultiOutputShaftAdapter(MechanismAdapter):
    """One input shaft coupled to two or more independently supported shafts."""

    plugin_id = "multi_output_shaft_coupling_v1"
    families = frozenset({"compound"})
    # Audited 2026-08-12 (catalog-wide consumption audit; per-family evidence
    # in docs/final_refactor/evidence/lane_fingerprint_drop_audit.md): the
    # multi-output-shaft lane path (multi_output_shaft_loads/comparison plus
    # the shared clock_gear_runtime, ideal_reference, ideal_comparison) never
    # reads ``plan.acceptance_contract`` or ``plan.semantic_binding``.
    lane_fingerprint_dropped_keys = (
        "acceptance_contract",
        "semantic_binding",
    )
    coordinate = GeneralizedCoordinateSpec(
        position_field="multi_output_primary_angle_rad",
        velocity_field="multi_output_primary_speed_rad_s",
        position_unit="rad",
        velocity_unit="rad/s",
        input_wrench_type="torque",
        input_unit="N*m",
        component_count=3,
    )
    worker_backend = "multi_output_shaft_coupling_v1"
    whole_machine_backend = "multi_output_shaft_coupling_v1"

    def supports(self, plan: MechanismPlan, adapter: AdapterSpec) -> bool:
        return (
            super().supports(plan, adapter)
            and isinstance(adapter.family_spec, dict)
            and adapter.family_spec.get("mechanism_kind")
            in {
                "crown_bevel_dual_output",
                "multi_output_shaft_coupling",
            }
        )

    def materialize(
        self,
        plan: MechanismPlan,
        adapter: AdapterSpec,
        *,
        workspace_root: str | Path,
        output_dir: str | Path,
    ) -> AdapterSpec:
        from .family_collision import materialize_family_collision_adapter
        from .multi_output_shaft_spec import multi_output_shaft_from_dict

        spec = multi_output_shaft_from_dict(adapter.family_spec)
        return materialize_family_collision_adapter(
            plan,
            adapter,
            spec.bodies,
            workspace_root=workspace_root,
            output_dir=output_dir,
        )

    def load_cases(self, plan: MechanismPlan) -> tuple[LoadCase, ...]:
        from .multi_output_shaft_loads import (
            default_multi_output_shaft_load_cases,
        )

        return default_multi_output_shaft_load_cases(plan)

    family_spec_kinds = frozenset({'crown_bevel_dual_output', 'multi_output_shaft_coupling'})

    def parse_family_spec(self, raw: Mapping[str, Any]) -> Any:
        from .multi_output_shaft_spec import multi_output_shaft_from_dict

        return multi_output_shaft_from_dict(raw)

    def validate_family_spec(self, adapter: AdapterSpec) -> None:
        self.parse_family_spec(adapter.family_spec)

    def simulate(
        self,
        execution: "ExecutionRequest",
        workspace: Path,
    ) -> dict[str, Any]:
        from .clock_gear_runtime import simulate_clock_gear_train

        return simulate_clock_gear_train(execution, workspace)

    def load_envelope(
        self,
        load_cases: tuple["LoadCase", ...],
        comparisons: tuple[dict[str, Any], ...],
    ) -> dict[str, Any] | None:
        from .multi_output_shaft_certificate import (
            multi_output_shaft_load_envelope,
        )

        return multi_output_shaft_load_envelope(load_cases, comparisons)

    def certificate_model(
        self,
        adapter: AdapterSpec,
    ) -> dict[str, Any] | None:
        from .multi_output_shaft_certificate import (
            multi_output_shaft_certificate_model,
        )

        return multi_output_shaft_certificate_model(adapter)

    def replacement_topology(
        self,
        adapter: AdapterSpec,
    ) -> dict[str, Any] | None:
        from .multi_output_shaft_certificate import (
            multi_output_shaft_replacement_topology,
        )

        return multi_output_shaft_replacement_topology(adapter)

    def wrench_body_roles(
        self,
        family_spec: Mapping[str, Any],
    ) -> tuple[frozenset[str], frozenset[str]]:
        inputs, outputs = super().wrench_body_roles(family_spec)
        extra = {
            str(family_spec.get(key) or "")
            for key in ("setting_input_body_id", "alarm_input_body_id")
        }
        return (
            frozenset(inputs | {item for item in extra if item}),
            outputs,
        )

    def expected_reduced_ideal_links(
        self,
        plan: MechanismPlan,
        adapter: AdapterSpec,
    ) -> tuple[str, ...]:
        from .multi_output_shaft_spec import multi_output_shaft_from_dict

        spec = multi_output_shaft_from_dict(adapter.family_spec)
        return (
            *("ChLinkLockRevolute" for _ in spec.shafts),
            *("ChLinkLockGear" for _ in spec.gear_meshes),
        )


    def ideal_model(
        self,
        plan: MechanismPlan,
        family_spec: Mapping[str, Any] | None,
    ) -> Any | None:
        if (
            not isinstance(family_spec, Mapping)
            or family_spec.get("mechanism_kind") not in self.family_spec_kinds
        ):
            return None
        from .ideal_reference import MultiOutputShaftModel

        return MultiOutputShaftModel()


    def ideal_comparison_protocol(
        self,
        plan: MechanismPlan,
        family_spec: Mapping[str, Any] | None,
    ) -> tuple[tuple[str, ...], float, str, str] | None:
        if (
            not isinstance(family_spec, Mapping)
            or family_spec.get("mechanism_kind") not in self.family_spec_kinds
        ):
            return None
        from .multi_output_shaft_spec import multi_output_shaft_from_dict

        spec = multi_output_shaft_from_dict(family_spec)
        return (
            tuple(
                f"{body_id}_angle_rad"
                for body_id in spec.output_body_ids
            ),
            spec.max_mesh_error_rad,
            "rad",
            "trusted_multi_output_shaft_family_spec",
        )

    def physical_behavior(
        self,
        plan: MechanismPlan,
        load_case: "LoadCase",
        trace: dict[str, Any],
        family_spec: Mapping[str, Any] | None,
    ) -> tuple[dict[str, Any], dict[str, Any], list[str]] | None:
        if (
            not isinstance(family_spec, Mapping)
            or family_spec.get("mechanism_kind") not in self.family_spec_kinds
        ):
            return None
        from .multi_output_shaft_comparison import (
            physical_multi_output_behavior,
        )

        return physical_multi_output_behavior(plan, load_case, trace)


    def task_channel_layout(
        self,
        plan: MechanismPlan,
        family_spec: Mapping[str, Any],
        body_ids: tuple[str, ...],
    ) -> tuple[
        set[str],
        set[str],
        dict[str, str],
        dict[str, tuple[float, float, float]],
    ] | None:
        input_ids: set[str] = set()
        output_ids: set[str] = set()
        fields: dict[str, str] = {}
        axes: dict[str, tuple[float, float, float]] = {}
        input_ids.add(str(family_spec.get("input_body_id") or ""))
        raw_outputs = family_spec.get("output_body_ids")
        if isinstance(raw_outputs, list):
            output_ids.update(str(item) for item in raw_outputs)
        for body_id in body_ids:
            fields[body_id] = f"{body_id}_angle_rad"
        for raw in family_spec.get("shafts", []):
            if isinstance(raw, Mapping):
                body_id = str(raw.get("body_id") or "")
                axis = raw.get("axis_abs")
                if (
                    body_id
                    and isinstance(axis, (list, tuple))
                    and len(axis) == 3
                ):
                    axes[body_id] = tuple(float(item) for item in axis)
        return input_ids, output_ids, fields, axes


class PrismaticDetentLatchAdapter(MechanismAdapter):
    """Press-to-release linear carriage with finite physical detent contact."""

    plugin_id = "prismatic_detent_latch_v1"
    families = frozenset({"compound"})
    # Audited 2026-08-26: the 2026-08-12 drop of acceptance_contract and
    # semantic_binding is withdrawn.  The three typed-contract gaps that
    # justified the drop are now closed — typed task_channel_layout
    # (input axis = wrench direction, output axis = blade joint),
    # strict detent trace programs (no duplicate phase-start samples;
    # adaptive termination objects; ``termination_reason`` string kept
    # for the old comparator), and utility-knife suite metrics in
    # evaluate_task_acceptance — so certificates consume the typed
    # contract.  Both keys stay in the lane fingerprint.  The
    # 2026-08-12 catalog audit in
    # docs/final_refactor/evidence/lane_fingerprint_drop_audit.md is
    # superseded for this family.
    lane_fingerprint_dropped_keys = ()
    coordinate = GeneralizedCoordinateSpec(
        position_field="blade_carriage_position_m",
        velocity_field="blade_carriage_speed_m_s",
        position_unit="m",
        velocity_unit="m/s",
        input_wrench_type="force",
        input_unit="N",
        component_count=2,
    )
    worker_backend = "prismatic_detent_latch_v1"
    whole_machine_backend = "prismatic_detent_latch_v1"

    def supports(self, plan: MechanismPlan, adapter: AdapterSpec) -> bool:
        return (
            super().supports(plan, adapter)
            and isinstance(adapter.family_spec, dict)
            and adapter.family_spec.get("mechanism_kind")
            == "prismatic_detent_latch"
        )

    def materialize(
        self,
        plan: MechanismPlan,
        adapter: AdapterSpec,
        *,
        workspace_root: str | Path,
        output_dir: str | Path,
    ) -> AdapterSpec:
        from .detent_slider_spec import prismatic_detent_latch_from_dict
        from .family_collision import materialize_family_collision_adapter

        spec = prismatic_detent_latch_from_dict(adapter.family_spec)
        return materialize_family_collision_adapter(
            plan,
            adapter,
            spec.bodies,
            workspace_root=workspace_root,
            output_dir=output_dir,
        )

    def load_cases(self, plan: MechanismPlan) -> tuple[LoadCase, ...]:
        from .detent_slider_loads import default_detent_slider_load_cases

        return default_detent_slider_load_cases(plan)

    family_spec_kinds = frozenset({'prismatic_detent_latch'})

    def parse_family_spec(self, raw: Mapping[str, Any]) -> Any:
        from .detent_slider_spec import prismatic_detent_latch_from_dict

        return prismatic_detent_latch_from_dict(raw)

    def validate_family_spec(self, adapter: AdapterSpec) -> None:
        self.parse_family_spec(adapter.family_spec)

    def simulate(
        self,
        execution: "ExecutionRequest",
        workspace: Path,
    ) -> dict[str, Any]:
        from .detent_slider_runtime import simulate_detent_slider

        return simulate_detent_slider(execution, workspace)

    def load_envelope(
        self,
        load_cases: tuple["LoadCase", ...],
        comparisons: tuple[dict[str, Any], ...],
    ) -> dict[str, Any] | None:
        from .detent_slider_certificate import detent_slider_load_envelope

        return detent_slider_load_envelope(load_cases, comparisons)

    def certificate_model(
        self,
        adapter: AdapterSpec,
    ) -> dict[str, Any] | None:
        from .detent_slider_certificate import detent_slider_certificate_model

        return detent_slider_certificate_model(adapter)

    def replacement_topology(
        self,
        adapter: AdapterSpec,
    ) -> dict[str, Any] | None:
        from .detent_slider_certificate import (
            detent_slider_replacement_topology,
        )

        return detent_slider_replacement_topology(adapter)

    def expected_reduced_ideal_links(
        self,
        plan: MechanismPlan,
        adapter: AdapterSpec,
    ) -> tuple[str, ...]:
        from .detent_slider_spec import prismatic_detent_latch_from_dict

        prismatic_detent_latch_from_dict(adapter.family_spec)
        return (
            "ChLinkLockPrismatic",
            "ChLinkLockPrismatic",
        )


    def ideal_model(
        self,
        plan: MechanismPlan,
        family_spec: Mapping[str, Any] | None,
    ) -> Any | None:
        if (
            not isinstance(family_spec, Mapping)
            or family_spec.get("mechanism_kind") not in self.family_spec_kinds
        ):
            return None
        from .ideal_reference import DetentSliderModel

        return DetentSliderModel()


    def ideal_comparison_protocol(
        self,
        plan: MechanismPlan,
        family_spec: Mapping[str, Any] | None,
    ) -> tuple[tuple[str, ...], float, str, str] | None:
        if (
            not isinstance(family_spec, Mapping)
            or family_spec.get("mechanism_kind") not in self.family_spec_kinds
        ):
            return None
        from .detent_slider_spec import prismatic_detent_latch_from_dict

        spec = prismatic_detent_latch_from_dict(family_spec)
        return (
            ("blade_carriage_position_m",),
            spec.detent_interface.capture_tolerance_m,
            "m",
            "trusted_prismatic_detent_latch_family_spec",
        )

    def physical_behavior(
        self,
        plan: MechanismPlan,
        load_case: "LoadCase",
        trace: dict[str, Any],
        family_spec: Mapping[str, Any] | None,
    ) -> tuple[dict[str, Any], dict[str, Any], list[str]] | None:
        if (
            not isinstance(family_spec, Mapping)
            or family_spec.get("mechanism_kind") not in self.family_spec_kinds
        ):
            return None
        from .detent_slider_comparison import (
            physical_detent_slider_behavior,
        )

        return physical_detent_slider_behavior(plan, load_case, trace)


    def task_channel_layout(
        self,
        plan: MechanismPlan,
        family_spec: Mapping[str, Any],
        body_ids: tuple[str, ...],
    ) -> tuple[
        set[str],
        set[str],
        dict[str, str],
        dict[str, tuple[float, float, float]],
    ] | None:
        from .detent_slider_spec import declared_input_wrench_direction_abs

        input_ids: set[str] = set()
        output_ids: set[str] = set()
        fields: dict[str, str] = {}
        axes: dict[str, tuple[float, float, float]] = {}
        input_id = str(family_spec.get("input_body_id") or "")
        output_id = str(family_spec.get("output_body_id") or "")
        input_ids.add(input_id)
        output_ids.add(output_id)
        fields[output_id] = "blade_carriage_position_m"
        blade_joint = family_spec.get("blade_prismatic_joint")
        axis = (
            blade_joint.get("axis_abs")
            if isinstance(blade_joint, Mapping)
            else None
        )
        if isinstance(axis, (list, tuple)) and len(axis) == 3:
            # Output motion axis is the blade prismatic joint.  The capsule
            # input port declares the press-and-slide wrench, which is not
            # that joint axis; derive it from the family_spec force split.
            axes[output_id] = tuple(float(item) for item in axis)
        wrench = declared_input_wrench_direction_abs(family_spec)
        if wrench is not None:
            axes[input_id] = wrench
        return input_ids, output_ids, fields, axes


class ClickPenCamLatchAdapter(MechanismAdapter):
    """Push-button, cylindrical indexing cam, and two-state refill latch."""

    plugin_id = "click_pen_cam_latch_v1"
    families = frozenset({"compound"})
    # Audited 2026-08-12 (catalog-wide consumption audit; per-family evidence
    # in docs/final_refactor/evidence/lane_fingerprint_drop_audit.md): the
    # click-pen lane path (click_pen_runtime/loads/comparison,
    # ideal_reference, ideal_comparison) never reads
    # ``plan.acceptance_contract`` or ``plan.semantic_binding``.
    lane_fingerprint_dropped_keys = (
        "acceptance_contract",
        "semantic_binding",
    )
    coordinate = GeneralizedCoordinateSpec(
        position_field="refill_extension_m",
        velocity_field="refill_speed_m_s",
        position_unit="m",
        velocity_unit="m/s",
        input_wrench_type="force",
        input_unit="N",
        component_count=3,
    )
    worker_backend = "click_pen_cam_latch_v1"
    whole_machine_backend = "click_pen_cam_latch_v1"

    def supports(self, plan: MechanismPlan, adapter: AdapterSpec) -> bool:
        return (
            super().supports(plan, adapter)
            and isinstance(adapter.family_spec, dict)
            and adapter.family_spec.get("mechanism_kind")
            == "click_pen_cam_latch"
        )

    def materialize(
        self,
        plan: MechanismPlan,
        adapter: AdapterSpec,
        *,
        workspace_root: str | Path,
        output_dir: str | Path,
    ) -> AdapterSpec:
        from .click_pen_spec import click_pen_cam_latch_from_dict
        from .family_collision import materialize_family_collision_adapter

        spec = click_pen_cam_latch_from_dict(adapter.family_spec)
        return materialize_family_collision_adapter(
            plan,
            adapter,
            spec.bodies,
            workspace_root=workspace_root,
            output_dir=output_dir,
        )

    def load_cases(self, plan: MechanismPlan) -> tuple[LoadCase, ...]:
        from .click_pen_loads import default_click_pen_load_cases

        return default_click_pen_load_cases(plan)

    family_spec_kinds = frozenset({'click_pen_cam_latch'})

    def parse_family_spec(self, raw: Mapping[str, Any]) -> Any:
        from .click_pen_spec import click_pen_cam_latch_from_dict

        return click_pen_cam_latch_from_dict(raw)

    def validate_family_spec(self, adapter: AdapterSpec) -> None:
        self.parse_family_spec(adapter.family_spec)

    def simulate(
        self,
        execution: "ExecutionRequest",
        workspace: Path,
    ) -> dict[str, Any]:
        from .click_pen_runtime import simulate_click_pen

        return simulate_click_pen(execution, workspace)

    def load_envelope(
        self,
        load_cases: tuple["LoadCase", ...],
        comparisons: tuple[dict[str, Any], ...],
    ) -> dict[str, Any] | None:
        from .click_pen_certificate import click_pen_load_envelope

        return click_pen_load_envelope(load_cases, comparisons)

    def certificate_model(
        self,
        adapter: AdapterSpec,
    ) -> dict[str, Any] | None:
        from .click_pen_certificate import click_pen_certificate_model

        return click_pen_certificate_model(adapter)

    def replacement_topology(
        self,
        adapter: AdapterSpec,
    ) -> dict[str, Any] | None:
        from .click_pen_certificate import click_pen_replacement_topology

        return click_pen_replacement_topology(adapter)

    def expected_reduced_ideal_links(
        self,
        plan: MechanismPlan,
        adapter: AdapterSpec,
    ) -> tuple[str, ...]:
        from .click_pen_spec import click_pen_cam_latch_from_dict

        click_pen_cam_latch_from_dict(adapter.family_spec)
        return (
            "ChLinkLockPrismatic",
            "ChLinkLockPrismatic",
            "ChLinkLockCylindrical",
        )


    def ideal_model(
        self,
        plan: MechanismPlan,
        family_spec: Mapping[str, Any] | None,
    ) -> Any | None:
        if (
            not isinstance(family_spec, Mapping)
            or family_spec.get("mechanism_kind") not in self.family_spec_kinds
        ):
            return None
        from .ideal_reference import ClickPenModel

        return ClickPenModel()


    def ideal_comparison_protocol(
        self,
        plan: MechanismPlan,
        family_spec: Mapping[str, Any] | None,
    ) -> tuple[tuple[str, ...], float, str, str] | None:
        if (
            not isinstance(family_spec, Mapping)
            or family_spec.get("mechanism_kind") not in self.family_spec_kinds
        ):
            return None
        from .click_pen_spec import click_pen_cam_latch_from_dict

        spec = click_pen_cam_latch_from_dict(family_spec)
        return (
            ("refill_extension_m",),
            spec.click_state_machine.latch_contact_tolerance_m,
            "m",
            "trusted_click_pen_cam_latch_family_spec",
        )

    def physical_behavior(
        self,
        plan: MechanismPlan,
        load_case: "LoadCase",
        trace: dict[str, Any],
        family_spec: Mapping[str, Any] | None,
    ) -> tuple[dict[str, Any], dict[str, Any], list[str]] | None:
        if (
            not isinstance(family_spec, Mapping)
            or family_spec.get("mechanism_kind") not in self.family_spec_kinds
        ):
            return None
        from .click_pen_comparison import physical_click_pen_behavior

        return physical_click_pen_behavior(plan, load_case, trace)


    def task_channel_layout(
        self,
        plan: MechanismPlan,
        family_spec: Mapping[str, Any],
        body_ids: tuple[str, ...],
    ) -> tuple[
        set[str],
        set[str],
        dict[str, str],
        dict[str, tuple[float, float, float]],
    ] | None:
        input_ids: set[str] = set()
        output_ids: set[str] = set()
        fields: dict[str, str] = {}
        axes: dict[str, tuple[float, float, float]] = {}
        input_id = str(family_spec.get("input_body_id") or "")
        output_id = str(family_spec.get("output_body_id") or "")
        input_ids.add(input_id)
        output_ids.add(output_id)
        fields[output_id] = "refill_extension_m"
        return input_ids, output_ids, fields, axes


class BicycleChainAdapter(MechanismAdapter):
    """Approved candidate for a finite bicycle chain/sprocket drivetrain."""

    plugin_id = "bicycle_chain_sprocket_drive_v2"
    families = frozenset({"compound"})
    # Audited 2026-08-12: this family's lane path DOES consume the
    # acceptance contract — ``ideal_comparison_protocol`` below derives its
    # thresholds from ``plan.acceptance_contract``, and the lane comparison
    # functions ``physical_bicycle_chain_behavior`` /
    # ``compare_bicycle_chain_trace_pair`` read it through
    # ``bicycle_chain_comparison._task_thresholds``.  ``acceptance_contract``
    # therefore stays in the fingerprint; only ``semantic_binding`` is
    # provably unread on any lane path (its sole evaluator is finalize's
    # ``evaluate_task_acceptance``).  Evidence:
    # docs/final_refactor/evidence/lane_fingerprint_drop_audit.md.
    lane_fingerprint_dropped_keys = ("semantic_binding",)
    coordinate = GeneralizedCoordinateSpec(
        position_field="rear_drive_wheel_angle_rad",
        velocity_field="rear_drive_wheel_speed_rad_s",
        position_unit="rad",
        velocity_unit="rad/s",
        input_wrench_type="torque",
        input_unit="N*m",
        component_count=2,
    )
    worker_backend = "bicycle_chain_sprocket_drive_v2"
    whole_machine_backend = "bicycle_chain_sprocket_drive_v2"
    whole_machine_artifact_branch = "physical"
    implementation_capability_ids = frozenset(
        {
            "actuation.indirect_output_only",
            "chain.finite_compliant_force_path",
            "chain.fixed_spatial_flex_zones",
            "chain.matching_sprocket_depth_limited",
            "chain.moving_path_evidence",
            "chain.saved_state_clearance_audit",
            "dashboard.chain_phase_animation",
        }
    )
    model_boundary_ids = frozenset(
        {
            "chain.no_full_flexible_body_physics",
            "chain.no_individual_link_dynamics",
            "chain.no_tooth_contact_dynamics",
        }
    )

    def supports(self, plan: MechanismPlan, adapter: AdapterSpec) -> bool:
        return (
            super().supports(plan, adapter)
            and isinstance(adapter.family_spec, dict)
            and adapter.family_spec.get("mechanism_kind")
            == "bicycle_chain_sprocket_drive"
        )

    def materialize(
        self,
        plan: MechanismPlan,
        adapter: AdapterSpec,
        *,
        workspace_root: str | Path,
        output_dir: str | Path,
    ) -> AdapterSpec:
        from .bicycle_chain_spec import bicycle_chain_from_dict
        from .family_collision import materialize_family_collision_adapter

        spec = bicycle_chain_from_dict(adapter.family_spec)
        return materialize_family_collision_adapter(
            plan,
            adapter,
            spec.bodies,
            workspace_root=workspace_root,
            output_dir=output_dir,
        )

    def load_cases(self, plan: MechanismPlan) -> tuple[LoadCase, ...]:
        from .bicycle_chain_loads import default_bicycle_chain_load_cases

        return default_bicycle_chain_load_cases(plan)

    initial_geometry_evidence_keys = frozenset({"bicycle_chain_geometry"})
    required_trace_audit_sections = frozenset({"moving_chain_path_audit"})

    family_spec_kinds = frozenset({"bicycle_chain_sprocket_drive"})

    def parse_family_spec(self, raw: Mapping[str, Any]) -> Any:
        from .bicycle_chain_spec import bicycle_chain_from_dict

        return bicycle_chain_from_dict(raw)

    def validate_family_spec(self, adapter: AdapterSpec) -> None:
        from .bicycle_chain_spec import bicycle_chain_from_dict

        family_spec = adapter.family_spec
        if isinstance(family_spec, dict):
            family_spec = {
                key: value
                for key, value in family_spec.items()
                if key != "collision_geometry"
            }
        bicycle_chain_from_dict(family_spec)

    def simulate(
        self,
        execution: "ExecutionRequest",
        workspace: Path,
    ) -> dict[str, Any]:
        from .bicycle_chain_runtime import simulate_bicycle_chain

        return simulate_bicycle_chain(execution, workspace)

    def load_envelope(
        self,
        load_cases: tuple["LoadCase", ...],
        comparisons: tuple[dict[str, Any], ...],
    ) -> dict[str, Any] | None:
        from .bicycle_chain_certificate import bicycle_chain_load_envelope

        return bicycle_chain_load_envelope(load_cases, comparisons)

    def certificate_model(
        self,
        adapter: AdapterSpec,
    ) -> dict[str, Any] | None:
        from .bicycle_chain_certificate import bicycle_chain_certificate_model

        return bicycle_chain_certificate_model(adapter)

    def replacement_topology(
        self,
        adapter: AdapterSpec,
    ) -> dict[str, Any] | None:
        from .bicycle_chain_certificate import (
            bicycle_chain_replacement_topology,
        )

        return bicycle_chain_replacement_topology(adapter)

    def expected_reduced_ideal_links(
        self,
        plan: MechanismPlan,
        adapter: AdapterSpec,
    ) -> tuple[str, ...]:
        from .bicycle_chain_spec import bicycle_chain_from_dict

        bicycle_chain_from_dict(adapter.family_spec)
        return (
            "ChLinkLockRevolute",
            "ChLinkLockRevolute",
            "AnalyticalBicycleChainTransmission",
        )


    def build_whole_machine_execution_plan(
        self,
        *,
        plan: MechanismPlan,
        workspace: Path,
        output_dir: Path,
        certificate_path: Path,
        certificate_sha256: str,
        replacement_path: Path,
        replacement_sha256: str,
        certificate: Mapping[str, Any],
        replacement_model: Mapping[str, Any],
        adapter: AdapterSpec,
    ) -> Any:
        from .bicycle_chain_whole_machine import build_bicycle_chain_execution_plan

        return build_bicycle_chain_execution_plan(
            plan=plan,
            workspace=workspace,
            certificate_path=certificate_path,
            certificate_sha256=certificate_sha256,
            replacement_path=replacement_path,
            replacement_sha256=replacement_sha256,
            certificate=certificate,
            replacement_model=replacement_model,
            adapter=adapter,
            plugin=self,
        )

    def load_whole_machine_plan(self, value: Mapping[str, Any]) -> Any | None:
        if value.get("coordinate_kind") != 'bicycle_chain_rotation':
            return None
        from .bicycle_chain_whole_machine import WholeMachineBicycleChainExecutionPlan

        return WholeMachineBicycleChainExecutionPlan.from_dict(dict(value))

    def simulate_whole_machine(
        self,
        plan: Any,
        *,
        workspace_root: Path,
    ) -> dict[str, Any]:
        from .bicycle_chain_whole_machine import (
            simulate_whole_machine_bicycle_chain,
        )
        from .whole_machine_worker import run_family_single_case

        return run_family_single_case(
            plan,
            lambda: simulate_whole_machine_bicycle_chain(
                plan, workspace_root
            ),
        )


    def ideal_model(
        self,
        plan: MechanismPlan,
        family_spec: Mapping[str, Any] | None,
    ) -> Any | None:
        if (
            not isinstance(family_spec, Mapping)
            or family_spec.get("mechanism_kind") not in self.family_spec_kinds
        ):
            return None
        from .ideal_reference import BicycleChainModel

        return BicycleChainModel()


    def ideal_comparison_protocol(
        self,
        plan: MechanismPlan,
        family_spec: Mapping[str, Any] | None,
    ) -> tuple[tuple[str, ...], float, str, str] | None:
        if (
            not isinstance(family_spec, Mapping)
            or family_spec.get("mechanism_kind") not in self.family_spec_kinds
        ):
            return None
        from .bicycle_chain_spec import bicycle_chain_from_dict
        from .ideal_comparison import _fail, _finite

        package_spec = {
            key: value
            for key, value in family_spec.items()
            if key != "collision_geometry"
        }
        spec = bicycle_chain_from_dict(package_spec)
        contract = (
            plan.acceptance_contract
            if isinstance(plan.acceptance_contract, Mapping)
            else {}
        )
        requirements = [
            item
            for item in contract.get("requirements", [])
            if isinstance(item, Mapping)
        ]
        ratio_requirement = next(
            (
                item
                for item in requirements
                if item.get("metric_id") == "angular_ratio"
            ),
            None,
        )
        rotation_requirement = next(
            (
                item
                for item in requirements
                if item.get("metric_id") == "continuous_rotation_rad"
            ),
            None,
        )
        ratio_target = (
            ratio_requirement.get("target")
            if isinstance(ratio_requirement, Mapping)
            else None
        )
        rotation_target = (
            rotation_requirement.get("target")
            if isinstance(rotation_requirement, Mapping)
            else None
        )
        if not isinstance(ratio_target, Mapping) or not isinstance(
            rotation_target, Mapping
        ):
            raise _fail(
                "IDEAL_COMPARISON_THRESHOLD_INVALID",
                "bicycle chain ideal comparison requires frozen task targets",
            )
        ratio_tolerance = _finite(
            ratio_target.get("tolerance"),
            "angular_ratio tolerance",
        )
        minimum_output_rotation = _finite(
            rotation_target.get("value"),
            "continuous_rotation_rad target",
        )
        # The frozen task expresses a relative output/input tolerance.  At the
        # minimum certified powered travel this is the corresponding absolute
        # output-angle error; the task metric still performs the authoritative
        # all-sample signed-ratio check independently.
        threshold = max(
            1.0e-9,
            ratio_tolerance
            * minimum_output_rotation
            / spec.signed_output_per_input,
        )
        return (
            ("rear_drive_wheel_angle_rad",),
            threshold,
            "rad",
            "trusted_bicycle_chain_task_ratio_envelope",
        )

    def physical_behavior(
        self,
        plan: MechanismPlan,
        load_case: "LoadCase",
        trace: dict[str, Any],
        family_spec: Mapping[str, Any] | None,
    ) -> tuple[dict[str, Any], dict[str, Any], list[str]] | None:
        if (
            not isinstance(family_spec, Mapping)
            or family_spec.get("mechanism_kind") not in self.family_spec_kinds
        ):
            return None
        from .bicycle_chain_comparison import (
            physical_bicycle_chain_behavior,
        )

        return physical_bicycle_chain_behavior(plan, load_case, trace)


    def task_channel_layout(
        self,
        plan: MechanismPlan,
        family_spec: Mapping[str, Any],
        body_ids: tuple[str, ...],
    ) -> tuple[
        set[str],
        set[str],
        dict[str, str],
        dict[str, tuple[float, float, float]],
    ] | None:
        input_ids: set[str] = set()
        output_ids: set[str] = set()
        fields: dict[str, str] = {}
        axes: dict[str, tuple[float, float, float]] = {}
        from .bicycle_chain_spec import bicycle_chain_from_dict

        bicycle = bicycle_chain_from_dict(
            {
                key: value
                for key, value in family_spec.items()
                if key != "collision_geometry"
            }
        )
        input_id = bicycle.input_body_id
        output_id = bicycle.output_body_id
        input_ids.add(input_id)
        output_ids.add(output_id)
        fields[input_id] = "input_crank_angle_rad"
        fields[output_id] = "rear_drive_wheel_angle_rad"
        axis = bicycle.axis_abs
        axes[input_id] = axis
        axes[output_id] = axis
        return input_ids, output_ids, fields, axes


class BicycleGroundRollAdapter(MechanismAdapter):
    """Straight-line bicycle test with finite wheel-ground contact."""

    plugin_id = "bicycle_ground_roll_v1"
    families = frozenset({"compound"})
    # Audited 2026-08-12 (catalog-wide consumption audit; per-family evidence
    # in docs/final_refactor/evidence/lane_fingerprint_drop_audit.md): the
    # ground-roll lane path (bicycle_ground_roll_runtime/loads/comparison,
    # ideal_reference, ideal_comparison) is an independent module family that
    # does not import ``bicycle_chain_comparison`` and never reads
    # ``plan.acceptance_contract`` or ``plan.semantic_binding``.
    lane_fingerprint_dropped_keys = (
        "acceptance_contract",
        "semantic_binding",
    )
    coordinate = GeneralizedCoordinateSpec(
        position_field="chassis_position_m",
        velocity_field="chassis_speed_m_s",
        position_unit="m",
        velocity_unit="m/s",
        input_wrench_type="torque",
        input_unit="N*m",
        component_count=3,
    )
    worker_backend = "bicycle_ground_roll_v1"
    whole_machine_backend = "bicycle_ground_roll_v1"
    whole_machine_artifact_branch = "physical"
    implementation_capability_ids = frozenset(
        {
            "actuation.front_wheel_indirect_only",
            "bicycle.chassis_ground_translation",
            "bicycle.front_wheel_passive_rolling",
            "contact.finite_coulomb_wheel_ground",
            "fixture.visible_upright_guide",
        }
    )
    model_boundary_ids = frozenset(
        {
            "bicycle.no_unassisted_balance_claim",
            "bicycle.no_tire_deformation_claim",
            "bicycle.no_steering_dynamics_claim",
        }
    )

    def supports(self, plan: MechanismPlan, adapter: AdapterSpec) -> bool:
        return (
            super().supports(plan, adapter)
            and isinstance(adapter.family_spec, dict)
            and adapter.family_spec.get("mechanism_kind")
            == "bicycle_ground_roll"
        )

    def materialize(
        self,
        plan: MechanismPlan,
        adapter: AdapterSpec,
        *,
        workspace_root: str | Path,
        output_dir: str | Path,
    ) -> AdapterSpec:
        from .bicycle_ground_roll_spec import bicycle_ground_roll_from_dict
        from .family_collision import materialize_family_collision_adapter

        spec = bicycle_ground_roll_from_dict(adapter.family_spec)
        return materialize_family_collision_adapter(
            plan,
            adapter,
            spec.bodies,
            workspace_root=workspace_root,
            output_dir=output_dir,
        )

    def load_cases(self, plan: MechanismPlan) -> tuple[LoadCase, ...]:
        from .bicycle_ground_roll_loads import (
            default_bicycle_ground_roll_load_cases,
        )

        return default_bicycle_ground_roll_load_cases(plan)

    family_spec_kinds = frozenset({"bicycle_ground_roll"})

    def parse_family_spec(self, raw: Mapping[str, Any]) -> Any:
        from .bicycle_ground_roll_spec import bicycle_ground_roll_from_dict

        return bicycle_ground_roll_from_dict(raw)

    def validate_family_spec(self, adapter: AdapterSpec) -> None:
        from .bicycle_ground_roll_spec import bicycle_ground_roll_from_dict

        family_spec = adapter.family_spec
        if isinstance(family_spec, dict):
            family_spec = {
                key: value
                for key, value in family_spec.items()
                if key != "collision_geometry"
            }
        bicycle_ground_roll_from_dict(family_spec)

    def simulate(
        self,
        execution: "ExecutionRequest",
        workspace: Path,
    ) -> dict[str, Any]:
        from .bicycle_ground_roll_runtime import simulate_bicycle_ground_roll

        return simulate_bicycle_ground_roll(execution, workspace)

    def load_envelope(
        self,
        load_cases: tuple["LoadCase", ...],
        comparisons: tuple[dict[str, Any], ...],
    ) -> dict[str, Any] | None:
        from .bicycle_ground_roll_certificate import (
            bicycle_ground_roll_load_envelope,
        )

        return bicycle_ground_roll_load_envelope(load_cases, comparisons)

    def certificate_model(
        self,
        adapter: AdapterSpec,
    ) -> dict[str, Any] | None:
        from .bicycle_ground_roll_certificate import (
            bicycle_ground_roll_certificate_model,
        )

        return bicycle_ground_roll_certificate_model(adapter)

    def replacement_topology(
        self,
        adapter: AdapterSpec,
    ) -> dict[str, Any] | None:
        from .bicycle_ground_roll_certificate import (
            bicycle_ground_roll_replacement_topology,
        )

        return bicycle_ground_roll_replacement_topology(adapter)

    def expected_reduced_ideal_links(
        self,
        plan: MechanismPlan,
        adapter: AdapterSpec,
    ) -> tuple[str, ...]:
        from .bicycle_ground_roll_spec import bicycle_ground_roll_from_dict

        bicycle_ground_roll_from_dict(adapter.family_spec)
        return ("AnalyticalStraightLineNoSlipReference",)


    def build_whole_machine_execution_plan(
        self,
        *,
        plan: MechanismPlan,
        workspace: Path,
        output_dir: Path,
        certificate_path: Path,
        certificate_sha256: str,
        replacement_path: Path,
        replacement_sha256: str,
        certificate: Mapping[str, Any],
        replacement_model: Mapping[str, Any],
        adapter: AdapterSpec,
    ) -> Any:
        from .bicycle_ground_roll_whole_machine import build_bicycle_ground_roll_execution_plan

        return build_bicycle_ground_roll_execution_plan(
            plan=plan,
            workspace=workspace,
            certificate_path=certificate_path,
            certificate_sha256=certificate_sha256,
            replacement_path=replacement_path,
            replacement_sha256=replacement_sha256,
            certificate=certificate,
            replacement_model=replacement_model,
            adapter=adapter,
            plugin=self,
        )

    def load_whole_machine_plan(self, value: Mapping[str, Any]) -> Any | None:
        if value.get("coordinate_kind") != 'bicycle_ground_roll':
            return None
        from .bicycle_ground_roll_whole_machine import WholeMachineBicycleGroundRollExecutionPlan

        return WholeMachineBicycleGroundRollExecutionPlan.from_dict(dict(value))

    def simulate_whole_machine(
        self,
        plan: Any,
        *,
        workspace_root: Path,
    ) -> dict[str, Any]:
        from .bicycle_ground_roll_whole_machine import (
            simulate_whole_machine_bicycle_ground_roll,
            summarize_whole_machine_bicycle_ground_roll,
        )
        from .whole_machine_worker import run_family_single_case

        return run_family_single_case(
            plan,
            lambda: simulate_whole_machine_bicycle_ground_roll(
                plan, workspace_root
            ),
            summarize=lambda trace: (
                summarize_whole_machine_bicycle_ground_roll(plan, trace)
            ),
        )


    def ideal_model(
        self,
        plan: MechanismPlan,
        family_spec: Mapping[str, Any] | None,
    ) -> Any | None:
        if (
            not isinstance(family_spec, Mapping)
            or family_spec.get("mechanism_kind") not in self.family_spec_kinds
        ):
            return None
        from .ideal_reference import BicycleGroundRollModel

        return BicycleGroundRollModel()


    def ideal_comparison_protocol(
        self,
        plan: MechanismPlan,
        family_spec: Mapping[str, Any] | None,
    ) -> tuple[tuple[str, ...], float, str, str] | None:
        if (
            not isinstance(family_spec, Mapping)
            or family_spec.get("mechanism_kind") not in self.family_spec_kinds
        ):
            return None
        from .bicycle_ground_roll_spec import bicycle_ground_roll_from_dict

        package_spec = {
            key: value
            for key, value in family_spec.items()
            if key != "collision_geometry"
        }
        spec = bicycle_ground_roll_from_dict(package_spec)
        return (
            ("front_wheel_angle_rad",),
            max(
                1.0e-9,
                spec.maximum_rolling_slip_ratio
                * spec.minimum_front_rotation_rad,
            ),
            "rad",
            "public_bicycle_ground_roll_slip_envelope",
        )

    def physical_behavior(
        self,
        plan: MechanismPlan,
        load_case: "LoadCase",
        trace: dict[str, Any],
        family_spec: Mapping[str, Any] | None,
    ) -> tuple[dict[str, Any], dict[str, Any], list[str]] | None:
        if (
            not isinstance(family_spec, Mapping)
            or family_spec.get("mechanism_kind") not in self.family_spec_kinds
        ):
            return None
        from .bicycle_ground_roll_comparison import (
            physical_bicycle_ground_roll_behavior,
        )

        return physical_bicycle_ground_roll_behavior(plan, load_case, trace)


    def task_channel_layout(
        self,
        plan: MechanismPlan,
        family_spec: Mapping[str, Any],
        body_ids: tuple[str, ...],
    ) -> tuple[
        set[str],
        set[str],
        dict[str, str],
        dict[str, tuple[float, float, float]],
    ] | None:
        input_ids: set[str] = set()
        output_ids: set[str] = set()
        fields: dict[str, str] = {}
        axes: dict[str, tuple[float, float, float]] = {}
        from .bicycle_ground_roll_spec import (
            bicycle_ground_roll_from_dict,
        )

        ground_roll = bicycle_ground_roll_from_dict(
            {
                key: value
                for key, value in family_spec.items()
                if key != "collision_geometry"
            }
        )
        input_ids.add(ground_roll.rear_wheel_body_id)
        output_ids.update(
            {
                ground_roll.chassis_body_id,
                ground_roll.front_wheel_body_id,
            }
        )
        fields[ground_roll.rear_wheel_body_id] = "rear_wheel_angle_rad"
        fields[ground_roll.front_wheel_body_id] = "front_wheel_angle_rad"
        fields[ground_roll.chassis_body_id] = "chassis_position_m"
        axes[ground_roll.rear_wheel_body_id] = ground_roll.wheel_axis_abs
        axes[ground_roll.front_wheel_body_id] = ground_roll.wheel_axis_abs
        axes[ground_roll.chassis_body_id] = ground_roll.travel_axis_abs
        return input_ids, output_ids, fields, axes


class CompoundMechanismAdapter(MechanismAdapter):
    """Single-input coupled linkages with multiple rigid bodies and pivots."""

    plugin_id = "revolute_linkage_v1"
    families = frozenset({"compound"})
    # Audited 2026-08-12 (catalog-wide consumption audit; per-family evidence
    # in docs/final_refactor/evidence/lane_fingerprint_drop_audit.md): the
    # compound-linkage lane path (compound_runtime, shared comparison/loads,
    # ideal_reference, ideal_comparison) never reads
    # ``plan.acceptance_contract`` or ``plan.semantic_binding``.
    lane_fingerprint_dropped_keys = (
        "acceptance_contract",
        "semantic_binding",
    )
    coordinate = GeneralizedCoordinateSpec(
        position_field="compound_output_angle_rad",
        velocity_field="compound_output_speed_rad_s",
        position_unit="rad",
        velocity_unit="rad/s",
        input_wrench_type="declared_force_or_torque",
        input_unit="N_or_Nm",
        component_count=2,
    )
    worker_backend = "revolute_linkage_v1"
    whole_machine_backend = "revolute_linkage_v1"

    def supports(self, plan: MechanismPlan, adapter: AdapterSpec) -> bool:
        return (
            super().supports(plan, adapter)
            and (
                not isinstance(adapter.family_spec, dict)
                or adapter.family_spec.get("mechanism_kind") in {None, "revolute_linkage"}
            )
        )

    def materialize(
        self,
        plan: MechanismPlan,
        adapter: AdapterSpec,
        *,
        workspace_root: str | Path,
        output_dir: str | Path,
    ) -> AdapterSpec:
        if adapter.family_spec is None:
            raise MechanismValidationError(
                "COMPOUND_SPEC_REQUIRED",
                "compound linkage must bind a validated family_spec before dispatch",
            )
        from .compound_spec import compound_linkage_from_dict
        from .family_collision import materialize_family_collision_adapter

        linkage = compound_linkage_from_dict(adapter.family_spec)
        materialized = materialize_family_collision_adapter(
            plan,
            adapter,
            linkage.bodies,
            workspace_root=workspace_root,
            output_dir=output_dir,
        )
        from .compound_runtime import _validated_compliant_surface_contacts
        from .family_collision import family_collision_geometry

        geometry = family_collision_geometry(
            materialized,
            expected_bodies=linkage.bodies,
        )
        _validated_compliant_surface_contacts(
            materialized.family_spec,
            geometry,
        )
        return materialized

    def load_cases(self, plan: MechanismPlan) -> tuple[LoadCase, ...]:
        from .loads import default_compound_load_cases

        return default_compound_load_cases(plan)

    family_spec_kinds = frozenset({None, 'revolute_linkage'})
    extra_family_spec_fields = frozenset({'compliant_surface_contacts'})

    def parse_family_spec(self, raw: Mapping[str, Any]) -> Any:
        from .compound_spec import compound_linkage_from_dict

        return compound_linkage_from_dict(raw)

    def validate_family_spec(self, adapter: AdapterSpec) -> None:
        self.parse_family_spec(adapter.family_spec)

    def simulate(
        self,
        execution: "ExecutionRequest",
        workspace: Path,
    ) -> dict[str, Any]:
        from .compound_runtime import simulate_compound_linkage

        return simulate_compound_linkage(execution, workspace)

    def expected_reduced_ideal_links(
        self,
        plan: MechanismPlan,
        adapter: AdapterSpec,
    ) -> tuple[str, ...]:
        from .compound_spec import compound_linkage_from_dict

        linkage = compound_linkage_from_dict(adapter.family_spec)
        return tuple("ChLinkLockRevolute" for _ in linkage.revolute_joints)


    def build_whole_machine_execution_plan(
        self,
        *,
        plan: MechanismPlan,
        workspace: Path,
        output_dir: Path,
        certificate_path: Path,
        certificate_sha256: str,
        replacement_path: Path,
        replacement_sha256: str,
        certificate: Mapping[str, Any],
        replacement_model: Mapping[str, Any],
        adapter: AdapterSpec,
    ) -> Any:
        from .whole_machine import build_compound_execution_plan

        return build_compound_execution_plan(
            plan=plan,
            workspace=workspace,
            certificate_path=certificate_path,
            certificate_sha256=certificate_sha256,
            replacement_path=replacement_path,
            replacement_sha256=replacement_sha256,
            certificate=certificate,
            replacement_model=replacement_model,
            adapter=adapter,
            plugin=self,
        )

    def load_whole_machine_plan(self, value: Mapping[str, Any]) -> Any | None:
        if value.get("coordinate_kind") != 'compound_output_rotation':
            return None
        from .whole_machine import WholeMachineCompoundExecutionPlan

        return WholeMachineCompoundExecutionPlan.from_dict(dict(value))

    def simulate_whole_machine(
        self,
        plan: Any,
        *,
        workspace_root: Path,
    ) -> dict[str, Any]:
        from .whole_machine_worker import run_compound_whole_machine

        return run_compound_whole_machine(plan)


    def ideal_model(
        self,
        plan: MechanismPlan,
        family_spec: Mapping[str, Any] | None,
    ) -> Any | None:
        kind = (
            family_spec.get("mechanism_kind")
            if isinstance(family_spec, Mapping)
            else None
        )
        if plan.family != "compound" or kind not in self.family_spec_kinds:
            return None
        from .compound_spec import compound_linkage_from_dict
        from .ideal_reference import (
            IdealReferenceError,
            PlanarFourBarModel,
            _planar_four_bar_geometry,
        )

        try:
            spec = compound_linkage_from_dict(family_spec)
            _planar_four_bar_geometry(spec)
        except MechanismValidationError as exc:
            raise IdealReferenceError(
                "IDEAL_REFERENCE_UNSUPPORTED",
                f"compound geometry has no supported ideal evaluator: {exc}",
            ) from exc
        return PlanarFourBarModel()


    def ideal_comparison_protocol(
        self,
        plan: MechanismPlan,
        family_spec: Mapping[str, Any] | None,
    ) -> tuple[tuple[str, ...], float, str, str] | None:
        kind = (
            family_spec.get("mechanism_kind")
            if isinstance(family_spec, Mapping)
            else None
        )
        if plan.family != "compound" or kind not in self.family_spec_kinds:
            return None
        from .compound_spec import compound_linkage_from_dict

        spec = compound_linkage_from_dict(family_spec)
        return (
            ("compound_output_angle_rad",),
            spec.output_angle_tolerance_rad,
            "rad",
            "trusted_planar_four_bar_family_spec",
        )

    def physical_behavior(
        self,
        plan: MechanismPlan,
        load_case: "LoadCase",
        trace: dict[str, Any],
        family_spec: Mapping[str, Any] | None,
    ) -> tuple[dict[str, Any], dict[str, Any], list[str]] | None:
        kind = (
            family_spec.get("mechanism_kind")
            if isinstance(family_spec, Mapping)
            else None
        )
        if plan.family != "compound" or kind not in self.family_spec_kinds:
            return None
        from .comparison import (
            CompoundKinematicThresholds,
            _compound_behavior_failures,
            _compound_metrics,
            _validate_compound_trace,
        )

        thresholds = CompoundKinematicThresholds.for_plan(plan)
        samples = _validate_compound_trace(
            trace,
            branch="physical",
            plan=plan,
            load_case=load_case,
        )
        metrics = _compound_metrics(
            samples,
            load_case,
            plan.axis_abs,
        )
        from .task_plugins import reviewer_task_program_policy

        reviewer_task_program = (
            reviewer_task_program_policy(load_case, plan=plan) is not None
        )
        failures = _compound_behavior_failures(
            metrics,
            branch="physical",
            load_case=load_case,
            thresholds=thresholds,
            reviewer_task_program=reviewer_task_program,
        )
        return metrics, thresholds.to_dict(), failures


    def task_channel_layout(
        self,
        plan: MechanismPlan,
        family_spec: Mapping[str, Any],
        body_ids: tuple[str, ...],
    ) -> tuple[
        set[str],
        set[str],
        dict[str, str],
        dict[str, tuple[float, float, float]],
    ] | None:
        input_ids: set[str] = set()
        output_ids: set[str] = set()
        fields: dict[str, str] = {}
        axes: dict[str, tuple[float, float, float]] = {}
        input_id = str(family_spec.get("input_body_id") or "")
        output_id = str(family_spec.get("output_body_id") or "")
        input_ids.add(input_id)
        output_ids.add(output_id)
        fields[input_id] = "compound_input_angle_rad"
        fields[output_id] = "compound_output_angle_rad"
        for raw in family_spec.get("revolute_joints", []):
            if isinstance(raw, Mapping):
                child = str(raw.get("child_body_id") or "")
                axis = raw.get("axis_abs")
                if (
                    child
                    and isinstance(axis, (list, tuple))
                    and len(axis) == 3
                ):
                    axes[child] = tuple(float(item) for item in axis)
        return input_ids, output_ids, fields, axes


class PlainPrismaticAdapter(MechanismAdapter):
    plugin_id = "plain_prismatic_v1"
    families = frozenset({"prismatic"})
    # Audited 2026-08-12 (catalog-wide consumption audit; per-family evidence
    # in docs/final_refactor/evidence/lane_fingerprint_drop_audit.md): the
    # plain-prismatic lane path (shared worker/loads/comparison,
    # ideal_reference, ideal_comparison) never reads
    # ``plan.acceptance_contract`` or ``plan.semantic_binding``.
    lane_fingerprint_dropped_keys = (
        "acceptance_contract",
        "semantic_binding",
    )
    coordinate = GeneralizedCoordinateSpec(
        position_field="generalized_progress_m",
        velocity_field="generalized_speed_m_s",
        position_unit="m",
        velocity_unit="m/s",
        input_wrench_type="force",
        input_unit="N",
    )
    worker_backend = "plain_prismatic_v1"
    whole_machine_backend = "plain_prismatic_v1"

    def materialize(
        self,
        plan: MechanismPlan,
        adapter: AdapterSpec,
        *,
        workspace_root: str | Path,
        output_dir: str | Path,
    ) -> AdapterSpec:
        if (
            adapter.adapter_id == DERIVED_PLAIN_PRISMATIC_ADAPTER_ID
            and adapter.geometry is None
        ):
            from .derived_adapter import materialize_plain_prismatic_adapter

            return materialize_plain_prismatic_adapter(
                plan,
                workspace_root=workspace_root,
                output_dir=output_dir,
                family_spec=adapter.family_spec,
            )
        return adapter

    def load_cases(self, plan: MechanismPlan) -> tuple[LoadCase, ...]:
        from .loads import default_prismatic_load_cases

        return default_prismatic_load_cases(plan)

    def expected_reduced_ideal_links(
        self,
        plan: MechanismPlan,
        adapter: AdapterSpec,
    ) -> tuple[str, ...]:
        return ("ChLinkLockPrismatic",)

    def simulate(
        self,
        execution: "ExecutionRequest",
        workspace: Path,
    ) -> dict[str, Any]:
        from .worker import simulate_plain_prismatic

        return simulate_plain_prismatic(execution, workspace)

    def wrench_body_roles(
        self,
        family_spec: Mapping[str, Any],
    ) -> tuple[frozenset[str], frozenset[str]]:
        return (frozenset({"moving_group"}), frozenset({"moving_group"}))


    def build_whole_machine_execution_plan(
        self,
        *,
        plan: MechanismPlan,
        workspace: Path,
        output_dir: Path,
        certificate_path: Path,
        certificate_sha256: str,
        replacement_path: Path,
        replacement_sha256: str,
        certificate: Mapping[str, Any],
        replacement_model: Mapping[str, Any],
        adapter: AdapterSpec,
    ) -> Any:
        from .whole_machine import build_plain_execution_plan

        return build_plain_execution_plan(
            plan=plan,
            workspace=workspace,
            certificate_path=certificate_path,
            certificate_sha256=certificate_sha256,
            replacement_path=replacement_path,
            replacement_sha256=replacement_sha256,
            certificate=certificate,
            replacement_model=replacement_model,
            adapter=adapter,
            plugin=self,
            kind='prismatic',
        )

    def load_whole_machine_plan(self, value: Mapping[str, Any]) -> Any | None:
        if value.get("coordinate_kind") is not None:
            return None
        from .whole_machine import WholeMachineExecutionPlan

        return WholeMachineExecutionPlan.from_dict(dict(value))

    def simulate_whole_machine(
        self,
        plan: Any,
        *,
        workspace_root: Path,
    ) -> dict[str, Any]:
        from .whole_machine_worker import (
            run_plain_whole_machine,
            simulate_plain_prismatic_stages,
        )

        return run_plain_whole_machine(
            plan, lambda: simulate_plain_prismatic_stages(plan)
        )


    def ideal_model(
        self,
        plan: MechanismPlan,
        family_spec: Mapping[str, Any] | None,
    ) -> Any | None:
        if plan.family != "prismatic":
            return None
        from .ideal_reference import PlainCoordinateModel

        return PlainCoordinateModel(
            model_id="plain_prismatic_coordinate_v1",
            position_field="generalized_progress_m",
            velocity_field="generalized_speed_m_s",
            unit="m",
        )


    def ideal_comparison_protocol(
        self,
        plan: MechanismPlan,
        family_spec: Mapping[str, Any] | None,
    ) -> tuple[tuple[str, ...], float, str, str] | None:
        if plan.family != "prismatic":
            return None
        return (
            ("generalized_progress_m",),
            max(1e-4, plan.characteristic_length_m * 0.005),
            "m",
            "plain_prismatic_scaled_protocol",
        )

    def physical_behavior(
        self,
        plan: MechanismPlan,
        load_case: "LoadCase",
        trace: dict[str, Any],
        family_spec: Mapping[str, Any] | None,
    ) -> tuple[dict[str, Any], dict[str, Any], list[str]] | None:
        if plan.family != "prismatic":
            return None
        from .comparison import (
            KinematicThresholds,
            _behavior_failures,
            _physical_scope_evaluation,
            _trace_metrics,
            _validate_trace,
        )

        thresholds = KinematicThresholds.for_plan(plan)
        samples = _validate_trace(
            trace,
            branch="physical",
            plan=plan,
            load_case=load_case,
        )
        metrics = _trace_metrics(samples, load_case, plan.axis_abs)
        failures = _behavior_failures(
            metrics,
            branch="physical",
            load_case=load_case,
            thresholds=thresholds,
        )
        scope_failures, _ = _physical_scope_evaluation(
            plan,
            metrics,
            load_case,
        )
        return metrics, thresholds.to_dict(), [*failures, *scope_failures]


from .nail_clipper_compound_lever import NailClipperCompoundLeverAdapter

_PLUGINS: tuple[MechanismAdapter, ...] = (
    PlainPrismaticAdapter(),
    PlainRevoluteAdapter(),
    RackPinionAdapter(),
    SpurGearClockAdapter(),
    # [kimi-scissor-lift] registered before the generic compound adapter
    ScissorLiftAdapter(),
    MultiOutputShaftAdapter(),
    PrismaticDetentLatchAdapter(),
    ClickPenCamLatchAdapter(),
    BicycleChainAdapter(),
    BicycleGroundRollAdapter(),
    NailClipperCompoundLeverAdapter(),
    CompoundMechanismAdapter(),
)

_PLUGIN_REGISTRATIONS = {
    plugin.plugin_id: {
        "schema_version": "simulation_ready_cad_installed_plugin_v2",
        "plugin_id": plugin.plugin_id,
        "status": "ACTIVE",
        "registration_source": "installed_harness_code",
    }
    for plugin in _PLUGINS
}

_PROJECT_ADAPTERS: dict[str, MechanismAdapter] = {}
_PROJECT_ADAPTER_CACHE: dict[tuple[str, str, str], MechanismAdapter] = {}
REVIEWER_PLUGIN_REGISTRY_SCHEMA = "simulation_ready_cad_reviewer_plugin_registry_v1"
_REVIEWER_ADAPTERS: dict[str, MechanismAdapter] = {}
_REVIEWER_ADAPTER_CACHE: dict[tuple[str, str], MechanismAdapter] = {}
_REVIEWER_ADAPTER_SOURCES: dict[str, dict[str, str]] = {}


def plugin_registration_or_none(plugin_id: str) -> dict[str, Any] | None:
    """Return this process's trusted author marker, never an id-prefix guess."""

    try:
        return plugin_registration_for(str(plugin_id))
    except MechanismValidationError:
        return None


def invoke_mechanism_plugin_callback(
    plugin: MechanismAdapter,
    callback: str,
    *arguments: Any,
    source_component: str,
    unexpected_code: str,
) -> Any:
    """Invoke a MechanismAdapter hook using its trusted registration owner."""

    return invoke_plugin_callback(
        lambda: getattr(plugin, callback)(*arguments),
        registration=plugin_registration_or_none(str(plugin.plugin_id)),
        plugin_id=str(plugin.plugin_id),
        plugin_kind="mechanism",
        callback=callback,
        source_component=source_component,
        unexpected_code=unexpected_code,
    )


def _plugin_callback_failure(
    plugin: MechanismAdapter,
    callback: str,
    exc: BaseException,
    *,
    code: str | None = None,
    message: str | None = None,
    details: Mapping[str, Any] | None = None,
    contract_violation: bool = False,
) -> MechanismValidationError:
    plugin_id = str(plugin.plugin_id)
    registration = plugin_registration_or_none(plugin_id)
    return plugin_boundary_error(
        exc,
        registration=registration,
        plugin_id=plugin_id,
        plugin_kind="mechanism",
        callback=callback,
        source_component=f"mechanism.family_plugins.{callback}",
        code=code,
        message=message,
        details=details,
        contract_violation=contract_violation,
        evidence_validity=EvidenceValidity.INVALID_MEASUREMENT,
    )


def _plugin_supports(
    plugin: MechanismAdapter,
    plan: MechanismPlan,
    adapter: AdapterSpec,
) -> bool:
    try:
        supported = plugin.supports(plan, adapter)
    except Exception as exc:
        boundary_error = _plugin_callback_failure(plugin, "supports", exc)
        if boundary_error is exc:
            raise
        raise boundary_error from exc
    if not isinstance(supported, bool):
        message = "mechanism adapter supports() must return bool"
        raise _plugin_callback_failure(
            plugin,
            "supports",
            TypeError(message),
            code="MECHANISM_ADAPTER_SUPPORTS_INVALID",
            message=message,
            contract_violation=True,
        )
    return supported


def _reviewer_artifact_failure(
    exc: MechanismValidationError,
    *,
    plugin_id: str | None = None,
) -> MechanismValidationError:
    """Attach the explicit reviewer-delivery provenance before registration."""

    if exc.failure_envelope is not None:
        return exc
    return MechanismValidationError(
        exc.code,
        exc.message,
        details=exc.details,
        failure=make_failure_envelope(
            exc.code,
            FailureOwner.REVIEWER_MECHANISM_PLUGIN,
            source_component="mechanism.family_plugins.reviewer_artifact",
            message=exc.message,
            evidence_validity=EvidenceValidity.NOT_APPLICABLE,
            plugin_id=plugin_id,
            plugin_kind="mechanism",
            registration_source="reviewer_mechanism_adapter",
            details=exc.details,
        ),
    )


def _reviewer_registry_failure(
    code: str,
    message: str,
) -> MechanismValidationError:
    """The Loop writes the registry; malformed registry structure is Harness-owned."""

    return MechanismValidationError(
        code,
        message,
        failure=make_failure_envelope(
            code,
            FailureOwner.HARNESS_CODE,
            source_component="mechanism.family_plugins.reviewer_registry",
            message=message,
            evidence_validity=EvidenceValidity.NOT_APPLICABLE,
        ),
    )


def reviewer_adapter_source(plugin_id: str) -> dict[str, str] | None:
    """Return the hash-bound source of one AI-reviewer MechanismAdapter."""

    source = _REVIEWER_ADAPTER_SOURCES.get(plugin_id)
    return None if source is None else dict(source)


def clear_reviewer_mechanism_adapters() -> None:
    """Drop in-process reviewer adapters. Tests reload a clean registry."""

    _REVIEWER_ADAPTERS.clear()
    _REVIEWER_ADAPTER_CACHE.clear()
    _REVIEWER_ADAPTER_SOURCES.clear()


def drop_reviewer_mechanism_adapter(plugin_id: str) -> None:
    """Unbind one in-process reviewer adapter so a revision can replace it."""

    _REVIEWER_ADAPTERS.pop(plugin_id, None)
    _REVIEWER_ADAPTER_SOURCES.pop(plugin_id, None)
    stale = [
        key
        for key, plugin in _REVIEWER_ADAPTER_CACHE.items()
        if plugin.plugin_id == plugin_id
    ]
    for key in stale:
        _REVIEWER_ADAPTER_CACHE.pop(key, None)


def _reviewer_kind_owner(mechanism_kind: Any) -> str | None:
    for plugin in _PLUGINS:
        if mechanism_kind in plugin.family_spec_kinds:
            return plugin.plugin_id
    for plugin in _REVIEWER_ADAPTERS.values():
        if mechanism_kind in plugin.family_spec_kinds:
            return plugin.plugin_id
    return None


def _load_reviewer_mechanism_adapter(
    source: str | Path,
    expected_sha256: str,
    *,
    bind: bool = True,
    replace_plugin_id: str | None = None,
) -> MechanismAdapter:
    """Import one AI-reviewer MechanismAdapter from its hash-bound file."""

    path = Path(source).resolve()
    if not str(path).endswith(".py") or path.is_symlink() or not path.is_file():
        raise MechanismValidationError(
            "REVIEWER_MECHANISM_ADAPTER_PATH_INVALID",
            "reviewer adapter must be a regular Python file",
        )
    if sha256_file(path) != expected_sha256:
        raise MechanismValidationError(
            "REVIEWER_MECHANISM_ADAPTER_HASH_MISMATCH",
            "reviewer adapter artifact does not match its approved hash",
        )
    cache_key = (str(path), expected_sha256)
    if bind:
        cached = _REVIEWER_ADAPTER_CACHE.get(cache_key)
        if cached is not None:
            return cached
    module_name = (
        "_simulation_ready_reviewer_adapter_"
        f"{expected_sha256}_{hashlib.sha256(str(path).encode('utf-8')).hexdigest()}"
    )
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise MechanismValidationError(
            "REVIEWER_MECHANISM_ADAPTER_LOAD_FAILED",
            f"cannot load reviewer adapter artifact: {path.name}",
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(module_name, None)
        raise MechanismValidationError(
            "REVIEWER_MECHANISM_ADAPTER_LOAD_FAILED", str(exc)
        ) from exc
    plugin = getattr(module, "MECHANISM_ADAPTER", None)
    if not isinstance(plugin, MechanismAdapter):
        raise MechanismValidationError(
            "REVIEWER_MECHANISM_ADAPTER_EXPORT_INVALID",
            "reviewer adapter file must export MECHANISM_ADAPTER",
        )
    if not plugin.plugin_id.startswith("reviewer."):
        raise MechanismValidationError(
            "REVIEWER_MECHANISM_ADAPTER_ID_INVALID",
            "reviewer adapter plugin_id must start with 'reviewer.'",
        )
    required_overrides = (
        (
            "materialize",
            "REVIEWER_MECHANISM_ADAPTER_MATERIALIZE_REQUIRED",
        ),
        (
            "parse_family_spec",
            "REVIEWER_MECHANISM_ADAPTER_FAMILY_SPEC_PARSER_REQUIRED",
        ),
        (
            "simulate",
            "REVIEWER_MECHANISM_ADAPTER_SIMULATE_REQUIRED",
        ),
        (
            "ideal_model",
            "REVIEWER_MECHANISM_ADAPTER_IDEAL_MODEL_REQUIRED",
        ),
        (
            "ideal_comparison_protocol",
            "REVIEWER_MECHANISM_ADAPTER_IDEAL_COMPARISON_PROTOCOL_REQUIRED",
        ),
        (
            "physical_behavior",
            "REVIEWER_MECHANISM_ADAPTER_PHYSICAL_BEHAVIOR_REQUIRED",
        ),
        (
            "task_channel_layout",
            "REVIEWER_MECHANISM_ADAPTER_TASK_CHANNEL_LAYOUT_REQUIRED",
        ),
        (
            "required_geometry_proofs",
            "REVIEWER_MECHANISM_ADAPTER_GEOMETRY_PROOFS_REQUIRED",
        ),
    )
    for method_name, code in required_overrides:
        if getattr(type(plugin), method_name) is getattr(
            MechanismAdapter, method_name
        ):
            raise MechanismValidationError(
                code,
                f"reviewer adapter must implement {method_name}()",
            )
    for method_name in (
        "validated_geometry_proof_requirements",
        "validate_execution_geometry_proofs",
    ):
        if getattr(type(plugin), method_name) is not getattr(
            MechanismAdapter, method_name
        ):
            raise MechanismValidationError(
                "REVIEWER_MECHANISM_ADAPTER_GEOMETRY_GATE_OVERRIDE_FORBIDDEN",
                f"reviewer adapter cannot override Harness-owned {method_name}()",
            )
    if plugin.plugin_id in _PLUGIN_REGISTRATIONS or plugin.plugin_id in _PROJECT_ADAPTERS:
        raise MechanismValidationError(
            "REVIEWER_MECHANISM_ADAPTER_ID_CONFLICT",
            f"reviewer adapter conflicts with installed plugin {plugin.plugin_id!r}",
        )
    kinds = plugin.family_spec_kinds
    if (
        not isinstance(kinds, frozenset)
        or not kinds
        or any(not isinstance(kind, str) or not kind for kind in kinds)
    ):
        raise MechanismValidationError(
            "REVIEWER_MECHANISM_ADAPTER_KIND_INVALID",
            "reviewer adapter must claim one or more mechanism_kind strings",
        )
    for kind in kinds:
        owner = _reviewer_kind_owner(kind)
        if (
            owner is not None
            and owner != plugin.plugin_id
            and owner != replace_plugin_id
        ):
            raise MechanismValidationError(
                "REVIEWER_MECHANISM_ADAPTER_KIND_CONFLICT",
                f"mechanism_kind {kind!r} is already owned by {owner!r}",
            )
    from .joint_realization import SUPPORTED_PROOF_KINDS

    proof_kinds = plugin.geometry_proof_kinds
    if (
        "geometry_proof_kinds" not in type(plugin).__dict__
        or not isinstance(proof_kinds, frozenset)
        or not proof_kinds <= SUPPORTED_PROOF_KINDS
    ):
        raise MechanismValidationError(
            "REVIEWER_MECHANISM_ADAPTER_GEOMETRY_PROOF_KIND_INVALID",
            "reviewer adapter must explicitly declare its registered geometry "
            "proof kinds; use an explicit empty frozenset only when the "
            "approved mechanism has no physical joint realization claim",
        )
    if (
        proof_kinds
        and "joint_realization" not in plugin.initial_geometry_evidence_keys
    ):
        raise MechanismValidationError(
            "REVIEWER_MECHANISM_ADAPTER_GEOMETRY_EVIDENCE_REQUIRED",
            "reviewer adapter must require joint_realization initial geometry "
            "evidence",
        )
    profiles = plugin.wrench_policy_profiles
    from .test_load_profiles import registered_test_load_profiles

    known_profiles = frozenset().union(*_WRENCH_POLICY_PROFILES.values()) | {
        profile.profile_kind for profile in registered_test_load_profiles()
    }
    if not isinstance(profiles, frozenset) or not profiles <= known_profiles:
        raise MechanismValidationError(
            "REVIEWER_MECHANISM_ADAPTER_WRENCH_PROFILES_INVALID",
            "reviewer adapter must declare existing wrench_policy_profiles",
        )
    existing = _REVIEWER_ADAPTERS.get(plugin.plugin_id)
    if existing is not None and existing is not plugin:
        prior = _REVIEWER_ADAPTER_SOURCES.get(plugin.plugin_id)
        replacing = (
            replace_plugin_id is not None
            and plugin.plugin_id == replace_plugin_id
        )
        if (
            prior is not None
            and prior.get("sha256") != expected_sha256
            and not replacing
        ):
            raise MechanismValidationError(
                "REVIEWER_MECHANISM_ADAPTER_ID_CONFLICT",
                f"reviewer adapter {plugin.plugin_id!r} is already bound",
            )
    if bind:
        _REVIEWER_ADAPTERS[plugin.plugin_id] = plugin
        _REVIEWER_ADAPTER_CACHE[cache_key] = plugin
        _REVIEWER_ADAPTER_SOURCES[plugin.plugin_id] = {
            "path": str(path),
            "sha256": expected_sha256,
        }
    return plugin


def load_reviewer_mechanism_adapter(
    source: str | Path,
    expected_sha256: str,
    *,
    bind: bool = True,
    replace_plugin_id: str | None = None,
) -> MechanismAdapter:
    """Import a reviewer adapter and preserve its delivery authority on error."""

    try:
        return _load_reviewer_mechanism_adapter(
            source,
            expected_sha256,
            bind=bind,
            replace_plugin_id=replace_plugin_id,
        )
    except MechanismValidationError as exc:
        owned = _reviewer_artifact_failure(
            exc,
            plugin_id=replace_plugin_id,
        )
        if owned is exc:
            raise
        raise owned from exc


def load_reviewer_plugin_registry(
    registry_path: str | Path,
    *,
    workspace_root: str | Path | None = None,
) -> tuple[MechanismAdapter, ...]:
    """Load hash-bound AI-reviewer adapters recorded for one project Loop."""

    # The registry is the exact project-owned snapshot for this Harness run.
    # A long-lived event-kernel process may have loaded older plugin bytes on
    # the previous preflight; retaining them would turn a legitimate plugin
    # replacement into an ID/kind ownership conflict.
    clear_reviewer_mechanism_adapters()
    from .task_plugins import clear_reviewer_task_adapters

    clear_reviewer_task_adapters()
    path = Path(registry_path)
    if path.is_symlink() or not path.is_file():
        return ()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise _reviewer_registry_failure(
            "REVIEWER_MECHANISM_ADAPTER_REGISTRY_INVALID",
            f"reviewer plugin registry is unreadable: {exc}",
        ) from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != REVIEWER_PLUGIN_REGISTRY_SCHEMA
        or not isinstance(payload.get("plugins"), list)
    ):
        raise _reviewer_registry_failure(
            "REVIEWER_MECHANISM_ADAPTER_REGISTRY_INVALID",
            "reviewer plugin registry is not the current schema",
        )
    root = None if workspace_root is None else Path(workspace_root).resolve()
    loaded: list[MechanismAdapter] = []
    for item in payload["plugins"]:
        if not isinstance(item, dict):
            raise _reviewer_registry_failure(
                "REVIEWER_MECHANISM_ADAPTER_REGISTRY_INVALID",
                "reviewer plugin registry entries must be objects",
            )
        relative = item.get("path")
        digest = item.get("sha256")
        if not isinstance(relative, str) or not isinstance(digest, str):
            raise _reviewer_registry_failure(
                "REVIEWER_MECHANISM_ADAPTER_REGISTRY_INVALID",
                "reviewer plugin registry entries need path and sha256",
            )
        source = Path(relative)
        if not source.is_absolute():
            if root is None:
                source = path.parent / relative
            else:
                source = root.joinpath(*Path(relative).parts)
        loaded.append(load_reviewer_mechanism_adapter(source, digest))
    task_entries = payload.get("task_plugins")
    if task_entries is None:
        return tuple(loaded)
    if not isinstance(task_entries, list):
        raise _reviewer_registry_failure(
            "REVIEWER_MECHANISM_ADAPTER_REGISTRY_INVALID",
            "reviewer plugin registry task_plugins must be a list",
        )
    from .task_plugins import load_reviewer_task_plugin_entries

    load_reviewer_task_plugin_entries(
        task_entries,
        registry_path=path,
        workspace_root=workspace_root,
    )
    return tuple(loaded)


def exec_project_adapter_in_process(
    package_dir: str | Path,
    relative: str,
    expected_sha256: str,
) -> MechanismAdapter:
    """Import one approved adapter in this process. Isolation wraps this."""

    try:
        source = ensure_inside(package_dir, relative)
    except (TypeError, ValueError) as exc:
        raise MechanismValidationError(
            "PROJECT_MECHANISM_ADAPTER_PATH_INVALID", str(exc)
        ) from exc
    if not relative.endswith(".py") or not source.is_file():
        raise MechanismValidationError(
            "PROJECT_MECHANISM_ADAPTER_PATH_INVALID",
            "project_adapter_file must name a Python package artifact",
        )
    if sha256_file(source) != expected_sha256:
        raise MechanismValidationError(
            "PROJECT_MECHANISM_ADAPTER_HASH_MISMATCH",
            f"project adapter artifact does not match its approved hash: {relative}",
        )

    resolved_package = str(Path(package_dir).resolve())
    cache_key = (resolved_package, relative, expected_sha256)
    plugin = _PROJECT_ADAPTER_CACHE.get(cache_key)
    if plugin is not None:
        return plugin

    for cached_key, cached_plugin in _PROJECT_ADAPTER_CACHE.items():
        if cached_key[2] != expected_sha256:
            continue
        _PROJECT_ADAPTER_CACHE[cache_key] = cached_plugin
        _PROJECT_ADAPTERS[cached_plugin.plugin_id] = cached_plugin
        return cached_plugin

    package_token = hashlib.sha256(resolved_package.encode("utf-8")).hexdigest()
    module_name = (
        f"_simulation_ready_project_adapter_{expected_sha256}_{package_token}"
    )
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise MechanismValidationError(
            "PROJECT_MECHANISM_ADAPTER_LOAD_FAILED",
            f"cannot load project adapter artifact: {relative}",
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(module_name, None)
        raise MechanismValidationError(
            "PROJECT_MECHANISM_ADAPTER_LOAD_FAILED", str(exc)
        ) from exc
    plugin = getattr(module, "MECHANISM_ADAPTER", None)
    if not isinstance(plugin, MechanismAdapter):
        raise MechanismValidationError(
            "PROJECT_MECHANISM_ADAPTER_EXPORT_INVALID",
            "project adapter file must export MECHANISM_ADAPTER",
        )
    if not plugin.plugin_id.startswith("project."):
        raise MechanismValidationError(
            "PROJECT_MECHANISM_ADAPTER_ID_INVALID",
            "project adapter plugin_id must start with 'project.'",
        )
    if plugin.plugin_id in _PLUGIN_REGISTRATIONS:
        raise MechanismValidationError(
            "PROJECT_MECHANISM_ADAPTER_ID_CONFLICT",
            f"project adapter conflicts with installed plugin {plugin.plugin_id!r}",
        )
    profiles = plugin.wrench_policy_profiles
    from .test_load_profiles import registered_test_load_profiles

    known_profiles = frozenset().union(*_WRENCH_POLICY_PROFILES.values()) | {
        profile.profile_kind for profile in registered_test_load_profiles()
    }
    if not isinstance(profiles, frozenset) or not profiles <= known_profiles:
        raise MechanismValidationError(
            "PROJECT_MECHANISM_ADAPTER_WRENCH_PROFILES_INVALID",
            "project adapter must declare existing wrench_policy_profiles",
        )
    _PROJECT_ADAPTERS[plugin.plugin_id] = plugin
    _PROJECT_ADAPTER_CACHE[cache_key] = plugin
    return plugin


def _load_project_mechanism_adapter(
    package_dir: str | Path,
    relative: str,
    expected_sha256: str,
    *,
    probe_timeout_seconds: float | None = None,
    call_timeout_seconds: float | None = None,
) -> MechanismAdapter:
    """Load the single approved Python adapter named by a design package."""

    from .project_adapter_isolation import (
        DEFAULT_CALL_SECONDS,
        DEFAULT_PROBE_SECONDS,
        DEFAULT_RSS_BYTES,
        IsolatedProjectAdapter,
        isolation_enabled,
        load_isolated_project_adapter,
    )

    try:
        source = ensure_inside(package_dir, relative)
    except (TypeError, ValueError) as exc:
        raise MechanismValidationError(
            "PROJECT_MECHANISM_ADAPTER_PATH_INVALID", str(exc)
        ) from exc
    if not relative.endswith(".py") or not source.is_file():
        raise MechanismValidationError(
            "PROJECT_MECHANISM_ADAPTER_PATH_INVALID",
            "project_adapter_file must name a Python package artifact",
        )
    if sha256_file(source) != expected_sha256:
        raise MechanismValidationError(
            "PROJECT_MECHANISM_ADAPTER_HASH_MISMATCH",
            f"project adapter artifact does not match its approved hash: {relative}",
        )
    resolved_package = str(Path(package_dir).resolve())
    cache_key = (resolved_package, relative, expected_sha256)
    plugin = _PROJECT_ADAPTER_CACHE.get(cache_key)
    if plugin is not None:
        return plugin
    call_timeout = (
        DEFAULT_CALL_SECONDS
        if call_timeout_seconds is None
        else call_timeout_seconds
    )
    probe_timeout = (
        DEFAULT_PROBE_SECONDS
        if probe_timeout_seconds is None
        else probe_timeout_seconds
    )
    if isolation_enabled():
        descriptor = None
        reused_rss = DEFAULT_RSS_BYTES
        reused_timeout = call_timeout
        for cached_key, cached_plugin in _PROJECT_ADAPTER_CACHE.items():
            if cached_key[2] != expected_sha256:
                continue
            cached_descriptor = getattr(cached_plugin, "_descriptor", None)
            if cached_descriptor is None:
                continue
            descriptor = cached_descriptor
            reused_rss = getattr(
                cached_plugin, "_max_rss_bytes", DEFAULT_RSS_BYTES
            )
            reused_timeout = getattr(
                cached_plugin, "_call_timeout_seconds", call_timeout
            )
            break
        if descriptor is None:
            plugin = load_isolated_project_adapter(
                package_dir,
                relative,
                expected_sha256,
                probe_timeout_seconds=probe_timeout,
                call_timeout_seconds=call_timeout,
            )
        else:
            plugin = IsolatedProjectAdapter(
                package_dir=package_dir,
                relative=relative,
                expected_sha256=expected_sha256,
                descriptor=descriptor,
                call_timeout_seconds=reused_timeout,
                max_rss_bytes=reused_rss,
            )
        _PROJECT_ADAPTERS[plugin.plugin_id] = plugin
        _PROJECT_ADAPTER_CACHE[cache_key] = plugin
        return plugin
    return exec_project_adapter_in_process(
        package_dir, relative, expected_sha256
    )


def _project_resource_failure(
    code: str,
    message: str,
    *,
    details: Mapping[str, Any] | None = None,
) -> MechanismValidationError:
    return MechanismValidationError(
        code,
        message,
        details=details,
        failure=make_failure_envelope(
            code,
            FailureOwner.WORKER_PLAN,
            source_component="mechanism.family_plugins.project_artifact",
            message=message,
            evidence_validity=EvidenceValidity.NOT_APPLICABLE,
            details=details,
        ),
    )


def load_project_mechanism_adapter(
    package_dir: str | Path,
    relative: str,
    expected_sha256: str,
    *,
    probe_timeout_seconds: float | None = None,
    call_timeout_seconds: float | None = None,
) -> MechanismAdapter:
    """Load an approved project adapter and keep resource ownership explicit."""

    try:
        return _load_project_mechanism_adapter(
            package_dir,
            relative,
            expected_sha256,
            probe_timeout_seconds=probe_timeout_seconds,
            call_timeout_seconds=call_timeout_seconds,
        )
    except MechanismValidationError as exc:
        if exc.failure_envelope is not None:
            raise
        raise _project_resource_failure(
            exc.code,
            exc.message,
            details=exc.details,
        ) from exc


def _project_mechanism_adapter(
    plan: MechanismPlan,
    adapter: AdapterSpec,
) -> MechanismAdapter | None:
    family_spec = adapter.family_spec
    if not isinstance(family_spec, Mapping):
        return None
    relative = family_spec.get("project_adapter_file")
    if relative is None:
        return None
    if not isinstance(relative, str) or not relative.endswith(".py"):
        raise _project_resource_failure(
            "PROJECT_MECHANISM_ADAPTER_PATH_INVALID",
            "adapter.family_spec.project_adapter_file must name a Python artifact",
        )
    expected = plan.artifact_hashes.get(relative)
    if expected is None:
        raise _project_resource_failure(
            "PROJECT_MECHANISM_ADAPTER_HASH_MISSING",
            f"artifact_hashes has no digest for {relative!r}",
        )
    plugin = load_project_mechanism_adapter(
        plan.package_dir,
        relative,
        expected,
    )
    if not _plugin_supports(plugin, plan, adapter):
        message = f"project adapter {plugin.plugin_id!r} does not support this plan"
        raise _plugin_callback_failure(
            plugin,
            "supports",
            MechanismValidationError(
                "PROJECT_MECHANISM_ADAPTER_UNSUPPORTED", message
            ),
            code="PROJECT_MECHANISM_ADAPTER_UNSUPPORTED",
            message=message,
            contract_violation=True,
        )
    return plugin


def family_spec_plugin_for_kind(mechanism_kind: Any) -> MechanismAdapter | None:
    """Resolve the registered family that owns a ``mechanism_kind`` value."""

    for plugin in _PLUGINS:
        if mechanism_kind in plugin.family_spec_kinds:
            return plugin
    for plugin in _REVIEWER_ADAPTERS.values():
        if mechanism_kind in plugin.family_spec_kinds:
            return plugin
    return None


def _wrench_policy(
    plugin_id: str,
    profile: str,
    wrench_kind: str,
    category: str,
    law_id: str,
    target_roles: tuple[str, ...],
    *,
    task_input: bool = False,
    transmission: bool = False,
    support: bool = False,
    capability_required: bool = False,
    component_provenance_required: bool = False,
) -> WrenchPolicy:
    return WrenchPolicy(
        policy_id=f"{plugin_id}.{profile}.{wrench_kind}",
        wrench_kind=wrench_kind,
        category=category,
        law_id=law_id,
        target_roles=target_roles,
        task_input=task_input,
        transmission=transmission,
        support=support,
        capability_required=capability_required,
        component_provenance_required=component_provenance_required,
    )


_WRENCH_POLICY_PROFILES: dict[str, frozenset[str]] = {
    "plain_prismatic_v1": frozenset(
        {"task_input", "task_input_controller"}
    ),
    "plain_revolute_v1": frozenset(
        {
            "task_input",
            "task_input_controller",
            "support",
            "constitutive",
            "limit",
        }
    ),
    "rack_pinion_v1": frozenset(
        {
            "task_input",
            "task_input_controller_150",
            "support",
            "transmission",
            "limit",
            "resistance",
        }
    ),
    "spur_gear_clock_v1": frozenset({"accumulator", "support"}),
    "multi_output_shaft_coupling_v1": frozenset({"accumulator", "support"}),
    "hydraulic_scissor_lift_v1": frozenset(
        {
            "task_input",
            "task_input_dynamic_axis",
            "task_input_dynamic_axis_reaction",
            "probe",
            "support",
            "constitutive",
            "limit",
        }
    ),
    "prismatic_detent_latch_v1": frozenset(
        {"task_input", "probe", "transmission", "limit"}
    ),
    "click_pen_cam_latch_v1": frozenset(
        {"task_input", "transmission", "limit"}
    ),
    "revolute_linkage_v1": frozenset(
        {"task_input", "support", "transmission", "limit"}
    ),
    "bicycle_chain_sprocket_drive_v2": frozenset(
        {
            "task_input",
            "support",
            "transmission",
        }
    ),
    "bicycle_ground_roll_v1": frozenset({"task_input"}),
    "nail_clipper_compound_lever_v1": frozenset(
        {"task_input", "support", "constitutive", "transmission", "limit"}
    ),
}


def wrench_policies_for_plugin(plugin_id: str) -> tuple[WrenchPolicy, ...]:
    """Return exact wrench permissions for ACTIVE or one bound test dispatch."""

    registration = plugin_registration_for(plugin_id)
    if not registration_allows_runtime_execution(registration):
        raise MechanismValidationError(
            "WRENCH_PLUGIN_NOT_ACTIVE",
            f"plugin {plugin_id!r} cannot declare wrench policies while inactive",
        )
    profiles = _WRENCH_POLICY_PROFILES.get(plugin_id)
    if profiles is None and plugin_id in _PROJECT_ADAPTERS:
        profiles = _PROJECT_ADAPTERS[plugin_id].wrench_policy_profiles
    if profiles is None and plugin_id in _REVIEWER_ADAPTERS:
        profiles = _REVIEWER_ADAPTERS[plugin_id].wrench_policy_profiles
    if profiles is None:
        raise MechanismValidationError(
            "WRENCH_POLICY_PROFILE_MISSING",
            f"plugin {plugin_id!r} has no wrench policy profile",
        )
    roles = ("declared_input", "task_output", "internal_moving")
    policies: list[WrenchPolicy] = []
    if "task_input" in profiles:
        kinds = (
            ("torque",)
            if plugin_id in {
                "bicycle_chain_sprocket_drive_v2",
                "bicycle_ground_roll_v1",
                "nail_clipper_compound_lever_v1",
            }
            else ("force", "torque")
        )
        for kind in kinds:
            policies.append(
                _wrench_policy(
                    plugin_id,
                    "task_input",
                    kind,
                    "task_input",
                    "validator_load_phase_v1",
                    ("declared_input",),
                    task_input=True,
                )
            )
    for profile, law_id in (
        (
            "task_input_controller",
            "validator_bounded_controller_phase_v1",
        ),
        (
            "task_input_controller_150",
            "validator_bounded_controller_phase_150_v1",
        ),
        (
            "task_input_reaction",
            "validator_opposed_load_phase_v1",
        ),
        (
            "task_input_dynamic_axis",
            "validator_dynamic_actuator_phase_v1",
        ),
        (
            "task_input_dynamic_axis_reaction",
            "validator_dynamic_actuator_reaction_phase_v1",
        ),
    ):
        if profile not in profiles:
            continue
        for kind in ("force", "torque"):
            policies.append(
                _wrench_policy(
                    plugin_id,
                    profile,
                    kind,
                    "task_input",
                    law_id,
                    ("declared_input",),
                    task_input=True,
                )
            )
    if "support" in profiles:
        for kind in ("force", "torque"):
            policies.append(
                _wrench_policy(
                    plugin_id,
                    (
                        "bicycle_finite_support"
                        if plugin_id == "bicycle_chain_sprocket_drive_v2"
                        else "support"
                    ),
                    kind,
                    (
                        "finite_support"
                        if plugin_id == "bicycle_chain_sprocket_drive_v2"
                        else "physical_support"
                    ),
                    (
                        "finite_6d_linear_support_v2"
                        if plugin_id == "bicycle_chain_sprocket_drive_v2"
                        else "finite_compliant_support_v1"
                    ),
                    roles,
                    support=True,
                )
            )
    if "constitutive" in profiles:
        for kind in ("force", "torque"):
            policies.append(
                _wrench_policy(
                    plugin_id,
                    "constitutive",
                    kind,
                    "constitutive_support",
                    "finite_constitutive_response_v1",
                    roles,
                    support=True,
                )
            )
    if "limit" in profiles:
        for kind in ("force", "torque"):
            policies.append(
                _wrench_policy(
                    plugin_id,
                    "limit",
                    kind,
                    "finite_limit",
                    "geometry_gated_limit_v1",
                    roles,
                    support=True,
                )
            )
    if "transmission" in profiles:
        kinds = (
            ("torque",)
            if plugin_id == "bicycle_chain_sprocket_drive_v2"
            else ("force", "torque")
        )
        for kind in kinds:
            policies.append(
                _wrench_policy(
                    plugin_id,
                    (
                        "bicycle_chain_transmission"
                        if plugin_id == "bicycle_chain_sprocket_drive_v2"
                        else "transmission"
                    ),
                    kind,
                    (
                        "transmission"
                        if plugin_id == "bicycle_chain_sprocket_drive_v2"
                        else "analytical_transmission"
                    ),
                    (
                        "finite_geometry_gated_chain_v2"
                        if plugin_id == "bicycle_chain_sprocket_drive_v2"
                        else "geometry_gated_transmission_v1"
                    ),
                    roles,
                    transmission=True,
                    capability_required=True,
                )
            )
    if "passive_storage" in profiles:
        policies.append(
            _wrench_policy(
                plugin_id,
                "passive_storage",
                "torque",
                "passive_energy_storage",
                "finite_passive_centering_v1",
                ("declared_input",),
                support=True,
                capability_required=True,
            )
        )
    if "passive_dissipation" in profiles:
        policies.append(
            _wrench_policy(
                plugin_id,
                "passive_dissipation",
                "torque",
                "passive_dissipation",
                "bounded_dissipative_drag_v1",
                ("task_output",),
            )
        )
    if "passive_test_load" in profiles:
        for kind in ("force", "torque"):
            policies.append(
                _wrench_policy(
                    plugin_id,
                    "passive_test_load",
                    kind,
                    "passive_test_load",
                    "validator_dissipative_test_load_v1",
                    ("task_output", "external_passive_fixture"),
                )
            )
    if "resistance" in profiles:
        for kind in ("force", "torque"):
            policies.append(
                _wrench_policy(
                    plugin_id,
                    "resistance",
                    kind,
                    "validator_resistance",
                    "validator_resistance_v1",
                    roles,
                )
            )
    if "probe" in profiles:
        for kind in ("force", "torque"):
            policies.append(
                _wrench_policy(
                    plugin_id,
                    "probe",
                    kind,
                    "validator_probe",
                    "validator_probe_v1",
                    roles,
                )
            )
    if "accumulator" in profiles:
        policies.append(
            _wrench_policy(
                plugin_id,
                "accumulator",
                "torque",
                "audited_wrench_accumulator",
                "audited_component_sum_v1",
                roles,
                component_provenance_required=True,
            )
        )
    return tuple(policies)


def registered_mechanism_adapters() -> tuple[MechanismAdapter, ...]:
    # Project adapters stay package-local and are never advertised to a
    # different task as built-in Harness capability.
    return _PLUGINS


def candidate_mechanism_adapter_inventory() -> list[dict[str, Any]]:
    """Compatibility view for the removed staged candidate workflow."""

    return []


def registered_candidate_definition(plugin_id: str) -> dict[str, Any]:
    """Compatibility view for the removed staged candidate workflow.

    The shipped registry contains only installed ACTIVE adapters; candidate
    records no longer exist, so every lookup fails closed.
    """

    raise MechanismValidationError(
        "MECHANISM_PLUGIN_NOT_REGISTERED",
        f"plugin {plugin_id!r} has no candidate state in this Harness",
    )


def plugin_registration_for(
    plugin_id: str,
    *,
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    registration = _PLUGIN_REGISTRATIONS.get(plugin_id)
    if registration is not None:
        expected = {
            "schema_version": "simulation_ready_cad_installed_plugin_v2",
            "plugin_id": plugin_id,
            "status": "ACTIVE",
            "registration_source": "installed_harness_code",
        }
    elif plugin_id in _PROJECT_ADAPTERS:
        expected = {
            "schema_version": "simulation_ready_cad_project_plugin_v1",
            "plugin_id": plugin_id,
            "status": "ACTIVE",
            "registration_source": "project_design_package",
        }
        registration = expected
    elif plugin_id in _REVIEWER_ADAPTERS:
        expected = {
            "schema_version": "simulation_ready_cad_reviewer_plugin_v1",
            "plugin_id": plugin_id,
            "status": "ACTIVE",
            "registration_source": "reviewer_mechanism_adapter",
        }
        registration = expected
    else:
        raise MechanismValidationError(
            "MECHANISM_PLUGIN_NOT_REGISTERED",
            f"plugin {plugin_id!r} is not installed in this Harness",
        )
    if registration != expected:
        raise MechanismValidationError(
            "MECHANISM_PLUGIN_REGISTRATION_INVALID",
            f"plugin registration is not the exact installed record for {plugin_id!r}",
        )
    return dict(registration)


def registration_allows_runtime_execution(
    registration: Mapping[str, Any],
) -> bool:
    """Accept only an adapter installed in the current Harness package."""

    if not isinstance(registration, Mapping):
        return False
    plugin_id = registration.get("plugin_id")
    status = registration.get("status")
    if not isinstance(plugin_id, str) or status != "ACTIVE":
        return False
    try:
        current = plugin_registration_for(plugin_id)
    except MechanismValidationError:
        return False
    return (
        current.get("status") == status
        and dict(registration) == current
    )


def _reviewer_mechanism_adapter(
    plan: MechanismPlan,
    adapter: AdapterSpec,
) -> MechanismAdapter | None:
    family_spec = adapter.family_spec
    if not isinstance(family_spec, dict):
        return None
    binding = family_spec.get("reviewer_adapter")
    if isinstance(binding, Mapping):
        source = binding.get("path")
        digest = binding.get("sha256")
        if isinstance(source, str) and isinstance(digest, str) and source and digest:
            plugin = load_reviewer_mechanism_adapter(source, digest)
            if _plugin_supports(plugin, plan, adapter):
                return plugin
            message = (
                f"reviewer adapter {plugin.plugin_id!r} does not support this plan"
            )
            raise _plugin_callback_failure(
                plugin,
                "supports",
                MechanismValidationError(
                    "REVIEWER_MECHANISM_ADAPTER_UNSUPPORTED", message
                ),
                code="REVIEWER_MECHANISM_ADAPTER_UNSUPPORTED",
                message=message,
                contract_violation=True,
            )
    kind = family_spec.get("mechanism_kind")
    plugin = family_spec_plugin_for_kind(kind)
    if plugin is None or plugin.plugin_id not in _REVIEWER_ADAPTERS:
        return None
    if not _plugin_supports(plugin, plan, adapter):
        return None
    return plugin


def mechanism_adapter_for(
    plan: MechanismPlan,
    adapter: AdapterSpec,
) -> MechanismAdapter:
    project_plugin = _project_mechanism_adapter(plan, adapter)
    if project_plugin is not None:
        registration = plugin_registration_for(project_plugin.plugin_id)
        if not registration_allows_runtime_execution(registration):
            raise MechanismValidationError(
                "MECHANISM_PLUGIN_NOT_ACTIVE",
                f"plugin {project_plugin.plugin_id!r} is not active",
            )
        return project_plugin
    reviewer_plugin = _reviewer_mechanism_adapter(plan, adapter)
    if reviewer_plugin is not None:
        try:
            registration = plugin_registration_for(reviewer_plugin.plugin_id)
            if not registration_allows_runtime_execution(registration):
                raise MechanismValidationError(
                    "MECHANISM_PLUGIN_NOT_ACTIVE",
                    f"plugin {reviewer_plugin.plugin_id!r} is not active",
                )
        except MechanismValidationError as exc:
            if exc.failure_envelope is not None:
                raise
            raise _plugin_callback_failure(
                reviewer_plugin,
                "runtime_binding",
                exc,
                code="REVIEWER_MECHANISM_ADAPTER_RUNTIME_BINDING_INVALID",
                message=f"{exc.code}: {exc.message}",
                contract_violation=True,
            ) from exc
        return reviewer_plugin
    matches = [plugin for plugin in _PLUGINS if _plugin_supports(plugin, plan, adapter)]
    if not matches:
        raise MechanismValidationError(
            "MECHANISM_PLUGIN_UNAVAILABLE",
            f"no trusted mechanism plugin supports family {plan.family!r}",
        )
    if len(matches) > 1:
        raise MechanismValidationError(
            "MECHANISM_PLUGIN_AMBIGUOUS",
            f"multiple trusted mechanism plugins support family {plan.family!r}",
        )
    plugin = matches[0]
    registration = plugin_registration_for(plugin.plugin_id)
    if not registration_allows_runtime_execution(registration):
        raise MechanismValidationError(
            "MECHANISM_PLUGIN_NOT_ACTIVE",
            f"plugin {plugin.plugin_id!r} is not active",
        )
    return plugin


def adapter_spec_for_plan(plan: MechanismPlan) -> AdapterSpec:
    if plan.adapter_id is None:
        raise MechanismValidationError(
            "MECHANISM_ADAPTER_UNAVAILABLE",
            f"mechanism {plan.capsule_id!r} has no trusted execution data",
        )
    try:
        return (
            adapter_from_dict(plan.adapter_spec)
            if plan.adapter_spec is not None
            else get_adapter(plan.adapter_id)
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise MechanismValidationError(
            "MECHANISM_ADAPTER_UNKNOWN", str(exc)
        ) from exc


def mechanism_adapter_for_plan(
    plan: MechanismPlan,
) -> tuple[MechanismAdapter, AdapterSpec]:
    adapter = adapter_spec_for_plan(plan)
    return mechanism_adapter_for(plan, adapter), adapter


def mechanism_adapter_inventory() -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for plugin in _PLUGINS:
        registration = plugin_registration_for(plugin.plugin_id)
        if registration.get("status") != "ACTIVE":
            continue
        inventory.append(
            {
                "plugin_id": plugin.plugin_id,
                "families": sorted(plugin.families),
                "coordinate": {
                    "position_field": plugin.coordinate.position_field,
                    "velocity_field": plugin.coordinate.velocity_field,
                    "position_unit": plugin.coordinate.position_unit,
                    "velocity_unit": plugin.coordinate.velocity_unit,
                    "input_wrench_type": plugin.coordinate.input_wrench_type,
                    "input_unit": plugin.coordinate.input_unit,
                    "component_count": plugin.coordinate.component_count,
                },
                "worker_backend": plugin.worker_backend,
                "whole_machine_backend": plugin.whole_machine_backend,
                "registration": registration,
                "allowed_wrench_policies": [
                    policy.to_dict()
                    for policy in wrench_policies_for_plugin(
                        plugin.plugin_id
                    )
                ],
            }
        )
    return inventory


_WORKER_HIDDEN_FAMILY_SPEC_FIELDS = frozenset(
    {"project_adapter_file", "mechanism_kind"}
)


def public_simulation_interface_catalog(
    catalog_joint_bindings: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Worker-facing list of installed plugins a joint template may dock to."""

    joints_by_plugin: dict[str, list[dict[str, Any]]] = {}
    examples_by_plugin: dict[str, set[str]] = {}
    kinds_from_catalog: dict[str, set[Any]] = {}
    for raw in catalog_joint_bindings:
        plugin_id = raw.get("plugin_id")
        if not isinstance(plugin_id, str) or not plugin_id:
            continue
        template_id = raw.get("template_id")
        variant_id = raw.get("variant_id")
        if not isinstance(template_id, str) or not isinstance(variant_id, str):
            continue
        joints_by_plugin.setdefault(plugin_id, []).append(
            {
                "template_id": template_id,
                "variant_id": variant_id,
                "family": raw.get("family"),
                "mechanism_kind": raw.get("mechanism_kind"),
                "runtime_status": raw.get("runtime_status"),
                "required_instance_fields": list(
                    raw.get("required_instance_fields") or ()
                ),
                "goal": raw.get("goal") if isinstance(raw.get("goal"), str) else None,
                "capability_ids": sorted(
                    str(item)
                    for item in (raw.get("capability_ids") or ())
                    if isinstance(item, str) and item
                ),
                "model_boundary_ids": sorted(
                    str(item)
                    for item in (raw.get("model_boundary_ids") or ())
                    if isinstance(item, str) and item
                ),
            }
        )
        kinds_from_catalog.setdefault(plugin_id, set()).add(
            raw.get("mechanism_kind")
        )
        for path in raw.get("family_spec_examples") or ():
            if isinstance(path, str) and path:
                examples_by_plugin.setdefault(plugin_id, set()).add(path)

    plugins: list[dict[str, Any]] = []
    for plugin in sorted(_PLUGINS, key=lambda item: item.plugin_id):
        registration = plugin_registration_for(plugin.plugin_id)
        if registration.get("status") != "ACTIVE":
            continue
        kinds = set(plugin.family_spec_kinds)
        kinds.update(kinds_from_catalog.get(plugin.plugin_id, ()))
        extras = sorted(
            field
            for field in plugin.extra_family_spec_fields
            if field not in _WORKER_HIDDEN_FAMILY_SPEC_FIELDS
        )
        catalog_joints = sorted(
            joints_by_plugin.get(plugin.plugin_id, ()),
            key=lambda item: (item["template_id"], item["variant_id"]),
        )
        capability_ids = set(plugin.implementation_capability_ids)
        model_boundary_ids = set(plugin.model_boundary_ids)
        for joint in catalog_joints:
            capability_ids.update(joint.get("capability_ids") or ())
            model_boundary_ids.update(joint.get("model_boundary_ids") or ())
        plugins.append(
            {
                "plugin_id": plugin.plugin_id,
                "families": sorted(plugin.families),
                "mechanism_kinds": sorted(
                    kinds,
                    key=lambda kind: (kind is not None, str(kind or "")),
                ),
                "extra_family_spec_fields": extras,
                "implementation_capability_ids": sorted(capability_ids),
                "model_boundary_ids": sorted(model_boundary_ids),
                "geometry_proof_kinds": sorted(
                    plugin.geometry_proof_kinds
                ),
                "coordinate": {
                    "position_field": plugin.coordinate.position_field,
                    "velocity_field": plugin.coordinate.velocity_field,
                    "position_unit": plugin.coordinate.position_unit,
                    "velocity_unit": plugin.coordinate.velocity_unit,
                    "input_wrench_type": plugin.coordinate.input_wrench_type,
                    "input_unit": plugin.coordinate.input_unit,
                    "component_count": plugin.coordinate.component_count,
                },
                "example_backend": {
                    "worker_backend": plugin.worker_backend,
                    "whole_machine_backend": plugin.whole_machine_backend,
                },
                "catalog_joints": catalog_joints,
                "family_spec_examples": sorted(
                    examples_by_plugin.get(plugin.plugin_id, ())
                ),
            }
        )
    return {
        "schema_version": "simulation_ready_cad_simulation_interfaces_v1",
        "document_role": (
            "Worked examples of installed Harness backends that already "
            "run catalog joints. Author a joint template as catalog-shaped "
            "JSON. Do not write, import, or invoke a plugin to create that "
            "JSON."
        ),
        "backend_contract": {
            "owner": "Harness MechanismAdapter",
            "family_spec": (
                "CAD instance document filled after a joint template is "
                "approved. The named Harness backend parses it."
            ),
            "execution": (
                "The backend's simulate() path on a PyChrono "
                "worker_backend / whole_machine_backend."
            ),
            "coordinates": (
                "GeneralizedCoordinateSpec: position, velocity, input wrench."
            ),
            "supports": (
                "physical_supports and constitutive_supports: finite "
                "bushings, axis springs, restoring springs, and optional "
                "compliant surface contacts."
            ),
            "wrenches": "validator-owned wrench ledger on that backend",
        },
        "binding_rule": (
            "A project joint JSON is a planning contract, not a plugin. "
            "Copy the catalog joint field layout closest to the missing "
            "capability. Look at plugins[] only as examples of how an "
            "admitted joint already binds to a Harness backend: "
            "mechanism_kinds, extra_family_spec_fields, "
            "implementation_capability_ids, model_boundary_ids, and "
            "geometry_proof_kinds. Every projected geometry proof is a "
            "mandatory authoritative-STEP gate before review or physics. "
            "Finite compliance appears on those examples "
            "(axis_spring, compliant_surface_contacts, restoring_spring, "
            "chain.fixed_spatial_flex_zones); it is not a fourth family. "
            "Name plugin_id only when an existing example backend already "
            "implements this variant (runtime_status ACTIVE). Otherwise "
            "set runtime_status IMPLEMENTATION_REQUIRED and plugin_id "
            "null with a unique mechanism_kind. Do not submit Python. "
            "The AI reviewer implements that kind after approving the "
            "joint table."
        ),
        "plugins": plugins,
    }
