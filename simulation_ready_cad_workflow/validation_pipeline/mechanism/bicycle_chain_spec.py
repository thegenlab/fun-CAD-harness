from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .compound_spec import CompoundBodySpec
from .io import canonical_sha256
from .models import (
    MechanismValidationError,
    finite,
    identifier,
    vector3,
)


BICYCLE_CHAIN_SCHEMA_VERSION = (
    "simulation_ready_cad_bicycle_chain_sprocket_drive_family_v2_proposed"
)
BICYCLE_CHAIN_MECHANISM_KIND = "bicycle_chain_sprocket_drive"
BICYCLE_CHAIN_PLUGIN_ID = "bicycle_chain_sprocket_drive_v2"
BICYCLE_CHAIN_TEMPLATE_PATH = (
    "simulation_ready_cad_workflow/validation_pipeline/mechanism/"
    "templates/bicycle_chain_sprocket_drive_v2.json"
)
BICYCLE_CHAIN_IMPLEMENTATION_APPROVAL_ID = (
    "capability-implementation-bicycle-8b19-r01-input-only-v1"
)
BICYCLE_CHAIN_IMPLEMENTATION_REQUEST_SHA256 = (
    "e285530e764dc06312d45705e5bd9580bc1128d6db90d16a5c22a25be24dc349"
)
BICYCLE_CHAIN_CRANK_PERTURBATION_PROGRAM = (
    "bicycle_reverse_crank_and_resistance"
)
BICYCLE_CHAIN_CRANK_PERTURBATION_TORQUE_NM = 0.25
MAX_SPROCKET_CONTACT_COLLISION_COMPONENT_IDS = 64
# Compatibility aliases remain importable while the candidate registration
# migrates to the approved input-crank-only diagnostic name.  They no longer
# describe or authorize a wrench on the rear wheel.
BICYCLE_CHAIN_REAR_RESISTANCE_PROGRAM = (
    BICYCLE_CHAIN_CRANK_PERTURBATION_PROGRAM
)
BICYCLE_CHAIN_REAR_RESISTANCE_TORQUE_NM = (
    BICYCLE_CHAIN_CRANK_PERTURBATION_TORQUE_NM
)


def _fail(code: str, message: str) -> MechanismValidationError:
    return MechanismValidationError(code, message)


def _clamp(value: float, maximum_magnitude: float) -> float:
    return max(-maximum_magnitude, min(maximum_magnitude, value))


def _positive(value: Any, label: str) -> float:
    result = finite(value, label)
    if result <= 0.0:
        raise _fail(
            "BICYCLE_CHAIN_PARAMETER_INVALID",
            f"{label} must be positive",
        )
    return result


def _nonnegative(value: Any, label: str) -> float:
    result = finite(value, label)
    if result < 0.0:
        raise _fail(
            "BICYCLE_CHAIN_PARAMETER_INVALID",
            f"{label} cannot be negative",
        )
    return result


def _approved_template_payload() -> dict[str, Any]:
    template = (
        Path(__file__).resolve().parent
        / "templates"
        / "bicycle_chain_sprocket_drive_v2.json"
    )
    try:
        value = json.loads(template.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise _fail(
            "BICYCLE_CHAIN_TEMPLATE_INVALID",
            f"cannot read the governed r04 template at {BICYCLE_CHAIN_TEMPLATE_PATH}",
        ) from exc
    if not isinstance(value, dict):
        raise _fail(
            "BICYCLE_CHAIN_TEMPLATE_INVALID",
            "the governed r04 template must be a JSON object",
        )
    return value


def _assert_closed_schema_tree(
    actual: Any,
    expected: Any,
    *,
    path: str = "family_spec",
) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            raise _fail(
                "BICYCLE_CHAIN_SPEC_SCHEMA_INVALID",
                f"{path} must be an object",
            )
        actual_keys = set(actual)
        expected_keys = set(expected)
        if actual_keys != expected_keys:
            raise _fail(
                "BICYCLE_CHAIN_SPEC_SCHEMA_INVALID",
                f"{path} must use the exact closed r04 schema; "
                f"unknown={sorted(actual_keys - expected_keys)!r}, "
                f"missing={sorted(expected_keys - actual_keys)!r}. "
                f"Copy {BICYCLE_CHAIN_TEMPLATE_PATH}",
            )
        for key in expected:
            _assert_closed_schema_tree(
                actual[key],
                expected[key],
                path=f"{path}.{key}",
            )
        return
    if isinstance(expected, list):
        if path.endswith(".contact_collision_component_ids"):
            if (
                not isinstance(actual, Sequence)
                or isinstance(actual, (str, bytes))
                or not actual
            ):
                raise _fail(
                    "BICYCLE_CHAIN_SPEC_SCHEMA_INVALID",
                    f"{path} must contain one or more entries",
                )
            for index, item in enumerate(actual):
                _assert_closed_schema_tree(
                    item,
                    expected[0],
                    path=f"{path}[{index}]",
                )
            return
        if (
            not isinstance(actual, Sequence)
            or isinstance(actual, (str, bytes))
            or len(actual) != len(expected)
        ):
            raise _fail(
                "BICYCLE_CHAIN_SPEC_SCHEMA_INVALID",
                f"{path} must contain exactly {len(expected)} entries",
            )
        for index, item in enumerate(expected):
            _assert_closed_schema_tree(
                actual[index],
                item,
                path=f"{path}[{index}]",
            )
        return
    if expected is None:
        if actual is not None:
            raise _fail(
                "BICYCLE_CHAIN_SPEC_SCHEMA_INVALID",
                f"{path} must be null",
            )
        return
    if isinstance(expected, bool):
        valid = isinstance(actual, bool)
    elif isinstance(expected, (int, float)):
        valid = (
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and math.isfinite(float(actual))
        )
    else:
        valid = isinstance(actual, type(expected))
    if not valid:
        raise _fail(
            "BICYCLE_CHAIN_SPEC_SCHEMA_INVALID",
            f"{path} has the wrong value type",
        )


def _assert_exact_approved_tree(
    actual: Any,
    expected: Any,
    *,
    path: str = "family_spec",
) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            raise _fail(
                "BICYCLE_CHAIN_SPEC_SCHEMA_INVALID",
                f"{path} must be an object",
            )
        actual_keys = set(actual)
        expected_keys = set(expected)
        if actual_keys != expected_keys:
            raise _fail(
                "BICYCLE_CHAIN_SPEC_SCHEMA_INVALID",
                f"{path} must use the exact closed r04 schema; "
                f"unknown={sorted(actual_keys - expected_keys)!r}, "
                f"missing={sorted(expected_keys - actual_keys)!r}",
            )
        for key in expected:
            _assert_exact_approved_tree(
                actual[key],
                expected[key],
                path=f"{path}.{key}",
            )
        return
    if isinstance(expected, list):
        if (
            not isinstance(actual, Sequence)
            or isinstance(actual, (str, bytes))
            or len(actual) != len(expected)
        ):
            raise _fail(
                "BICYCLE_CHAIN_SPEC_SCHEMA_INVALID",
                f"{path} must contain exactly {len(expected)} entries",
            )
        for index, item in enumerate(expected):
            _assert_exact_approved_tree(
                actual[index],
                item,
                path=f"{path}[{index}]",
            )
        return
    if isinstance(expected, bool):
        matches = isinstance(actual, bool) and actual is expected
    elif isinstance(expected, (int, float)):
        matches = (
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and math.isfinite(float(actual))
            and float(actual) == float(expected)
        )
    else:
        matches = type(actual) is type(expected) and actual == expected
    if not matches:
        raise _fail(
            "BICYCLE_CHAIN_PARAMETER_NOT_APPROVED",
            f"{path} must equal the implementation-approved r04 value {expected!r}",
        )


@dataclass(frozen=True)
class BicycleFrameSpec:
    frame_id: str
    body_id: str
    origin_abs_m: tuple[float, float, float]
    axis_abs: tuple[float, float, float]


@dataclass(frozen=True)
class BicycleInputPortSpec:
    port_id: str
    target_body_id: str
    frame_id: str
    wrench_kind: str
    axis_abs: tuple[float, float, float]
    maximum_magnitude_nm: float


@dataclass(frozen=True)
class BicycleObservableSpec:
    observable_id: str
    body_id: str
    frame_id: str
    quantity: str


@dataclass(frozen=True)
class BicycleBearingSpec:
    support_id: str
    fixed_body_id: str
    moving_body_id: str
    frame_id: str
    radial_clearance_m: float
    radial_stiffness_n_m: float
    axial_stiffness_n_m: float
    translation_damping_n_s_m: float
    tilt_stiffness_n_m_rad: float
    tilt_damping_n_m_s_rad: float
    maximum_force_n: float
    maximum_torque_nm: float
    spin_damping_n_m_s_rad: float = 0.0
    return_stiffness_n_m_rad: float = 0.0
    return_damping_n_m_s_rad: float = 0.0
    maximum_return_torque_nm: float = 0.0
    constitutive_model: str = ""
    axis_abs: tuple[float, float, float] = (0.0, 1.0, 0.0)
    translation_stiffness_xyz_n_m: tuple[float, float, float] = (
        0.0,
        0.0,
        0.0,
    )
    translation_damping_xyz_n_s_m: tuple[float, float, float] = (
        0.0,
        0.0,
        0.0,
    )
    off_axis_rotation_stiffness_xyz_nm_rad: tuple[float, float, float] = (
        0.0,
        0.0,
        0.0,
    )
    off_axis_rotation_damping_xyz_nm_s_rad: tuple[float, float, float] = (
        0.0,
        0.0,
        0.0,
    )
    primary_axis_home_rad: float = 0.0
    primary_axis_centering_stiffness_nm_rad: float = 0.0
    primary_axis_damping_nm_s_rad: float = 0.0
    maximum_total_support_torque_nm: float = 0.0
    primary_axis_maximum_abs_torque_nm: float | None = None
    zero_state_stored_energy_j: float = 0.0


@dataclass(frozen=True)
class BicycleBodyDynamicsSpec:
    body_id: str
    rigid_group_id: str
    source_solid_ids: tuple[str, ...]
    role: str
    fixed: bool
    mass_kg: float
    center_of_mass_abs_m: tuple[float, float, float]
    inertia_diagonal_kg_m2: tuple[float, float, float]
    initial_position_abs_m: tuple[float, float, float]
    initial_orientation_wxyz: tuple[float, float, float, float]
    initial_linear_velocity_abs_m_s: tuple[float, float, float]
    initial_angular_velocity_abs_rad_s: tuple[float, float, float]


@dataclass(frozen=True)
class BicycleSprocketSpec:
    sprocket_id: str
    body_id: str
    shaft_frame_id: str
    pitch_loop_center_abs_m: tuple[float, float, float]
    pitch_axis_abs: tuple[float, float, float]
    pitch_radius_m: float
    toroidal_minor_radius_m: float
    step_solid_id: str
    contact_collision_component_ids: tuple[str, ...]
    approved_contact_collision_component_ids: tuple[str, ...]


@dataclass(frozen=True)
class BicycleGeometryEntitySpec:
    body_id: str
    step_solid_id: str
    surface_kind: str
    major_radius_m: float
    minor_radius_m: float
    center_abs_m: tuple[float, float, float]
    axis_abs: tuple[float, float, float]
    surface_exposure_required: bool
    minimum_exposed_face_count: int


@dataclass(frozen=True)
class BicycleGeometryGateSpec:
    gate_id: str
    evidence_source: str
    required_step_entities: tuple[BicycleGeometryEntitySpec, ...]
    pitch_radius_tolerance_m: float
    pitch_center_tolerance_m: float
    maximum_axis_parallel_error_rad: float
    maximum_pitch_plane_offset_m: float
    visible_chain_part_id: str
    maximum_chain_sprocket_separation_m: float
    visible_chain_wrap_band_m: float
    minimum_visible_chain_wrap_angle_rad: float
    minimum_visible_chain_wrap_sample_count: int
    zero_transmission_wrench_when_inactive: bool


@dataclass(frozen=True)
class BicycleTaskPhaseSpec:
    phase_id: str
    wrench_kind: str
    direction_abs: tuple[float, float, float] | None
    magnitude_si: float
    duration_s: float

    @property
    def input_torque_nm(self) -> float:
        return self.magnitude_si if self.wrench_kind == "torque" else 0.0


@dataclass(frozen=True)
class BicycleTaskProgramSpec:
    program_id: str
    integration_step_s: float
    saved_sample_stride_steps: int
    maximum_sample_gap_cadences: float
    minimum_powered_sample_count: int
    phases: tuple[BicycleTaskPhaseSpec, ...]
    engagement_phase_ids: tuple[str, ...]
    powered_phase_ids: tuple[str, ...]
    return_phase_ids: tuple[str, ...]
    adaptive_phase_ids: tuple[str, ...]
    return_input_policy: str
    engagement_task_motion_credit: bool
    engagement_ratio_metric_credit: bool

    def phase(self, phase_id: str) -> BicycleTaskPhaseSpec:
        for phase in self.phases:
            if phase.phase_id == phase_id:
                return phase
        raise KeyError(phase_id)


@dataclass(frozen=True)
class BicycleHeldOutValidationDomainSpec:
    case_id: str
    selection_authority: str
    minimum_torque_nm: float
    nominal_torque_nm: float
    maximum_torque_nm: float
    required_evidence_torque_nm: tuple[float, ...]
    fixed_parameters: tuple[tuple[str, float], ...]
    no_other_parameter_may_vary: bool
    no_hidden_defaults: bool


@dataclass(frozen=True)
class BicycleChainTransmissionSpec:
    support_id: str
    input_body_id: str
    output_body_id: str
    input_frame_id: str
    output_frame_id: str
    input_pitch_radius_m: float
    output_pitch_radius_m: float
    chain_stiffness_n_m: float
    chain_damping_n_s_m: float
    maximum_differential_tension_n: float
    maximum_alignment_error_m: float
    maximum_axis_angle_error_rad: float
    maximum_abs_chain_error_m: float = 0.0
    initial_chain_error_m: float = 0.0
    initial_chain_error_rate_m_s: float = 0.0
    initial_stored_energy_j: float = 0.0
    physical_link_kind: str = ""
    ideal_constraint_in_physical_branch: bool = False
    preload_allowed: bool = False
    output_feedforward_allowed: bool = False
    geometry_gate_id: str = ""

    @property
    def signed_output_per_input(self) -> float:
        return self.input_pitch_radius_m / self.output_pitch_radius_m


@dataclass(frozen=True)
class BicycleAcceptanceThresholds:
    """Validator-owned values derived from a frozen task contract."""

    powered_rotation_min_rad: float
    signed_output_per_input: float
    angular_ratio_tolerance: float
    return_to_home_max_rad: float

    def validated(self) -> BicycleAcceptanceThresholds:
        powered = _positive(
            self.powered_rotation_min_rad,
            "powered_rotation_min_rad",
        )
        ratio = finite(
            self.signed_output_per_input,
            "signed_output_per_input",
        )
        tolerance = _nonnegative(
            self.angular_ratio_tolerance,
            "angular_ratio_tolerance",
        )
        return_home = _nonnegative(
            self.return_to_home_max_rad,
            "return_to_home_max_rad",
        )
        if ratio <= 0.0 or tolerance >= 1.0:
            raise _fail(
                "BICYCLE_CHAIN_TASK_CONTRACT_INVALID",
                "ratio must be positive and tolerance must be below one",
            )
        return BicycleAcceptanceThresholds(
            powered_rotation_min_rad=powered,
            signed_output_per_input=ratio,
            angular_ratio_tolerance=tolerance,
            return_to_home_max_rad=return_home,
        )


@dataclass(frozen=True)
class BicycleChainSpec:
    schema_version: str
    mechanism_kind: str
    mechanism_id: str
    bodies: tuple[CompoundBodySpec, ...]
    fixed_body_id: str
    input_body_id: str
    output_body_id: str
    input_frame: BicycleFrameSpec
    output_frame: BicycleFrameSpec
    input_port: BicycleInputPortSpec
    input_observable: BicycleObservableSpec
    output_observable: BicycleObservableSpec
    actuation_observable: BicycleObservableSpec
    transmission: BicycleChainTransmissionSpec
    input_bearing: BicycleBearingSpec
    output_bearing: BicycleBearingSpec
    maximum_resistive_test_load_nm: float
    max_support_drift_m: float
    max_off_axis_rotation_rad: float
    friction: float
    capability_state: str = ""
    task_id: str = ""
    body_dynamics: tuple[BicycleBodyDynamicsSpec, ...] = ()
    sprockets: tuple[BicycleSprocketSpec, ...] = ()
    geometry_gate: BicycleGeometryGateSpec | None = None
    task_program: BicycleTaskProgramSpec | None = None
    held_out_validation_domain: BicycleHeldOutValidationDomainSpec | None = None
    acceptance_thresholds: BicycleAcceptanceThresholds | None = None
    required_negative_tests: tuple[str, ...] = ()
    source_payload_json: str | None = None

    @property
    def axis_abs(self) -> tuple[float, float, float]:
        return self.input_frame.axis_abs

    @property
    def input_origin_abs_m(self) -> tuple[float, float, float]:
        return self.input_frame.origin_abs_m

    @property
    def output_origin_abs_m(self) -> tuple[float, float, float]:
        return self.output_frame.origin_abs_m

    @property
    def input_frame_id(self) -> str:
        return self.input_frame.frame_id

    @property
    def output_frame_id(self) -> str:
        return self.output_frame.frame_id

    @property
    def input_port_id(self) -> str:
        return self.input_port.port_id

    @property
    def input_observable_id(self) -> str:
        return self.input_observable.observable_id

    @property
    def output_observable_id(self) -> str:
        return self.output_observable.observable_id

    @property
    def actuation_observable_id(self) -> str:
        return self.actuation_observable.observable_id

    @property
    def input_pitch_radius_m(self) -> float:
        return self.transmission.input_pitch_radius_m

    @property
    def output_pitch_radius_m(self) -> float:
        return self.transmission.output_pitch_radius_m

    @property
    def input_sprocket(self) -> BicycleSprocketSpec:
        return next(
            item for item in self.sprockets
            if item.body_id == self.input_body_id
        )

    @property
    def output_sprocket(self) -> BicycleSprocketSpec:
        return next(
            item for item in self.sprockets
            if item.body_id == self.output_body_id
        )

    @property
    def signed_output_per_input(self) -> float:
        return self.transmission.signed_output_per_input

    @property
    def chain_stiffness_n_m(self) -> float:
        return self.transmission.chain_stiffness_n_m

    @property
    def chain_damping_n_s_m(self) -> float:
        return self.transmission.chain_damping_n_s_m

    @property
    def maximum_differential_tension_n(self) -> float:
        return self.transmission.maximum_differential_tension_n

    @property
    def maximum_alignment_error_m(self) -> float:
        return self.transmission.maximum_alignment_error_m

    @property
    def maximum_axis_angle_error_rad(self) -> float:
        return self.transmission.maximum_axis_angle_error_rad

    @property
    def maximum_task_input_torque_nm(self) -> float:
        return self.input_port.maximum_magnitude_nm

    @property
    def nominal_input_torque_nm(self) -> float:
        if self.task_program is not None:
            return self.task_program.phase(
                "measured_powered_drive"
            ).input_torque_nm
        return 0.60 * self.maximum_task_input_torque_nm

    def expected_runtime_bindings(
        self,
        *,
        task_id: str,
        assembly_contract: Mapping[str, Any],
        assembly_contract_sha256: str,
    ) -> dict[str, Any]:
        identifier(task_id, "task_id")
        if self.task_id and task_id != self.task_id:
            raise _fail(
                "BICYCLE_CHAIN_BINDING_INVALID",
                "runtime task ID differs from the unique governed task target",
            )
        contract_binding = resolve_bicycle_contract_bindings(
            self,
            assembly_contract,
            assembly_contract_sha256=assembly_contract_sha256,
        )
        roles = contract_binding["roles"]
        return {
            "task_id": task_id,
            "fixed_body_id": self.fixed_body_id,
            "input_body_id": self.input_body_id,
            "output_body_id": self.output_body_id,
            "fixed_rigid_group_id": roles["fixed"]["rigid_group_id"],
            "input_rigid_group_id": roles["input"]["rigid_group_id"],
            "output_rigid_group_id": roles["output"]["rigid_group_id"],
            "fixed_source_solid_ids": roles["fixed"]["source_solid_ids"],
            "input_source_solid_ids": roles["input"]["source_solid_ids"],
            "output_source_solid_ids": roles["output"]["source_solid_ids"],
            "fixed_part_id": roles["fixed"]["part_id"],
            "input_part_id": roles["input"]["part_id"],
            "output_part_id": roles["output"]["part_id"],
            "fixed_occurrence_id": roles["fixed"]["occurrence_id"],
            "input_occurrence_id": roles["input"]["occurrence_id"],
            "output_occurrence_id": roles["output"]["occurrence_id"],
            "input_frame_id": self.input_frame_id,
            "output_frame_id": self.output_frame_id,
            "input_port_id": self.input_port_id,
            "input_observable_id": self.input_observable_id,
            "output_observable_id": self.output_observable_id,
            "actuation_observable_id": self.actuation_observable_id,
            "input_axis_abs": list(self.input_frame.axis_abs),
            "output_axis_abs": list(self.output_frame.axis_abs),
            "assembly_contract_sha256": assembly_contract_sha256,
            "contract_role_binding_sha256": contract_binding[
                "binding_sha256"
            ],
        }

    def to_dict(self) -> dict[str, Any]:
        if self.source_payload_json is not None:
            value = json.loads(self.source_payload_json)
            if not isinstance(value, dict):
                raise AssertionError("stored bicycle-chain payload is not an object")
            return value
        value = asdict(self)
        value.pop("source_payload_json", None)
        value.pop("capability_state", None)
        value.pop("task_id", None)
        value.pop("body_dynamics", None)
        value.pop("sprockets", None)
        value.pop("geometry_gate", None)
        value.pop("task_program", None)
        value.pop("held_out_validation_domain", None)
        value.pop("acceptance_thresholds", None)
        value.pop("required_negative_tests", None)
        value["observables"] = {
            "input_angle": value.pop("input_observable"),
            "output_angle": value.pop("output_observable"),
            "output_actuation_log": value.pop("actuation_observable"),
        }
        value["input_bearing"].pop("spin_damping_n_m_s_rad")
        for field in (
            "return_stiffness_n_m_rad",
            "return_damping_n_m_s_rad",
            "maximum_return_torque_nm",
        ):
            value["output_bearing"].pop(field)
        return value


def _r04_bearing(
    value: Mapping[str, Any],
) -> BicycleBearingSpec:
    translation_stiffness = tuple(value["translation_stiffness_n_m"])
    translation_damping = tuple(value["translation_damping_n_s_m"])
    off_axis_stiffness = tuple(
        value["off_axis_rotation_stiffness_nm_rad"]
    )
    off_axis_damping = tuple(
        value["off_axis_rotation_damping_nm_s_rad"]
    )
    primary_cap = value.get("primary_axis_maximum_abs_torque_nm")
    return BicycleBearingSpec(
        support_id=value["support_id"],
        fixed_body_id=value["fixed_body_id"],
        moving_body_id=value["moving_body_id"],
        frame_id=value["frame_id"],
        # The signed r04 constitutive model is linear: it declares no
        # clearance/dead-zone. This compatibility alias is therefore exact 0.
        radial_clearance_m=0.0,
        radial_stiffness_n_m=float(translation_stiffness[0]),
        axial_stiffness_n_m=float(translation_stiffness[1]),
        translation_damping_n_s_m=float(translation_damping[0]),
        tilt_stiffness_n_m_rad=float(off_axis_stiffness[0]),
        tilt_damping_n_m_s_rad=float(off_axis_damping[0]),
        maximum_force_n=value["maximum_force_n"],
        maximum_torque_nm=value["maximum_total_support_torque_nm"],
        spin_damping_n_m_s_rad=value[
            "primary_axis_damping_nm_s_rad"
        ],
        return_stiffness_n_m_rad=value[
            "primary_axis_centering_stiffness_nm_rad"
        ],
        return_damping_n_m_s_rad=value[
            "primary_axis_damping_nm_s_rad"
        ],
        maximum_return_torque_nm=(
            float(primary_cap) if primary_cap is not None else 0.0
        ),
        constitutive_model=value["constitutive_model"],
        axis_abs=tuple(value["axis_abs"]),
        translation_stiffness_xyz_n_m=translation_stiffness,
        translation_damping_xyz_n_s_m=translation_damping,
        off_axis_rotation_stiffness_xyz_nm_rad=off_axis_stiffness,
        off_axis_rotation_damping_xyz_nm_s_rad=off_axis_damping,
        primary_axis_home_rad=value["primary_axis_home_rad"],
        primary_axis_centering_stiffness_nm_rad=value[
            "primary_axis_centering_stiffness_nm_rad"
        ],
        primary_axis_damping_nm_s_rad=value[
            "primary_axis_damping_nm_s_rad"
        ],
        maximum_total_support_torque_nm=value[
            "maximum_total_support_torque_nm"
        ],
        primary_axis_maximum_abs_torque_nm=(
            float(primary_cap) if primary_cap is not None else None
        ),
        zero_state_stored_energy_j=value["zero_state_stored_energy_j"],
    )


def _normalized_r04_candidate(
    family: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate parameterized identity/placement bindings, then normalize them.

    The implementation family is reusable: legal identifiers and the absolute
    placement of its two parallel axes may differ from the review demonstrator.
    Constitutive values, phase policy, evidence domain, and task semantics do
    not vary.
    """

    normalized = copy.deepcopy(dict(family))
    identifier(family["mechanism_id"], "mechanism_id")
    normalized["mechanism_id"] = expected["mechanism_id"]

    bodies = family["bodies"]
    normalized_bodies = normalized["bodies"]
    expected_bodies = expected["bodies"]
    body_ids: list[str] = []
    group_ids: list[str] = []
    solid_ids: list[str] = []
    for index, (body, expected_body) in enumerate(
        zip(bodies, expected_bodies)
    ):
        label = f"bodies[{index}]"
        body_id = identifier(body["body_id"], f"{label}.body_id")
        group_id = identifier(
            body["rigid_group_id"],
            f"{label}.rigid_group_id",
        )
        solid_id = identifier(
            body["source_solid_ids"][0],
            f"{label}.source_solid_ids[0]",
        )
        center = vector3(
            body["center_of_mass_abs_m"],
            f"{label}.center_of_mass_abs_m",
            allow_zero=True,
        )
        initial = vector3(
            body["initial_position_abs_m"],
            f"{label}.initial_position_abs_m",
            allow_zero=True,
        )
        if body["role"] != "fixed" and center != initial:
            raise _fail(
                "BICYCLE_CHAIN_BODY_ROLE_INVALID",
                f"{label} moving center of mass must equal its initial position",
            )
        body_ids.append(body_id)
        group_ids.append(group_id)
        solid_ids.append(solid_id)
        for key in (
            "body_id",
            "rigid_group_id",
            "source_solid_ids",
            "center_of_mass_abs_m",
            "initial_position_abs_m",
        ):
            normalized_bodies[index][key] = copy.deepcopy(
                expected_body[key]
            )
    if (
        len(set(body_ids)) != 3
        or len(set(group_ids)) != 3
        or len(set(solid_ids)) != 3
    ):
        raise _fail(
            "BICYCLE_CHAIN_BODY_ROLE_INVALID",
            "body, rigid-group, and source-solid IDs must each be unique",
        )
    fixed_body_id, input_body_id, output_body_id = body_ids

    dofs = family["degrees_of_freedom"]
    normalized_dofs = normalized["degrees_of_freedom"]
    expected_dofs = expected["degrees_of_freedom"]
    moving_body_indices = (1, 2)
    frame_origins: list[tuple[float, float, float]] = []
    dof_ids: list[str] = []
    for index, body_index in enumerate(moving_body_indices):
        dof = dofs[index]
        dof_id = identifier(dof["dof_id"], f"degrees_of_freedom[{index}].dof_id")
        origin = vector3(
            dof["origin_abs_m"],
            f"degrees_of_freedom[{index}].origin_abs_m",
            allow_zero=True,
        )
        if (
            dof["body_id"] != body_ids[body_index]
            or origin
            != tuple(bodies[body_index]["initial_position_abs_m"])
        ):
            raise _fail(
                "BICYCLE_CHAIN_BODY_ROLE_INVALID",
                "each revolute coordinate must bind its moving body at that "
                "body's declared initial position",
            )
        dof_ids.append(dof_id)
        frame_origins.append(origin)
        normalized_dofs[index]["dof_id"] = expected_dofs[index]["dof_id"]
        normalized_dofs[index]["body_id"] = expected_dofs[index]["body_id"]
        normalized_dofs[index]["origin_abs_m"] = copy.deepcopy(
            expected_dofs[index]["origin_abs_m"]
        )
    if len(set(dof_ids)) != 2:
        raise _fail(
            "BICYCLE_CHAIN_FRAME_BINDING_INVALID",
            "the input and output DOF IDs must be distinct",
        )
    if (
        frame_origins[0][0] == frame_origins[1][0]
        and frame_origins[0][2] == frame_origins[1][2]
    ):
        raise _fail(
            "BICYCLE_CHAIN_FRAME_BINDING_INVALID",
            "the two parallel revolute axes must be spatially distinct",
        )

    supports = family["finite_bearing_supports"]
    normalized_supports = normalized["finite_bearing_supports"]
    expected_supports = expected["finite_bearing_supports"]
    support_ids: list[str] = []
    frame_ids: list[str] = []
    for index, moving_body_id in enumerate(
        (input_body_id, output_body_id)
    ):
        support = supports[index]
        support_id = identifier(
            support["support_id"],
            f"finite_bearing_supports[{index}].support_id",
        )
        frame_id = identifier(
            support["frame_id"],
            f"finite_bearing_supports[{index}].frame_id",
        )
        if (
            support["fixed_body_id"] != fixed_body_id
            or support["moving_body_id"] != moving_body_id
        ):
            raise _fail(
                "BICYCLE_CHAIN_BODY_ROLE_INVALID",
                "finite bearings must bind the fixed chassis to the correct "
                "input/output moving body",
            )
        support_ids.append(support_id)
        frame_ids.append(frame_id)
        for key in (
            "support_id",
            "fixed_body_id",
            "moving_body_id",
            "frame_id",
        ):
            normalized_supports[index][key] = expected_supports[index][key]
    if len(set(support_ids)) != 2 or len(set(frame_ids)) != 2:
        raise _fail(
            "BICYCLE_CHAIN_FRAME_BINDING_INVALID",
            "support and shaft-frame IDs must be distinct",
        )

    sprockets = family["sprockets"]
    normalized_sprockets = normalized["sprockets"]
    expected_sprockets = expected["sprockets"]
    sprocket_ids: list[str] = []
    pitch_centers: list[tuple[float, float, float]] = []
    for index, body_index in enumerate(moving_body_indices):
        sprocket = sprockets[index]
        sprocket_id = identifier(
            sprocket["sprocket_id"],
            f"sprockets[{index}].sprocket_id",
        )
        center = vector3(
            sprocket["pitch_loop_center_abs_m"],
            f"sprockets[{index}].pitch_loop_center_abs_m",
            allow_zero=True,
        )
        if (
            sprocket["body_id"] != body_ids[body_index]
            or sprocket["shaft_frame_id"] != frame_ids[index]
            or sprocket["step_solid_id"] != solid_ids[body_index]
            or center[0] != frame_origins[index][0]
            or center[2] != frame_origins[index][2]
        ):
            raise _fail(
                "BICYCLE_CHAIN_GEOMETRY_BINDING_INVALID",
                "each pitch torus must bind its moving body, source solid, "
                "shaft frame, and revolute-axis centerline",
            )
        contact_component_ids = sprocket.get(
            "contact_collision_component_ids"
        )
        if (
            not isinstance(contact_component_ids, list)
            or not contact_component_ids
            or len(contact_component_ids)
            > MAX_SPROCKET_CONTACT_COLLISION_COMPONENT_IDS
        ):
            raise _fail(
                "BICYCLE_CHAIN_GEOMETRY_BINDING_INVALID",
                "each sprocket must bind a non-empty bounded list of exact "
                "collision components allowed to contact the moving chain",
            )
        normalized_contact_component_ids = [
            identifier(
                component_id,
                (
                    f"sprockets[{index}]."
                    f"contact_collision_component_ids[{component_index}]"
                ),
            )
            for component_index, component_id in enumerate(
                contact_component_ids
            )
        ]
        if len(set(normalized_contact_component_ids)) != len(
            normalized_contact_component_ids
        ):
            raise _fail(
                "BICYCLE_CHAIN_GEOMETRY_BINDING_INVALID",
                "sprocket contact collision component ids must be unique",
            )
        approved_contact_component_ids = list(
            expected_sprockets[index][
                "contact_collision_component_ids"
            ]
        )
        if normalized_contact_component_ids[
            : len(approved_contact_component_ids)
        ] != approved_contact_component_ids:
            raise _fail(
                "BICYCLE_CHAIN_GEOMETRY_BINDING_INVALID",
                "each sprocket contact list must preserve the exact "
                "implementation-approved component IDs as its prefix",
            )
        sprocket_ids.append(sprocket_id)
        pitch_centers.append(center)
        for key in (
            "sprocket_id",
            "body_id",
            "shaft_frame_id",
            "pitch_loop_center_abs_m",
            "step_solid_id",
            "contact_collision_component_ids",
        ):
            normalized_sprockets[index][key] = copy.deepcopy(
                expected_sprockets[index][key]
            )
    if len(set(sprocket_ids)) != 2 or pitch_centers[0][1] != pitch_centers[1][1]:
        raise _fail(
            "BICYCLE_CHAIN_GEOMETRY_BINDING_INVALID",
            "sprocket IDs must be distinct and pitch torus centers must share "
            "one pitch plane",
        )

    transmission = family["chain_transmission"]
    expected_transmission = expected["chain_transmission"]
    identifier(
        transmission["transmission_id"],
        "chain_transmission.transmission_id",
    )
    gate_id = identifier(
        transmission["geometry_gate_id"],
        "chain_transmission.geometry_gate_id",
    )
    if (
        transmission["driver_body_id"] != input_body_id
        or transmission["driven_body_id"] != output_body_id
    ):
        raise _fail(
            "BICYCLE_CHAIN_BODY_ROLE_INVALID",
            "the finite chain must drive only input crank to rear task target",
        )
    for key in (
        "transmission_id",
        "driver_body_id",
        "driven_body_id",
        "geometry_gate_id",
    ):
        normalized["chain_transmission"][key] = expected_transmission[key]

    gate = family["geometry_gates"][0]
    expected_gate = expected["geometry_gates"][0]
    if gate["gate_id"] != gate_id:
        raise _fail(
            "BICYCLE_CHAIN_GEOMETRY_BINDING_INVALID",
            "the transmission and geometry gate IDs must match",
        )
    normalized["geometry_gates"][0]["gate_id"] = expected_gate["gate_id"]
    for index, body_index in enumerate(moving_body_indices):
        entity = gate["required_step_entities"][index]
        expected_entity = expected_gate["required_step_entities"][index]
        center = vector3(
            entity["center_abs_m"],
            f"geometry_gates[0].required_step_entities[{index}].center_abs_m",
            allow_zero=True,
        )
        if (
            entity["body_id"] != body_ids[body_index]
            or entity["step_solid_id"] != solid_ids[body_index]
            or center != pitch_centers[index]
        ):
            raise _fail(
                "BICYCLE_CHAIN_GEOMETRY_BINDING_INVALID",
                "geometry-gate STEP torus evidence must bind the declared "
                "sprocket body, solid, and pitch center",
            )
        for key in ("body_id", "step_solid_id", "center_abs_m"):
            normalized["geometry_gates"][0]["required_step_entities"][
                index
            ][key] = copy.deepcopy(expected_entity[key])

    port = family["input_ports"][0]
    expected_port = expected["input_ports"][0]
    identifier(port["port_id"], "input_ports[0].port_id")
    if (
        port["target_body_id"] != input_body_id
        or port["frame_id"] != frame_ids[0]
    ):
        raise _fail(
            "BICYCLE_CHAIN_INPUT_PORT_INVALID",
            "the unique task-input port must bind only the crank shaft frame",
        )
    for key in ("port_id", "target_body_id", "frame_id"):
        normalized["input_ports"][0][key] = expected_port[key]

    target = family["unique_task_targets"][0]
    expected_target = expected["unique_task_targets"][0]
    for key in ("pose_observable_id", "actuation_observable_id"):
        identifier(target[key], f"unique_task_targets[0].{key}")
    if target["target_body_id"] != output_body_id:
        raise _fail(
            "BICYCLE_CHAIN_BODY_ROLE_INVALID",
            "rear drive wheel must remain the unique task target",
        )
    for key in (
        "target_body_id",
        "pose_observable_id",
        "actuation_observable_id",
    ):
        normalized["unique_task_targets"][0][key] = expected_target[key]

    domain = family["held_out_validation_domain"]
    expected_domain = expected["held_out_validation_domain"]
    fixed_parameters = domain["fixed_parameters"]
    expected_bindings = (
        f"bodies[{input_body_id}].inertia_diagonal_kg_m2[1]",
        f"bodies[{output_body_id}].inertia_diagonal_kg_m2[1]",
        expected_domain["fixed_parameters"][2]["field_binding"],
        expected_domain["fixed_parameters"][3]["field_binding"],
        f"finite_bearing_supports[{support_ids[0]}]."
        "primary_axis_damping_nm_s_rad",
        f"finite_bearing_supports[{support_ids[1]}]."
        "primary_axis_damping_nm_s_rad",
    )
    for index, binding in enumerate(expected_bindings):
        if fixed_parameters[index]["field_binding"] != binding:
            raise _fail(
                "BICYCLE_CHAIN_HELD_OUT_DOMAIN_INVALID",
                "held-out fixed-parameter bindings must resolve to this "
                "candidate's exact body and finite-support IDs",
            )
        normalized["held_out_validation_domain"]["fixed_parameters"][
            index
        ]["field_binding"] = expected_domain["fixed_parameters"][index][
            "field_binding"
        ]

    allowed = family["wrench_policy"]["allowed"]
    expected_allowed = expected["wrench_policy"]["allowed"]
    expected_targets = (
        input_body_id,
        input_body_id,
        output_body_id,
        input_body_id,
        output_body_id,
    )
    for index, target_body_id in enumerate(expected_targets):
        if allowed[index]["target_body_id"] != target_body_id:
            raise _fail(
                "BICYCLE_CHAIN_WRENCH_POLICY_INVALID",
                "wrench policy target does not preserve crank-only task input "
                "and indirect rear-wheel actuation",
            )
        normalized["wrench_policy"]["allowed"][index][
            "target_body_id"
        ] = expected_allowed[index]["target_body_id"]
    for index in (1, 2):
        if allowed[index]["geometry_gate_id"] != gate_id:
            raise _fail(
                "BICYCLE_CHAIN_WRENCH_POLICY_INVALID",
                "both chain reaction wrenches must use the declared geometry gate",
            )
        normalized["wrench_policy"]["allowed"][index][
            "geometry_gate_id"
        ] = expected_allowed[index]["geometry_gate_id"]
    return normalized


def bicycle_chain_from_dict(
    value: Mapping[str, Any] | None,
) -> BicycleChainSpec:
    """Parse only the exact counter-7, implementation-authorized r04 family.

    Approval permits implementation and candidate registration. It does not
    activate the plugin and it cannot authorize a whole-machine PASS.
    """

    expected = _approved_template_payload()
    _assert_closed_schema_tree(value, expected)
    normalized = _normalized_r04_candidate(value, expected)
    _assert_exact_approved_tree(normalized, expected)
    family = copy.deepcopy(dict(value))

    dynamics = tuple(
        BicycleBodyDynamicsSpec(
            body_id=item["body_id"],
            rigid_group_id=item["rigid_group_id"],
            source_solid_ids=tuple(item["source_solid_ids"]),
            role=item["role"],
            fixed=item["fixed"],
            mass_kg=item["mass_kg"],
            center_of_mass_abs_m=tuple(item["center_of_mass_abs_m"]),
            inertia_diagonal_kg_m2=tuple(
                item["inertia_diagonal_kg_m2"]
            ),
            initial_position_abs_m=tuple(item["initial_position_abs_m"]),
            initial_orientation_wxyz=tuple(
                item["initial_orientation_wxyz"]
            ),
            initial_linear_velocity_abs_m_s=tuple(
                item["initial_linear_velocity_abs_m_s"]
            ),
            initial_angular_velocity_abs_rad_s=tuple(
                item["initial_angular_velocity_abs_rad_s"]
            ),
        )
        for item in family["bodies"]
    )
    bodies = tuple(
        CompoundBodySpec(
            body_id=item.body_id,
            group_ids=(item.rigid_group_id,),
            fixed=item.fixed,
            mass_kg=item.mass_kg,
            center_abs_m=item.center_of_mass_abs_m,
            # Collision materialization consumes authored STEP solids and does
            # not use a proxy size. Zero is an explicit not-applicable sentinel.
            size_m=(0.0, 0.0, 0.0),
            rotation_abs_quat_wxyz=item.initial_orientation_wxyz,
        )
        for item in dynamics
    )
    fixed = next(item for item in dynamics if item.role == "fixed")
    input_body = next(item for item in dynamics if item.role == "input")
    output = next(
        item for item in dynamics if item.role == "unique_task_target"
    )
    input_dof, output_dof = family["degrees_of_freedom"]
    input_support, output_support = family["finite_bearing_supports"]
    input_sprocket_value, output_sprocket_value = family["sprockets"]
    sprockets = tuple(
        BicycleSprocketSpec(
            sprocket_id=item["sprocket_id"],
            body_id=item["body_id"],
            shaft_frame_id=item["shaft_frame_id"],
            pitch_loop_center_abs_m=tuple(
                item["pitch_loop_center_abs_m"]
            ),
            pitch_axis_abs=tuple(item["pitch_axis_abs"]),
            pitch_radius_m=item["pitch_radius_m"],
            toroidal_minor_radius_m=item["toroidal_minor_radius_m"],
            step_solid_id=item["step_solid_id"],
            contact_collision_component_ids=tuple(
                item["contact_collision_component_ids"]
            ),
            approved_contact_collision_component_ids=tuple(
                expected["sprockets"][index][
                    "contact_collision_component_ids"
                ]
            ),
        )
        for index, item in enumerate(family["sprockets"])
    )
    gate_value = family["geometry_gates"][0]
    geometry_gate = BicycleGeometryGateSpec(
        gate_id=gate_value["gate_id"],
        evidence_source=gate_value["evidence_source"],
        required_step_entities=tuple(
            BicycleGeometryEntitySpec(
                body_id=item["body_id"],
                step_solid_id=item["step_solid_id"],
                surface_kind=item["surface_kind"],
                major_radius_m=item["major_radius_m"],
                minor_radius_m=item["minor_radius_m"],
                center_abs_m=tuple(item["center_abs_m"]),
                axis_abs=tuple(item["axis_abs"]),
                surface_exposure_required=item[
                    "surface_exposure_required"
                ],
                minimum_exposed_face_count=item[
                    "minimum_exposed_face_count"
                ],
            )
            for item in gate_value["required_step_entities"]
        ),
        pitch_radius_tolerance_m=gate_value[
            "pitch_radius_tolerance_m"
        ],
        pitch_center_tolerance_m=gate_value[
            "pitch_center_tolerance_m"
        ],
        maximum_axis_parallel_error_rad=gate_value[
            "maximum_axis_parallel_error_rad"
        ],
        maximum_pitch_plane_offset_m=gate_value[
            "maximum_pitch_plane_offset_m"
        ],
        visible_chain_part_id=gate_value["visible_chain_part_id"],
        maximum_chain_sprocket_separation_m=gate_value[
            "maximum_chain_sprocket_separation_m"
        ],
        visible_chain_wrap_band_m=gate_value[
            "visible_chain_wrap_band_m"
        ],
        minimum_visible_chain_wrap_angle_rad=gate_value[
            "minimum_visible_chain_wrap_angle_rad"
        ],
        minimum_visible_chain_wrap_sample_count=gate_value[
            "minimum_visible_chain_wrap_sample_count"
        ],
        zero_transmission_wrench_when_inactive=gate_value[
            "zero_transmission_wrench_when_inactive"
        ],
    )
    port_value = family["input_ports"][0]
    input_frame = BicycleFrameSpec(
        frame_id=input_support["frame_id"],
        body_id=input_body.body_id,
        origin_abs_m=tuple(input_dof["origin_abs_m"]),
        axis_abs=tuple(input_dof["axis_abs"]),
    )
    output_frame = BicycleFrameSpec(
        frame_id=output_support["frame_id"],
        body_id=output.body_id,
        origin_abs_m=tuple(output_dof["origin_abs_m"]),
        axis_abs=tuple(output_dof["axis_abs"]),
    )
    input_port = BicycleInputPortSpec(
        port_id=port_value["port_id"],
        target_body_id=port_value["target_body_id"],
        frame_id=port_value["frame_id"],
        wrench_kind=port_value["wrench_kind"],
        axis_abs=tuple(port_value["direction_abs"]),
        maximum_magnitude_nm=port_value["maximum_magnitude_nm"],
    )
    target = family["unique_task_targets"][0]
    transmission_value = family["chain_transmission"]
    transmission = BicycleChainTransmissionSpec(
        support_id=transmission_value["transmission_id"],
        input_body_id=transmission_value["driver_body_id"],
        output_body_id=transmission_value["driven_body_id"],
        input_frame_id=input_support["frame_id"],
        output_frame_id=output_support["frame_id"],
        input_pitch_radius_m=transmission_value[
            "input_pitch_radius_m"
        ],
        output_pitch_radius_m=transmission_value[
            "output_pitch_radius_m"
        ],
        chain_stiffness_n_m=transmission_value["chain_stiffness_n_m"],
        chain_damping_n_s_m=transmission_value["chain_damping_n_s_m"],
        maximum_differential_tension_n=transmission_value[
            "maximum_abs_chain_force_n"
        ],
        maximum_alignment_error_m=gate_value[
            "maximum_pitch_plane_offset_m"
        ],
        maximum_axis_angle_error_rad=gate_value[
            "maximum_axis_parallel_error_rad"
        ],
        maximum_abs_chain_error_m=transmission_value[
            "maximum_abs_chain_error_m"
        ],
        initial_chain_error_m=transmission_value[
            "initial_chain_error_m"
        ],
        initial_chain_error_rate_m_s=transmission_value[
            "initial_chain_error_rate_m_s"
        ],
        initial_stored_energy_j=transmission_value[
            "initial_stored_energy_j"
        ],
        physical_link_kind=transmission_value["physical_link_kind"],
        ideal_constraint_in_physical_branch=transmission_value[
            "ideal_constraint_in_physical_branch"
        ],
        preload_allowed=transmission_value["preload_allowed"],
        output_feedforward_allowed=transmission_value[
            "output_feedforward_allowed"
        ],
        geometry_gate_id=transmission_value["geometry_gate_id"],
    )
    program_value = family["task_program_policy"]
    task_program = BicycleTaskProgramSpec(
        program_id=program_value["program_id"],
        integration_step_s=program_value["integration_step_s"],
        saved_sample_stride_steps=program_value[
            "saved_sample_stride_steps"
        ],
        maximum_sample_gap_cadences=program_value[
            "maximum_sample_gap_cadences"
        ],
        minimum_powered_sample_count=program_value[
            "minimum_powered_sample_count"
        ],
        phases=tuple(
            BicycleTaskPhaseSpec(
                phase_id=item["phase_id"],
                wrench_kind=item["wrench_kind"],
                direction_abs=(
                    tuple(item["direction_abs"])
                    if item["direction_abs"] is not None
                    else None
                ),
                magnitude_si=item["magnitude_si"],
                duration_s=item["duration_s"],
            )
            for item in program_value["phases"]
        ),
        engagement_phase_ids=tuple(program_value["engagement_phase_ids"]),
        powered_phase_ids=tuple(program_value["powered_phase_ids"]),
        return_phase_ids=tuple(program_value["return_phase_ids"]),
        adaptive_phase_ids=tuple(program_value["adaptive_phase_ids"]),
        return_input_policy=program_value["return_input_policy"],
        engagement_task_motion_credit=program_value[
            "engagement_policy"
        ]["task_motion_credit"],
        engagement_ratio_metric_credit=program_value[
            "engagement_policy"
        ]["ratio_metric_credit"],
    )
    domain_value = family["held_out_validation_domain"]
    varied = domain_value["varied_parameters"][0]
    held_out = BicycleHeldOutValidationDomainSpec(
        case_id=domain_value["case_id"],
        selection_authority=domain_value["selection_authority"],
        minimum_torque_nm=varied["minimum"],
        nominal_torque_nm=varied["nominal"],
        maximum_torque_nm=varied["maximum"],
        required_evidence_torque_nm=tuple(
            item["coupled_positive_crank_task_torque_nm"]
            for item in domain_value[
                "required_evidence_points_after_implementation_before_activation"
            ]
        ),
        fixed_parameters=tuple(
            (item["parameter_id"], item["value"])
            for item in domain_value["fixed_parameters"]
        ),
        no_other_parameter_may_vary=domain_value[
            "no_other_parameter_may_vary"
        ],
        no_hidden_defaults=domain_value["no_hidden_defaults"],
    )
    return BicycleChainSpec(
        schema_version=family["schema_version"],
        mechanism_kind=family["mechanism_kind"],
        mechanism_id=family["mechanism_id"],
        bodies=bodies,
        fixed_body_id=fixed.body_id,
        input_body_id=input_body.body_id,
        output_body_id=output.body_id,
        input_frame=input_frame,
        output_frame=output_frame,
        input_port=input_port,
        input_observable=BicycleObservableSpec(
            observable_id="input_crank_pose_6d",
            body_id=input_body.body_id,
            frame_id=input_frame.frame_id,
            quantity="pose_6d",
        ),
        output_observable=BicycleObservableSpec(
            observable_id=target["pose_observable_id"],
            body_id=output.body_id,
            frame_id=output_frame.frame_id,
            quantity="pose_6d",
        ),
        actuation_observable=BicycleObservableSpec(
            observable_id=target["actuation_observable_id"],
            body_id=output.body_id,
            frame_id=output_frame.frame_id,
            quantity="actuation_log",
        ),
        transmission=transmission,
        input_bearing=_r04_bearing(input_support),
        output_bearing=_r04_bearing(output_support),
        # The signed held-out domain allows only the coupled crank torque to
        # vary; no independent resistive load or contact friction is approved.
        maximum_resistive_test_load_nm=0.0,
        max_support_drift_m=geometry_gate.maximum_pitch_plane_offset_m,
        max_off_axis_rotation_rad=(
            geometry_gate.maximum_axis_parallel_error_rad
        ),
        friction=0.0,
        capability_state=family["capability_state"],
        task_id=target["task_id"],
        body_dynamics=dynamics,
        sprockets=sprockets,
        geometry_gate=geometry_gate,
        task_program=task_program,
        held_out_validation_domain=held_out,
        acceptance_thresholds=BicycleAcceptanceThresholds(
            powered_rotation_min_rad=program_value[
                "powered_measurement_policy"
            ]["minimum_output_rotation_rad"],
            signed_output_per_input=program_value[
                "powered_measurement_policy"
            ]["signed_ratio_target"],
            angular_ratio_tolerance=program_value[
                "powered_measurement_policy"
            ]["signed_ratio_tolerance"],
            return_to_home_max_rad=program_value[
                "return_measurement_policy"
            ]["maximum_rear_home_error_rad"],
        ).validated(),
        required_negative_tests=tuple(family["required_negative_tests"]),
        source_payload_json=json.dumps(
            family,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
    )


def approved_bicycle_chain_family_spec_fixture() -> dict[str, Any]:
    """Return a copyable complete fixture; parsing never calls this helper."""

    return copy.deepcopy(_approved_template_payload())


def _contract_sequence(
    contract: Mapping[str, Any],
    field: str,
) -> list[Mapping[str, Any]]:
    raw = contract.get(field)
    if (
        not isinstance(raw, Sequence)
        or isinstance(raw, (str, bytes))
        or any(not isinstance(item, Mapping) for item in raw)
    ):
        raise _fail(
            "BICYCLE_CHAIN_CONTRACT_BINDING_INVALID",
            f"assembly_contract.{field} must be an object array",
        )
    return list(raw)


def _contract_identifier(value: Any, label: str) -> str:
    try:
        return identifier(value, label)
    except MechanismValidationError as exc:
        raise _fail(
            "BICYCLE_CHAIN_CONTRACT_BINDING_INVALID",
            f"{label} is not a canonical identifier",
        ) from exc


def _contract_frame_axis(
    value: Any,
    *,
    label: str,
) -> tuple[float, float, float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 4
    ):
        raise _fail(
            "BICYCLE_CHAIN_CONTRACT_FRAME_BINDING_INVALID",
            f"{label} must be an xyzw quaternion",
        )
    x_value, y_value, z_value, w_value = (
        finite(item, f"{label}[{index}]")
        for index, item in enumerate(value)
    )
    norm = math.sqrt(
        x_value * x_value
        + y_value * y_value
        + z_value * z_value
        + w_value * w_value
    )
    if abs(norm - 1.0) > 1.0e-6:
        raise _fail(
            "BICYCLE_CHAIN_CONTRACT_FRAME_BINDING_INVALID",
            f"{label} must be normalized",
        )
    # Contract mechanism frames use local +Z as the shaft axis.
    axis = (
        2.0 * (w_value * y_value + x_value * z_value),
        2.0 * (y_value * z_value - w_value * x_value),
        1.0 - 2.0 * (x_value * x_value + y_value * y_value),
    )
    length = math.sqrt(sum(item * item for item in axis))
    if length <= 1.0e-12:
        raise _fail(
            "BICYCLE_CHAIN_CONTRACT_FRAME_BINDING_INVALID",
            f"{label} does not define a shaft axis",
        )
    return tuple(item / length for item in axis)


def resolve_bicycle_contract_bindings(
    spec: BicycleChainSpec,
    contract: Mapping[str, Any],
    *,
    assembly_contract_sha256: str,
) -> dict[str, Any]:
    """Bind typed bodies/frames to independent assembly-contract identities.

    Part and occurrence IDs are deliberately not guessed from body, group or
    STEP-solid names.  They are accepted only after the intake-bound contract
    resolves exactly one physical part for each typed rigid group and exact
    source-solid inventory.
    """

    if not isinstance(contract, Mapping):
        raise _fail(
            "BICYCLE_CHAIN_CONTRACT_BINDING_INVALID",
            "assembly_contract must be an object",
        )
    if (
        not isinstance(assembly_contract_sha256, str)
        or len(assembly_contract_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in assembly_contract_sha256
        )
    ):
        raise _fail(
            "BICYCLE_CHAIN_CONTRACT_BINDING_INVALID",
            "assembly_contract_sha256 must be a lowercase SHA-256 digest",
        )
    groups = _contract_sequence(contract, "rigid_groups")
    parts = _contract_sequence(contract, "parts")
    frames = _contract_sequence(contract, "frames")

    def unique_ids(
        rows: Sequence[Mapping[str, Any]],
        field: str,
    ) -> None:
        values = [
            _contract_identifier(item.get(field), f"{field}[{index}]")
            for index, item in enumerate(rows)
        ]
        if len(values) != len(set(values)):
            raise _fail(
                "BICYCLE_CHAIN_CONTRACT_BINDING_INVALID",
                f"assembly_contract has duplicate {field} values",
            )

    unique_ids(groups, "rigid_group_id")
    unique_ids(parts, "part_id")
    unique_ids(parts, "occurrence_id")
    unique_ids(frames, "frame_id")

    dynamics_by_role = {
        item.role: item
        for item in spec.body_dynamics
    }
    if set(dynamics_by_role) != {"fixed", "input", "unique_task_target"}:
        raise _fail(
            "BICYCLE_CHAIN_CONTRACT_BINDING_INVALID",
            "typed bicycle bodies do not define the three unique roles",
        )
    role_specs = {
        "fixed": dynamics_by_role["fixed"],
        "input": dynamics_by_role["input"],
        "output": dynamics_by_role["unique_task_target"],
    }
    role_bindings: dict[str, dict[str, Any]] = {}
    used_solids: set[str] = set()
    for role, body in role_specs.items():
        matching_groups = [
            item
            for item in groups
            if item.get("rigid_group_id") == body.rigid_group_id
        ]
        if len(matching_groups) != 1:
            raise _fail(
                "BICYCLE_CHAIN_ROLE_GROUP_BINDING_INVALID",
                f"{role} must bind exactly one typed rigid group",
            )
        group = matching_groups[0]
        group_part_ids = group.get("part_ids")
        if (
            not isinstance(group_part_ids, Sequence)
            or isinstance(group_part_ids, (str, bytes))
            or not group_part_ids
            or len(group_part_ids) != len(set(group_part_ids))
            or any(not isinstance(item, str) for item in group_part_ids)
        ):
            raise _fail(
                "BICYCLE_CHAIN_ROLE_GROUP_BINDING_INVALID",
                f"{role} rigid group has an invalid part inventory",
            )
        expected_solids = tuple(body.source_solid_ids)
        expected_contract_role = "fixed" if body.fixed else "dynamic"
        matching_parts = []
        for item in parts:
            raw_solids = item.get("source_solid_ids")
            actual_solids = (
                tuple(str(value) for value in raw_solids)
                if isinstance(raw_solids, Sequence)
                and not isinstance(raw_solids, (str, bytes))
                else ()
            )
            if (
                item.get("rigid_group_id") == body.rigid_group_id
                and item.get("role") == expected_contract_role
                and actual_solids == expected_solids
            ):
                matching_parts.append(item)
        if len(matching_parts) != 1:
            raise _fail(
                "BICYCLE_CHAIN_ROLE_PART_BINDING_INVALID",
                f"{role} must resolve exactly one non-context contract part "
                "from its typed group and source solids",
            )
        part = matching_parts[0]
        if (
            part.get("part_id") not in group_part_ids
            or tuple(group_part_ids) != (part.get("part_id"),)
        ):
            raise _fail(
                "BICYCLE_CHAIN_ROLE_PART_BINDING_INVALID",
                f"{role} rigid group must own exactly its one typed part",
            )
        part_id = _contract_identifier(
            part.get("part_id"),
            f"{role}.part_id",
        )
        occurrence_id = _contract_identifier(
            part.get("occurrence_id"),
            f"{role}.occurrence_id",
        )
        if used_solids.intersection(expected_solids):
            raise _fail(
                "BICYCLE_CHAIN_ROLE_SOLID_BINDING_OVERLAP",
                "typed bicycle roles cannot share source STEP solids",
            )
        used_solids.update(expected_solids)
        role_bindings[role] = {
            "body_id": body.body_id,
            "rigid_group_id": body.rigid_group_id,
            "part_id": part_id,
            "occurrence_id": occurrence_id,
            "contract_role": expected_contract_role,
            "source_solid_ids": list(expected_solids),
        }

    frame_bindings: dict[str, dict[str, Any]] = {}
    for role, frame_spec in (
        ("input", spec.input_frame),
        ("output", spec.output_frame),
    ):
        matches = [
            item
            for item in frames
            if item.get("frame_id") == frame_spec.frame_id
        ]
        if len(matches) != 1:
            raise _fail(
                "BICYCLE_CHAIN_CONTRACT_FRAME_BINDING_INVALID",
                f"{role} must bind exactly one assembly-contract shaft frame",
            )
        frame = matches[0]
        origin = vector3(
            frame.get("origin_m"),
            f"{role}.frame.origin_m",
            allow_zero=True,
        )
        axis = _contract_frame_axis(
            frame.get("orientation_xyzw"),
            label=f"{role}.frame.orientation_xyzw",
        )
        if (
            frame.get("rigid_group_id")
            != role_bindings[role]["rigid_group_id"]
            or max(
                abs(actual - expected)
                for actual, expected in zip(
                    origin,
                    frame_spec.origin_abs_m,
                )
            )
            > 1.0e-9
            or max(
                abs(actual - expected)
                for actual, expected in zip(
                    axis,
                    frame_spec.axis_abs,
                )
            )
            > 1.0e-6
        ):
            raise _fail(
                "BICYCLE_CHAIN_CONTRACT_FRAME_BINDING_INVALID",
                f"{role} shaft frame group, center or axis changed",
            )
        frame_bindings[role] = {
            "frame_id": frame_spec.frame_id,
            "rigid_group_id": role_bindings[role]["rigid_group_id"],
            "origin_abs_m": list(origin),
            "axis_abs": list(axis),
        }

    result = {
        "schema_version": "bicycle_chain_contract_role_binding_v1",
        "assembly_contract_sha256": assembly_contract_sha256,
        "roles": role_bindings,
        "frames": frame_bindings,
    }
    result["binding_sha256"] = canonical_sha256(result)
    return result


def chain_engagement_active(
    spec: BicycleChainSpec,
    *,
    bindings_match: bool,
    input_pitch_radius_m: float,
    output_pitch_radius_m: float,
    axis_angle_error_rad: float,
    axial_alignment_error_m: float,
) -> bool:
    values = (
        input_pitch_radius_m,
        output_pitch_radius_m,
        axis_angle_error_rad,
        axial_alignment_error_m,
    )
    if any(not math.isfinite(float(item)) for item in values):
        return False
    return bool(
        bindings_match
        and abs(input_pitch_radius_m - spec.input_pitch_radius_m) <= 1.0e-9
        and abs(output_pitch_radius_m - spec.output_pitch_radius_m) <= 1.0e-9
        and 0.0 <= axis_angle_error_rad
        <= spec.maximum_axis_angle_error_rad
        and 0.0 <= axial_alignment_error_m
        <= spec.maximum_alignment_error_m
    )


def chain_reaction_torques(
    spec: BicycleChainSpec,
    *,
    coordinate_error_m: float,
    coordinate_rate_m_s: float,
    gate_active: bool,
) -> tuple[float, float, float]:
    error = finite(coordinate_error_m, "coordinate_error_m")
    rate = finite(coordinate_rate_m_s, "coordinate_rate_m_s")
    if not gate_active:
        return (0.0, 0.0, 0.0)
    tension = _clamp(
        spec.chain_stiffness_n_m * error
        + spec.chain_damping_n_s_m * rate,
        spec.maximum_differential_tension_n,
    )
    return (
        tension,
        -spec.input_pitch_radius_m * tension,
        spec.output_pitch_radius_m * tension,
    )


def crank_centering_torque(
    spec: BicycleChainSpec,
    *,
    angle_from_home_rad: float,
    angular_speed_rad_s: float,
    gate_active: bool,
) -> float:
    displacement = finite(angle_from_home_rad, "angle_from_home_rad")
    speed = finite(angular_speed_rad_s, "angular_speed_rad_s")
    if not gate_active:
        return 0.0
    raw = (
        -spec.input_bearing.return_stiffness_n_m_rad * displacement
        - spec.input_bearing.return_damping_n_m_s_rad * speed
    )
    return _clamp(raw, spec.input_bearing.maximum_return_torque_nm)


def validate_runtime_bindings(
    spec: BicycleChainSpec,
    value: Mapping[str, Any],
    *,
    task_id: str,
    assembly_contract: Mapping[str, Any],
    assembly_contract_sha256: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _fail(
            "BICYCLE_CHAIN_BINDING_INVALID",
            "runtime binding must be an object",
        )
    expected = spec.expected_runtime_bindings(
        task_id=task_id,
        assembly_contract=assembly_contract,
        assembly_contract_sha256=assembly_contract_sha256,
    )
    if set(value) != set(expected):
        raise _fail(
            "BICYCLE_CHAIN_BINDING_INVALID",
            "runtime binding schema is open or incomplete",
        )
    normalized = dict(value)
    for field in ("input_axis_abs", "output_axis_abs"):
        normalized[field] = list(
            vector3(normalized[field], field, unit=True)
        )
    if normalized != expected:
        raise _fail(
            "BICYCLE_CHAIN_BINDING_INVALID",
            "runtime body, group, solid, part, occurrence, frame, port, "
            "observable, contract hash, or axis changed",
        )
    if normalized["input_body_id"] == normalized["output_body_id"]:
        raise _fail(
            "BICYCLE_CHAIN_BINDING_INVALID",
            "input and output observables must bind distinct bodies",
        )
    return normalized


def validate_task_input_target(
    spec: BicycleChainSpec,
    *,
    target_body_id: str,
    phase_id: str,
    torque_nm: float,
) -> None:
    torque = finite(torque_nm, "torque_nm")
    if target_body_id != spec.input_body_id:
        raise _fail(
            "BICYCLE_CHAIN_DIRECT_OUTPUT_DRIVE",
            "validator task input may target only the input crank",
        )
    if abs(torque) > spec.maximum_task_input_torque_nm:
        raise _fail(
            "BICYCLE_CHAIN_TASK_INPUT_BOUND_EXCEEDED",
            "validator task input exceeds the approved finite bound",
        )
    if spec.task_program is None:
        raise _fail(
            "BICYCLE_CHAIN_TASK_PROGRAM_MISSING",
            "r04 task-input validation requires the typed task program",
        )
    driven_phases = (
        set(spec.task_program.engagement_phase_ids)
        | set(spec.task_program.powered_phase_ids)
    )
    return_phases = set(spec.task_program.return_phase_ids)
    if phase_id in return_phases and torque != 0.0:
        raise _fail(
            "BICYCLE_CHAIN_RETURN_TASK_INPUT_NONZERO",
            "task input must be exact numeric zero throughout both return phases",
        )
    if (
        phase_id not in driven_phases
        and phase_id not in return_phases
        and phase_id != "zero_state_no_input_drift"
        and torque != 0.0
    ):
        raise _fail(
            "BICYCLE_CHAIN_TASK_INPUT_PHASE_INVALID",
            "nonzero task input is allowed only in the two declared non-return "
            "phases or the dedicated crank-only adversarial program",
        )


_TRACE_SAMPLE_KEYS = {
    "time_s",
    "phase_id",
    "input_crank_angle_rad",
    "rear_drive_wheel_angle_rad",
    "task_input_torque_nm",
}
_ROUNDTRIP_PHASES = (
    "finite_engagement_settle",
    "measured_powered_drive",
    "passive_return",
    "return_settle",
)


def _validated_trace_samples(
    samples: Sequence[Mapping[str, Any]],
    *,
    maximum_sample_gap_s: float,
) -> list[dict[str, Any]]:
    if (
        not isinstance(samples, Sequence)
        or isinstance(samples, (str, bytes))
        or len(samples) < 4
    ):
        raise _fail(
            "BICYCLE_CHAIN_TRACE_INCONCLUSIVE",
            "trace needs at least four full-cadence samples",
        )
    maximum_gap = _positive(maximum_sample_gap_s, "maximum_sample_gap_s")
    result: list[dict[str, Any]] = []
    previous_time: float | None = None
    for index, raw in enumerate(samples):
        if not isinstance(raw, Mapping) or set(raw) != _TRACE_SAMPLE_KEYS:
            raise _fail(
                "BICYCLE_CHAIN_TRACE_INCONCLUSIVE",
                f"trace sample {index} is open or incomplete",
            )
        sample = dict(raw)
        time_s = finite(sample["time_s"], f"samples[{index}].time_s")
        for field in (
            "input_crank_angle_rad",
            "rear_drive_wheel_angle_rad",
            "task_input_torque_nm",
        ):
            sample[field] = finite(
                sample[field],
                f"samples[{index}].{field}",
            )
        if not isinstance(sample["phase_id"], str) or not sample["phase_id"]:
            raise _fail(
                "BICYCLE_CHAIN_TRACE_INCONCLUSIVE",
                f"trace sample {index} has no phase",
            )
        if previous_time is not None and (
            time_s <= previous_time
            or time_s - previous_time > maximum_gap + 1.0e-12
        ):
            raise _fail(
                "BICYCLE_CHAIN_TRACE_INCONCLUSIVE",
                "trace timestamps must be unique, increasing and full-cadence",
            )
        sample["time_s"] = time_s
        previous_time = time_s
        result.append(sample)
    return result


def validate_roundtrip_phase_coverage(
    samples: Sequence[Mapping[str, Any]],
    *,
    spec: BicycleChainSpec | None = None,
    maximum_sample_gap_s: float | None = None,
) -> None:
    compressed: list[str] = []
    for sample in samples:
        phase = str(sample["phase_id"])
        if not compressed or compressed[-1] != phase:
            compressed.append(phase)
    if tuple(compressed) != _ROUNDTRIP_PHASES:
        raise _fail(
            "BICYCLE_CHAIN_PHASE_COVERAGE_INVALID",
            "the fixed engagement, measured-power, passive-return and "
            "return-settle boundaries are required exactly once and in order",
        )
    if spec is None:
        return
    if spec.task_program is None:
        raise _fail(
            "BICYCLE_CHAIN_TASK_PROGRAM_MISSING",
            "phase coverage requires the typed r04 task program",
        )
    gap = (
        spec.task_program.integration_step_s
        * spec.task_program.maximum_sample_gap_cadences
        if maximum_sample_gap_s is None
        else _positive(maximum_sample_gap_s, "maximum_sample_gap_s")
    )
    for phase in spec.task_program.phases:
        phase_samples = [
            sample for sample in samples
            if sample["phase_id"] == phase.phase_id
        ]
        observed_span = (
            float(phase_samples[-1]["time_s"])
            - float(phase_samples[0]["time_s"])
        )
        # Runtime timestamps are accumulated across as many as 48,800
        # integration steps.  Use the same nanosecond-scale boundary
        # roundoff already required by the physical trace comparison; this is
        # still orders of magnitude below one approved sampling cadence.
        if abs(observed_span - phase.duration_s) > gap + 1.0e-9:
            raise _fail(
                "BICYCLE_CHAIN_PHASE_COVERAGE_INVALID",
                f"{phase.phase_id} must cover its fixed {phase.duration_s} s "
                "duration at full cadence",
            )


def validate_powered_ratio(
    samples: Sequence[Mapping[str, Any]],
    thresholds: BicycleAcceptanceThresholds,
) -> None:
    contract = thresholds.validated()
    engagement = [
        sample
        for sample in samples
        if sample["phase_id"] == "finite_engagement_settle"
    ]
    if not engagement:
        raise _fail(
            "BICYCLE_CHAIN_PHASE_COVERAGE_INVALID",
            "the fully traced engagement phase and boundary baseline are required",
        )
    baseline = engagement[-1]
    home_input = float(baseline["input_crank_angle_rad"])
    home_output = float(baseline["rear_drive_wheel_angle_rad"])
    powered = [
        sample
        for sample in samples
        if sample["phase_id"] == "measured_powered_drive"
    ]
    if not powered:
        raise _fail(
            "BICYCLE_CHAIN_PHASE_COVERAGE_INVALID",
            "powered phase has no samples",
        )
    for index, sample in enumerate(powered):
        input_delta = float(sample["input_crank_angle_rad"]) - home_input
        output_delta = (
            float(sample["rear_drive_wheel_angle_rad"]) - home_output
        )
        if abs(input_delta) <= 1.0e-12:
            valid = (
                abs(
                    output_delta
                    - contract.signed_output_per_input * input_delta
                )
                <= 1.0e-12
            )
        else:
            valid = (
                abs(
                    output_delta / input_delta
                    - contract.signed_output_per_input
                )
                <= contract.angular_ratio_tolerance
            )
        if not valid:
            raise _fail(
                "BICYCLE_CHAIN_SIGNED_RATIO_INVALID",
                f"powered sample {index} violates the signed all-sample ratio",
            )
    adjacent = [baseline, *powered]
    for index, (first, second) in enumerate(
        zip(adjacent, adjacent[1:])
    ):
        input_delta = (
            float(second["input_crank_angle_rad"])
            - float(first["input_crank_angle_rad"])
        )
        output_delta = (
            float(second["rear_drive_wheel_angle_rad"])
            - float(first["rear_drive_wheel_angle_rad"])
        )
        if abs(input_delta) <= 1.0e-12:
            valid = abs(output_delta) <= 1.0e-12
        else:
            valid = (
                abs(
                    output_delta / input_delta
                    - contract.signed_output_per_input
                )
                <= contract.angular_ratio_tolerance
            )
        if not valid:
            raise _fail(
                "BICYCLE_CHAIN_SIGNED_RATIO_INVALID",
                f"powered increment {index} violates the signed full-cadence ratio",
            )


def evaluate_bicycle_roundtrip_trace(
    spec: BicycleChainSpec,
    samples: Sequence[Mapping[str, Any]],
    thresholds: BicycleAcceptanceThresholds,
    *,
    maximum_sample_gap_s: float | None = None,
) -> dict[str, float]:
    contract = thresholds.validated()
    if spec.task_program is None:
        raise _fail(
            "BICYCLE_CHAIN_TASK_PROGRAM_MISSING",
            "roundtrip evaluation requires the typed r04 task program",
        )
    frozen = spec.acceptance_thresholds
    if (
        frozen is None
        or contract.powered_rotation_min_rad
        != frozen.powered_rotation_min_rad
        or contract.signed_output_per_input
        != frozen.signed_output_per_input
        or contract.angular_ratio_tolerance
        != frozen.angular_ratio_tolerance
        or contract.return_to_home_max_rad
        != frozen.return_to_home_max_rad
    ):
        raise _fail(
            "BICYCLE_CHAIN_THRESHOLD_TAMPER",
            "task thresholds must come unchanged from the frozen r04 requirements",
        )
    approved_gap = (
        spec.task_program.integration_step_s
        * spec.task_program.maximum_sample_gap_cadences
    )
    gap = approved_gap if maximum_sample_gap_s is None else _positive(
        maximum_sample_gap_s,
        "maximum_sample_gap_s",
    )
    if gap > approved_gap + 1.0e-12:
        raise _fail(
            "BICYCLE_CHAIN_TRACE_INCONCLUSIVE",
            "runtime sampling cannot be sparser than the approved full cadence",
        )
    trace = _validated_trace_samples(
        samples,
        maximum_sample_gap_s=gap,
    )
    validate_roundtrip_phase_coverage(
        trace,
        spec=spec,
        maximum_sample_gap_s=gap,
    )
    validate_powered_ratio(trace, contract)
    for sample in trace:
        validate_task_input_target(
            spec,
            target_body_id=spec.input_body_id,
            phase_id=str(sample["phase_id"]),
            torque_nm=float(sample["task_input_torque_nm"]),
        )
    domain = spec.held_out_validation_domain
    if domain is None:
        raise _fail(
            "BICYCLE_CHAIN_HELD_OUT_DOMAIN_MISSING",
            "roundtrip evaluation requires the typed r04 held-out domain",
        )
    coupled_torques = {
        float(sample["task_input_torque_nm"])
        for sample in trace
        if sample["phase_id"]
        in {"finite_engagement_settle", "measured_powered_drive"}
    }
    if (
        len(coupled_torques) != 1
        or next(iter(coupled_torques)) < domain.minimum_torque_nm
        or next(iter(coupled_torques)) > domain.maximum_torque_nm
    ):
        raise _fail(
            "BICYCLE_CHAIN_HELD_OUT_DOMAIN_INVALID",
            "one coupled positive crank torque in the closed 3.18/3.22 N*m "
            "interval must be used identically in both non-return phases",
        )
    engagement_boundary = next(
        sample
        for sample in reversed(trace)
        if sample["phase_id"] == "finite_engagement_settle"
    )
    powered_baseline = float(
        engagement_boundary["rear_drive_wheel_angle_rad"]
    )
    powered_excursion = max(
        float(sample["rear_drive_wheel_angle_rad"])
        - powered_baseline
        for sample in trace
        if sample["phase_id"] == "measured_powered_drive"
    )
    if powered_excursion < contract.powered_rotation_min_rad:
        raise _fail(
            "BICYCLE_CHAIN_POWERED_ROTATION_INSUFFICIENT",
            "only positive signed motion inside the powered window can be "
            "credited as powered travel",
        )
    return_error = abs(
        float(trace[-1]["rear_drive_wheel_angle_rad"])
        - float(trace[0]["rear_drive_wheel_angle_rad"])
    )
    if return_error > contract.return_to_home_max_rad:
        raise _fail(
            "BICYCLE_CHAIN_RETURN_TO_HOME_INVALID",
            "rear wheel did not return within the frozen home tolerance",
        )
    return {
        "continuous_rotation_rad": powered_excursion,
        "angular_ratio": contract.signed_output_per_input,
        "return_to_start_rad": return_error,
    }


def validate_zero_input_trace(
    samples: Sequence[Mapping[str, Any]],
    *,
    maximum_sample_gap_s: float = 0.000250001,
    drift_tolerance_rad: float = 1.0e-8,
) -> None:
    trace = _validated_trace_samples(
        samples,
        maximum_sample_gap_s=maximum_sample_gap_s,
    )
    home_input = float(trace[0]["input_crank_angle_rad"])
    home_output = float(trace[0]["rear_drive_wheel_angle_rad"])
    for sample in trace:
        if (
            sample["task_input_torque_nm"] != 0.0
            or abs(float(sample["input_crank_angle_rad"]) - home_input)
            > drift_tolerance_rad
            or abs(float(sample["rear_drive_wheel_angle_rad"]) - home_output)
            > drift_tolerance_rad
        ):
            raise _fail(
                "BICYCLE_CHAIN_ZERO_INPUT_ENERGY_CREATED",
                "zero-input home trace created motion or external energy",
            )


_WRENCH_RECORD_KEYS = {
    "time_s",
    "phase_id",
    "wrench_id",
    "owner",
    "category",
    "target_body_id",
    "torque_nm",
    "axis_abs",
    "geometry_gate_id",
    "gate_active",
    "plugin_id",
    "plugin_instance_id",
    "chain_state_id",
}
_CHAIN_INPUT_WRENCH_ID = "finite_chain_input_torque"
_CHAIN_OUTPUT_WRENCH_ID = "finite_chain_output_torque"


def validate_wrench_provenance(
    spec: BicycleChainSpec,
    records: Sequence[Mapping[str, Any]],
    *,
    direct_actuation_flag: bool | None = None,
) -> None:
    del direct_actuation_flag
    if (
        not isinstance(records, Sequence)
        or isinstance(records, (str, bytes))
        or not records
    ):
        raise _fail(
            "BICYCLE_CHAIN_WRENCH_PROVENANCE_INVALID",
            "a boolean claim cannot replace the complete wrench ledger",
        )
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(records):
        if not isinstance(raw, Mapping) or set(raw) != _WRENCH_RECORD_KEYS:
            raise _fail(
                "BICYCLE_CHAIN_WRENCH_PROVENANCE_INVALID",
                f"wrench record {index} is open or incomplete",
            )
        record = dict(raw)
        record["time_s"] = finite(
            record["time_s"],
            f"records[{index}].time_s",
        )
        record["torque_nm"] = finite(
            record["torque_nm"],
            f"records[{index}].torque_nm",
        )
        record["axis_abs"] = list(
            vector3(record["axis_abs"], f"records[{index}].axis_abs", unit=True)
        )
        if record["axis_abs"] != list(spec.axis_abs):
            raise _fail(
                "BICYCLE_CHAIN_WRENCH_PROVENANCE_INVALID",
                "wrench axis differs from the frozen input axis",
            )
        if record["category"] == "task_input":
            validate_task_input_target(
                spec,
                target_body_id=str(record["target_body_id"]),
                phase_id=str(record["phase_id"]),
                torque_nm=float(record["torque_nm"]),
            )
        if not bool(record["gate_active"]) and record["torque_nm"] != 0.0:
            raise _fail(
                "BICYCLE_CHAIN_GATE_OFF_WRENCH_NONZERO",
                "inactive geometry gates require bitwise-zero wrench",
            )
        normalized.append(record)
    chain_by_time: dict[float, list[dict[str, Any]]] = {}
    for record in normalized:
        if record["wrench_id"] in {
            _CHAIN_INPUT_WRENCH_ID,
            _CHAIN_OUTPUT_WRENCH_ID,
        }:
            chain_by_time.setdefault(float(record["time_s"]), []).append(record)
    for time_s, pair in chain_by_time.items():
        if (
            len(pair) != 2
            or {item["wrench_id"] for item in pair}
            != {_CHAIN_INPUT_WRENCH_ID, _CHAIN_OUTPUT_WRENCH_ID}
        ):
            raise _fail(
                "BICYCLE_CHAIN_WRENCH_CONTEXT_INVALID",
                f"chain wrench pair is incomplete at {time_s}",
            )
        contexts = {
            (
                item["plugin_id"],
                item["plugin_instance_id"],
                item["chain_state_id"],
                item["geometry_gate_id"],
                item["gate_active"],
                item["phase_id"],
            )
            for item in pair
        }
        if (
            len(contexts) != 1
            or pair[0]["plugin_id"] != BICYCLE_CHAIN_PLUGIN_ID
            or pair[0]["owner"] != "registered_plugin"
            or pair[1]["owner"] != "registered_plugin"
            or pair[0]["category"] != "transmission"
            or pair[1]["category"] != "transmission"
        ):
            raise _fail(
                "BICYCLE_CHAIN_WRENCH_CONTEXT_INVALID",
                "paired chain reactions do not share one registered context",
            )


def task_success_authorized(
    *,
    metrics_pass: bool,
    physical_audit_pass: bool,
) -> bool:
    if metrics_pass and not physical_audit_pass:
        raise _fail(
            "BICYCLE_CHAIN_PHYSICAL_AUDIT_FAILED",
            "favorable metrics cannot override a failed physical audit",
        )
    return bool(metrics_pass and physical_audit_pass)
