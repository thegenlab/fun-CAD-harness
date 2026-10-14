from __future__ import annotations

"""Validator-owned motion-program, sampling, and channel evidence.

Task acceptance must not infer a powered stroke from whichever sample happens
to contain the largest coordinate.  This module turns the immutable
``LoadCase`` into explicit powered/return windows, validates that a completed
trace actually covers that program, and resolves semantic task roles onto the
runtime bodies and canonical trace channels owned by the active mechanism
adapter.
"""

import math
import sys
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from ..contracts.failure_envelope import (
    EvidenceValidity,
    FailureOwner,
    make_failure_envelope,
    plugin_owner_for_registration,
)
from .io import canonical_sha256
from .models import LoadCase, MechanismPlan, MechanismValidationError
from .plugin_boundary import invoke_plugin_callback, plugin_boundary_error


TASK_PROGRAM_EVIDENCE_SCHEMA = "task_program_evidence_v1"
TASK_CHANNEL_REGISTRY_SCHEMA = "task_channel_registry_v1"
VALIDATED_TRACE_PROGRAM_SCHEMA = "validated_trace_program_v1"
TASK_WRENCH_COMPONENTS_SCHEMA = "task_wrench_components_v1"

MOTION_WINDOW_METRICS = frozenset(
    {
        "continuous_rotation_rad",
        "return_to_start_rad",
        "output_torque_nm",
        "signed_displacement_m",
        "cumulative_travel_m",
        "angular_ratio",
        "displacement_ratio",
        "rolling_slip_ratio",
        "powered_contact_coverage_ratio",
        "ground_separation_m",
        "clock.display_direction_consistent",
        "clock.maximum_hand_drift_rad",
        "clock.hour_zero_error_rad",
        "clock.minute_zero_error_rad",
        "clock.second_zero_error_rad",
        "clock.maximum_release_drift_rad",
        "egg_beater_outputs_opposite_directions",
        "egg_beater_output_speed_magnitude_ratio",
        "egg_beater_crank_speed_magnitude_ratio",
        "egg_beater_axis_bound_direction_consistency",
        "utility_knife_reverse_load_magnitude",
        "utility_knife_loaded_hold_duration",
        "utility_knife_maximum_loaded_slip",
        "tongs.hold_duration_s",
        "tongs.release_input_zero",
        "gate_zero_input_hold",
        "gate_zero_input_hold_duration_s",
        "gate_maximum_downward_slip_m",
    }
)
RATIO_METRICS = frozenset({"angular_ratio", "displacement_ratio"})


@dataclass(frozen=True)
class TaskProgramEvidenceSpec:
    """Exact validator-selected portions of one immutable load program."""

    program: str
    powered_phase_ids: tuple[str, ...]
    return_phase_ids: tuple[str, ...]
    return_input_policy: str
    adaptive_phase_ids: tuple[str, ...]
    minimum_powered_sample_count: int
    maximum_sample_gap_cadences: float
    executed_phase_extension_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.program:
            raise MechanismValidationError(
                "TASK_PROGRAM_MISSING",
                "task motion evidence requires a validator load program",
            )
        if not self.powered_phase_ids:
            raise MechanismValidationError(
                "TASK_POWERED_PHASES_MISSING",
                "task motion evidence requires at least one powered phase",
            )
        if set(self.powered_phase_ids) & set(self.return_phase_ids):
            raise MechanismValidationError(
                "TASK_PROGRAM_WINDOW_OVERLAP",
                "powered and return phase windows cannot overlap",
            )
        if self.return_input_policy not in {"zero", "selected_input_only"}:
            raise MechanismValidationError(
                "TASK_RETURN_INPUT_POLICY_INVALID",
                "return input policy must be zero or selected_input_only",
            )
        if (
            isinstance(self.minimum_powered_sample_count, bool)
            or self.minimum_powered_sample_count < 3
        ):
            raise MechanismValidationError(
                "TASK_POWERED_SAMPLE_POLICY_INVALID",
                "task programs require at least three powered samples",
            )
        if (
            not math.isfinite(self.maximum_sample_gap_cadences)
            or self.maximum_sample_gap_cadences <= 0.0
            or self.maximum_sample_gap_cadences > 1.0
        ):
            raise MechanismValidationError(
                "TASK_SAMPLE_CADENCE_POLICY_INVALID",
                "task program cadence threshold must be in (0, 1]",
            )
        if not set(self.adaptive_phase_ids).issubset(
            set(self.powered_phase_ids) | set(self.return_phase_ids)
        ):
            raise MechanismValidationError(
                "TASK_ADAPTIVE_PHASE_POLICY_INVALID",
                "adaptive phases must belong to a registered task window",
            )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = TASK_PROGRAM_EVIDENCE_SCHEMA
        payload["powered_phase_ids"] = list(self.powered_phase_ids)
        payload["return_phase_ids"] = list(self.return_phase_ids)
        payload["adaptive_phase_ids"] = list(self.adaptive_phase_ids)
        payload["executed_phase_extension_keys"] = list(
            self.executed_phase_extension_keys
        )
        payload["program_evidence_sha256"] = canonical_sha256(payload)
        return payload


@dataclass(frozen=True)
class ResolvedTaskChannel:
    semantic_id: str
    rigid_group_id: str
    body_id: str
    frame_id: str
    canonical_field: str | None
    axis_abs: tuple[float, float, float] | None = None
    wrench_type: str | None = None
    quantity: str | None = None
    actuation_body_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.axis_abs is not None:
            payload["axis_abs"] = list(self.axis_abs)
        payload["actuation_body_ids"] = list(self.actuation_body_ids)
        return payload


@dataclass(frozen=True)
class TaskChannelRegistry:
    """Runtime body/channel mapping derived only from trusted adapter code."""

    plugin_id: str
    input_body_ids: tuple[str, ...]
    output_body_ids: tuple[str, ...]
    inputs: Mapping[str, ResolvedTaskChannel]
    observables: Mapping[str, ResolvedTaskChannel]

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": TASK_CHANNEL_REGISTRY_SCHEMA,
            "plugin_id": self.plugin_id,
            "input_body_ids": list(self.input_body_ids),
            "output_body_ids": list(self.output_body_ids),
            "inputs": {
                key: value.to_dict()
                for key, value in sorted(self.inputs.items())
            },
            "observables": {
                key: value.to_dict()
                for key, value in sorted(self.observables.items())
            },
        }
        payload["channel_registry_sha256"] = canonical_sha256(payload)
        return payload

    def resolve(
        self,
        raw: Mapping[str, Any],
        *,
        role: str,
    ) -> ResolvedTaskChannel:
        id_field = "port_id" if role == "input" else "observable_id"
        semantic_id = raw.get(id_field)
        if not isinstance(semantic_id, str) or not semantic_id:
            raise MechanismValidationError(
                "TASK_CHANNEL_BINDING_INCOMPLETE",
                f"{role} binding lacks its exact semantic ID",
            )
        inventory = self.inputs if role == "input" else self.observables
        registered = inventory.get(semantic_id)
        if registered is None:
            raise MechanismValidationError(
                "TASK_CHANNEL_ID_UNREGISTERED",
                f"{role} channel {semantic_id!r} is not registered by the adapter",
            )
        expected = {
            "rigid_group_id": registered.rigid_group_id,
            "frame_id": registered.frame_id,
            (
                "wrench_type" if role == "input" else "quantity"
            ): (
                registered.wrench_type
                if role == "input"
                else registered.quantity
            ),
        }
        for field, expected_value in expected.items():
            if raw.get(field) != expected_value:
                raise MechanismValidationError(
                    "TASK_CHANNEL_REGISTRY_MISMATCH",
                    f"{role} channel {semantic_id!r} changes {field}",
                )
        if role == "input":
            raw_axis = raw.get("direction_abs")
            if (
                registered.axis_abs is None
                or not isinstance(raw_axis, (list, tuple))
                or len(raw_axis) != 3
                or any(
                    isinstance(actual, bool)
                    or not isinstance(actual, (int, float))
                    or not math.isfinite(float(actual))
                    for actual in raw_axis
                )
                or any(
                    not math.isfinite(float(expected_value))
                    for expected_value in registered.axis_abs
                )
                or any(
                    abs(float(actual) - expected_value) > 1.0e-9
                    for actual, expected_value in zip(
                        raw_axis, registered.axis_abs
                    )
                )
            ):
                raise MechanismValidationError(
                    "TASK_CHANNEL_AXIS_MISMATCH",
                    f"input channel {semantic_id!r} changes its registered axis",
                )
        return registered


@dataclass(frozen=True)
class ValidatedTraceProgram:
    """Indices and schedule produced after strict trace-program validation."""

    program_spec: TaskProgramEvidenceSpec
    powered_sample_indices: tuple[int, ...]
    return_sample_indices: tuple[int, ...]
    powered_baseline_index: int
    home_sample_index: int
    final_sample_index: int
    executed_phase_ids: tuple[str, ...]
    executed_load_sha256: str
    max_sample_gap_s: float

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": VALIDATED_TRACE_PROGRAM_SCHEMA,
            "program_spec": self.program_spec.to_dict(),
            "powered_sample_indices": list(self.powered_sample_indices),
            "return_sample_indices": list(self.return_sample_indices),
            "powered_baseline_index": self.powered_baseline_index,
            "home_sample_index": self.home_sample_index,
            "final_sample_index": self.final_sample_index,
            "executed_phase_ids": list(self.executed_phase_ids),
            "executed_load_sha256": self.executed_load_sha256,
            "max_sample_gap_s": self.max_sample_gap_s,
        }
        payload["validated_trace_program_sha256"] = canonical_sha256(payload)
        return payload


def _finite(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise MechanismValidationError(
            "TASK_TRACE_NUMBER_INVALID",
            f"{label} must be a finite number",
        )
    return float(value)


def registered_adaptive_termination_evidence(
    phase: Any,
    *,
    actual_coordinate_si: float,
    reached: bool,
) -> dict[str, Any] | None:
    """Build the only accepted adaptive-stop evidence object."""

    if phase.stop_condition is None:
        return None
    coordinate = canonical_registered_stop_coordinate(
        str(phase.stop_condition),
        target_coordinate_si=_finite(
            phase.target_coordinate,
            f"{phase.phase_id} termination target",
        ),
        actual_coordinate_si=_finite(
            actual_coordinate_si,
            f"{phase.phase_id} termination coordinate",
        ),
    )
    if reached and not registered_stop_comparator_reached(
        str(phase.stop_condition),
        target_coordinate_si=float(phase.target_coordinate),
        actual_coordinate_si=coordinate,
    ):
        raise MechanismValidationError(
            "TASK_ADAPTIVE_TERMINATION_EVIDENCE_INVALID",
            f"{phase.phase_id} claims a stop without satisfying its comparator",
        )
    return {
        "schema_version": "registered_adaptive_stop_v1",
        "stop_condition": phase.stop_condition,
        "target_coordinate_si": phase.target_coordinate,
        "actual_coordinate_si": coordinate,
        "reason": (
            "registered_stop_condition_reached"
            if reached
            else "duration_limit"
        ),
        "branch_termination_policy": phase.branch_termination_policy,
    }


def canonical_registered_stop_coordinate(
    stop_condition: str,
    *,
    target_coordinate_si: float,
    actual_coordinate_si: float,
) -> float:
    """Normalize only representational one-sided error at a stop boundary."""

    target = _finite(target_coordinate_si, "registered stop target")
    actual = _finite(actual_coordinate_si, "registered stop coordinate")
    tolerance = (
        64.0
        * sys.float_info.epsilon
        * max(1.0, abs(target), abs(actual))
    )
    if (
        stop_condition in {"position_at_least", "progress_at_least"}
        and actual < target
        and target - actual <= tolerance
    ) or (
        stop_condition in {"position_at_most", "progress_at_most"}
        and actual > target
        and actual - target <= tolerance
    ):
        return target
    if stop_condition not in {
        "position_at_least",
        "progress_at_least",
        "position_at_most",
        "progress_at_most",
    }:
        raise MechanismValidationError(
            "TASK_ADAPTIVE_TERMINATION_EVIDENCE_INVALID",
            f"unsupported registered stop comparator {stop_condition!r}",
        )
    return actual


def registered_stop_comparator_reached(
    stop_condition: str,
    *,
    target_coordinate_si: float,
    actual_coordinate_si: float,
) -> bool:
    target = _finite(target_coordinate_si, "registered stop target")
    actual = canonical_registered_stop_coordinate(
        stop_condition,
        target_coordinate_si=target,
        actual_coordinate_si=actual_coordinate_si,
    )
    return (
        stop_condition in {"position_at_least", "progress_at_least"}
        and actual >= target
    ) or (
        stop_condition in {"position_at_most", "progress_at_most"}
        and actual <= target
    )


_CLOCK_PHASE_EXTENSION_KEYS = (
    "active_input_body_id",
    "active_input_point_abs_m",
    "termination_reason",
    "setting_clutch_engaged",
    "actuation_channels",
)

_DETENT_PHASE_EXTENSION_KEYS = (
    "actuation_channels",
    "termination_reason",
)


_PROGRAM_POLICIES: dict[
    str,
    dict[str, tuple[tuple[Any, ...], ...]],
] = {
    "plain_prismatic_v1": {
        "task_range_roundtrip": (
            (
                (
                    "task_outbound_force",
                    "task_outbound_settle",
                    "task_return_force",
                    "task_return_settle",
                ),
                ("task_outbound_force",),
                (
                    "task_outbound_settle",
                    "task_return_force",
                    "task_return_settle",
                ),
                "selected_input_only",
                ("task_outbound_force", "task_return_force"),
                3,
                1.0,
                (),
            ),
        ),
    },
    "plain_revolute_v1": {
        "task_angle_roundtrip": (
            (
                (
                    "task_outbound_torque",
                    "task_outbound_hold",
                    "task_passive_spring_return",
                    "task_return_settle",
                ),
                ("task_outbound_torque", "task_outbound_hold"),
                ("task_passive_spring_return", "task_return_settle"),
                "zero",
                ("task_outbound_torque",),
                3,
                1.0,
                (),
            ),
            (
                (
                    "task_outbound_torque",
                    "task_outbound_settle",
                    "task_return_torque",
                    "task_return_settle",
                ),
                ("task_outbound_torque",),
                (
                    "task_outbound_settle",
                    "task_return_torque",
                    "task_return_settle",
                ),
                "selected_input_only",
                ("task_outbound_torque", "task_return_torque"),
                3,
                1.0,
                (),
            ),
        ),
        # Phase layouts mirror the tabletop pickup fixture branch of
        # loads.default_revolute_load_cases.  The closing stroke and every
        # driven verify/lift phase carry the hand input; the settle and
        # passive phases run with zero task input.  The close phase carries
        # the registered position_at_least stop, so it is the only adaptive
        # phase.  The plain revolute worker records no executed-phase
        # extension keys beyond the base/adaptive schema.
        "tong_tabletop_pickup": (
            (
                (
                    "tong_pickup_initial_table_settle",
                    "tong_pickup_approach",
                    "tong_pickup_close",
                    "tong_pickup_grip_verify",
                    "tong_pickup_lift",
                    "tong_pickup_verify",
                ),
                (
                    "tong_pickup_close",
                    "tong_pickup_grip_verify",
                    "tong_pickup_lift",
                    "tong_pickup_verify",
                ),
                (
                    "tong_pickup_initial_table_settle",
                    "tong_pickup_approach",
                ),
                "zero",
                ("tong_pickup_close",),
                3,
                1.0,
                (),
            ),
        ),
        "tong_tabletop_hold": (
            (
                (
                    "tong_hold_initial_table_settle",
                    "tong_hold_approach",
                    "tong_hold_close",
                    "tong_hold_grip_verify",
                    "tong_hold_lift",
                    "tong_hold_verify",
                ),
                (
                    "tong_hold_close",
                    "tong_hold_grip_verify",
                    "tong_hold_lift",
                    "tong_hold_verify",
                ),
                (
                    "tong_hold_initial_table_settle",
                    "tong_hold_approach",
                ),
                "zero",
                ("tong_hold_close",),
                3,
                1.0,
                (),
            ),
        ),
        "tong_tabletop_release": (
            (
                (
                    "tong_release_initial_table_settle",
                    "tong_release_approach",
                    "tong_release_close",
                    "tong_release_grip_verify",
                    "tong_release_lift",
                    "tong_release_preload",
                    "tong_release_gradual_open",
                    "tong_release_passive_return",
                    "tong_release_final_table_settle",
                    "tong_release_observe",
                ),
                (
                    "tong_release_close",
                    "tong_release_grip_verify",
                    "tong_release_lift",
                    "tong_release_preload",
                    "tong_release_gradual_open",
                ),
                (
                    "tong_release_initial_table_settle",
                    "tong_release_approach",
                    "tong_release_passive_return",
                    "tong_release_final_table_settle",
                    "tong_release_observe",
                ),
                "zero",
                ("tong_release_close",),
                3,
                1.0,
                (),
            ),
        ),
    },
    "revolute_linkage_v1": {
        "task_output_angle_roundtrip": (
            (
                (
                    "press_pedal",
                    "open_settle",
                    "release_pedal",
                    "closed_settle",
                ),
                ("press_pedal",),
                ("open_settle", "release_pedal", "closed_settle"),
                "selected_input_only",
                ("press_pedal",),
                3,
                1.0,
                (),
            ),
        ),
    },
    "rack_pinion_v1": {
        "task_rack_roundtrip": (
            (
                (
                    "crank_lift",
                    "upper_settle",
                    "zero_input_hold",
                    "crank_lower",
                    "home_settle",
                ),
                ("crank_lift",),
                ("upper_settle", "zero_input_hold", "crank_lower", "home_settle"),
                "selected_input_only",
                ("crank_lift", "crank_lower"),
                3,
                1.0,
                (),
            ),
        ),
    },
    "bicycle_chain_sprocket_drive_v2": {
        "task_bicycle_powered_roundtrip": (
            (
                (
                    "finite_engagement_settle",
                    "measured_powered_drive",
                    "passive_return",
                    "return_settle",
                ),
                ("measured_powered_drive",),
                ("passive_return", "return_settle"),
                "zero",
                (),
                3,
                1.0,
                (),
            ),
        ),
    },
    "bicycle_ground_roll_v1": {
        "task_bicycle_ground_roll": (
            (
                (
                    "ground_contact_settle",
                    "rear_wheel_powered_drive",
                    "passive_roll_observe",
                ),
                ("rear_wheel_powered_drive",),
                ("passive_roll_observe",),
                "zero",
                (),
                3,
                1.0,
                (),
            ),
        ),
    },
    "hydraulic_scissor_lift_v1": {
        "task_platform_roundtrip": (
            (
                (
                    "cylinder_extend",
                    "raised_settle",
                    "cylinder_retract",
                    "home_settle",
                ),
                ("cylinder_extend",),
                ("raised_settle", "cylinder_retract", "home_settle"),
                "selected_input_only",
                ("cylinder_extend", "cylinder_retract"),
                3,
                1.0,
                (),
            ),
        ),
    },
    "multi_output_shaft_coupling_v1": {
        "task_egg_beater_cycle": (
            (
                (
                    "crank_forward",
                    "coast_after_forward",
                    "crank_reverse",
                    "coast_after_reverse",
                ),
                ("crank_forward",),
                (
                    "coast_after_forward",
                    "crank_reverse",
                    "coast_after_reverse",
                ),
                "selected_input_only",
                (),
                3,
                1.0,
                _CLOCK_PHASE_EXTENSION_KEYS,
            ),
        ),
    },
    "spur_gear_clock_v1": {
        "task_clock_cycle": (
            (
                (
                    "wind_and_advance",
                    "running_settle",
                    "controlled_return",
                    "home_settle",
                ),
                ("wind_and_advance",),
                ("running_settle", "controlled_return", "home_settle"),
                "selected_input_only",
                (),
                3,
                1.0,
                _CLOCK_PHASE_EXTENSION_KEYS,
            ),
        ),
        "alarm_clock_cycle": (
            (
                (
                    "wind_and_advance",
                    "running_settle",
                    "controlled_return",
                    "home_settle",
                ),
                ("wind_and_advance",),
                ("running_settle", "controlled_return", "home_settle"),
                "selected_input_only",
                (),
                3,
                1.0,
                _CLOCK_PHASE_EXTENSION_KEYS,
            ),
        ),
        "manual_time_setting_clock": (
            (
                (
                    "engage_and_set_forward",
                    "setting_settle",
                    "set_reverse",
                    "setting_home_settle",
                ),
                ("engage_and_set_forward",),
                (
                    "setting_settle",
                    "set_reverse",
                    "setting_home_settle",
                ),
                "selected_input_only",
                (),
                3,
                1.0,
                _CLOCK_PHASE_EXTENSION_KEYS,
            ),
        ),
        "hour_hand_alarm_setting_clock": (
            (
                (
                    "turn_hour_hand_to_alarm_target",
                    "alarm_setting_latched",
                ),
                ("turn_hour_hand_to_alarm_target",),
                ("alarm_setting_latched",),
                "zero",
                (),
                3,
                1.0,
                _CLOCK_PHASE_EXTENSION_KEYS,
            ),
        ),
        "scheduled_alarm_clock": (
            (
                (
                    "turn_hour_hand_to_alarm_target",
                    "alarm_setting_latched",
                    "restore_hour_hand_to_current_time",
                    "alarm_target_ready",
                    "advance_to_alarm_time",
                    "ring_with_timekeeping",
                    "scheduled_alarm_settle",
                ),
                ("advance_to_alarm_time", "ring_with_timekeeping"),
                ("scheduled_alarm_settle",),
                "zero",
                ("advance_to_alarm_time",),
                3,
                1.0,
                _CLOCK_PHASE_EXTENSION_KEYS,
            ),
        ),
    },
    "prismatic_detent_latch_v1": {
        # Phase layouts mirror detent_slider_loads.default_detent_slider_load_cases.
        # Prepress ramps and the slide stroke carry the task input; the release
        # and the externally loaded hold run with zero task input.  Every phase
        # whose load definition carries a stop_condition is registered adaptive.
        "task_detent_extend": (
            (
                (
                    "prepress_latch_ramp_25",
                    "prepress_latch_ramp_50",
                    "prepress_latch_ramp_75",
                    "prepress_latch_ramp_100",
                    "press_and_slide_to_detent",
                    "release_into_detent",
                ),
                (
                    "prepress_latch_ramp_25",
                    "prepress_latch_ramp_50",
                    "prepress_latch_ramp_75",
                    "prepress_latch_ramp_100",
                    "press_and_slide_to_detent",
                ),
                ("release_into_detent",),
                "zero",
                ("press_and_slide_to_detent",),
                3,
                1.0,
                _DETENT_PHASE_EXTENSION_KEYS,
            ),
        ),
        "task_detent_hold_under_load": (
            (
                (
                    "prepress_latch_ramp_25",
                    "prepress_latch_ramp_50",
                    "prepress_latch_ramp_75",
                    "prepress_latch_ramp_100",
                    "press_and_slide_to_detent",
                    "release_into_detent",
                    "axial_load_hold",
                ),
                (
                    "prepress_latch_ramp_25",
                    "prepress_latch_ramp_50",
                    "prepress_latch_ramp_75",
                    "prepress_latch_ramp_100",
                    "press_and_slide_to_detent",
                ),
                ("release_into_detent", "axial_load_hold"),
                "zero",
                ("press_and_slide_to_detent",),
                3,
                1.0,
                _DETENT_PHASE_EXTENSION_KEYS,
            ),
        ),
        "task_detent_second_position": (
            (
                (
                    "prepress_latch_ramp_25",
                    "prepress_latch_ramp_50",
                    "prepress_latch_ramp_75",
                    "prepress_latch_ramp_100",
                    "press_and_slide_to_detent",
                    "release_into_detent",
                ),
                (
                    "prepress_latch_ramp_25",
                    "prepress_latch_ramp_50",
                    "prepress_latch_ramp_75",
                    "prepress_latch_ramp_100",
                    "press_and_slide_to_detent",
                ),
                ("release_into_detent",),
                "zero",
                ("press_and_slide_to_detent",),
                3,
                1.0,
                _DETENT_PHASE_EXTENSION_KEYS,
            ),
        ),
        "task_detent_cycle": (
            (
                (
                    "prepress_latch_for_extension_ramp_25",
                    "prepress_latch_for_extension_ramp_50",
                    "prepress_latch_for_extension_ramp_75",
                    "prepress_latch_for_extension_ramp_100",
                    "press_and_slide_to_outer_detent",
                    "release_outer_detent",
                    "align_outer_detent_for_unlock",
                    "prepress_latch_for_retraction_ramp_25",
                    "prepress_latch_for_retraction_ramp_50",
                    "prepress_latch_for_retraction_ramp_75",
                    "prepress_latch_for_retraction_ramp_100",
                    "press_and_retract_home",
                    "release_home_detent",
                ),
                (
                    "prepress_latch_for_extension_ramp_25",
                    "prepress_latch_for_extension_ramp_50",
                    "prepress_latch_for_extension_ramp_75",
                    "prepress_latch_for_extension_ramp_100",
                    "press_and_slide_to_outer_detent",
                ),
                (
                    "release_outer_detent",
                    "align_outer_detent_for_unlock",
                    "prepress_latch_for_retraction_ramp_25",
                    "prepress_latch_for_retraction_ramp_50",
                    "prepress_latch_for_retraction_ramp_75",
                    "prepress_latch_for_retraction_ramp_100",
                    "press_and_retract_home",
                    "release_home_detent",
                ),
                "selected_input_only",
                (
                    "press_and_slide_to_outer_detent",
                    "align_outer_detent_for_unlock",
                    "press_and_retract_home",
                ),
                3,
                1.0,
                _DETENT_PHASE_EXTENSION_KEYS,
            ),
        ),
    },
}


def installed_task_programs() -> frozenset[str]:
    """Program names that already have a validator-owned window policy."""

    programs: set[str] = set()
    for layouts in _PROGRAM_POLICIES.values():
        programs.update(layouts)
    return frozenset(programs)


def task_program_spec_for_load_case(
    load_case: LoadCase,
    *,
    plugin_id: str,
    plan: MechanismPlan | None = None,
) -> TaskProgramEvidenceSpec:
    """Return an explicit ACTIVE-adapter program policy.

    No certification fallback infers phases from wrench directions, names, or
    sample extrema.  A new program must be added to this validator registry as
    part of capability activation.
    """

    layouts = _PROGRAM_POLICIES.get(plugin_id, {}).get(load_case.program, ())
    actual_phase_ids = tuple(phase.phase_id for phase in load_case.phases)
    match = next(
        (
            (
                powered,
                returned,
                return_policy,
                adaptive,
                minimum_powered_samples,
                maximum_gap_cadences,
                extension_keys,
            )
            for (
                phase_ids,
                powered,
                returned,
                return_policy,
                adaptive,
                minimum_powered_samples,
                maximum_gap_cadences,
                extension_keys,
            ) in layouts
            if phase_ids == actual_phase_ids
        ),
        None,
    )
    if match is None:
        from .task_plugins import reviewer_task_program_policy

        plugin_spec = reviewer_task_program_policy(load_case, plan=plan)
        if plugin_spec is not None:
            return plugin_spec
        from .family_plugins import plugin_registration_for

        try:
            mechanism_registration = plugin_registration_for(plugin_id)
        except MechanismValidationError:
            mechanism_registration = None
        if plugin_owner_for_registration(
            mechanism_registration,
            plugin_kind="mechanism",
            expected_plugin_id=plugin_id,
        ) is FailureOwner.REVIEWER_MECHANISM_PLUGIN:
            details = {
                "source_mechanism_plugin_id": plugin_id,
                "program": load_case.program,
                "load_case_id": load_case.load_case_id,
            }
            raise MechanismValidationError(
                "REVIEWER_TASK_ADAPTER_POLICY_MISSING",
                "reviewer mechanism adapter has no TaskAdapter window policy "
                f"for {load_case.program!r} with phases {actual_phase_ids!r}",
                details=details,
                failure=make_failure_envelope(
                    "REVIEWER_TASK_ADAPTER_POLICY_MISSING",
                    FailureOwner.REVIEWER_TASK_PLUGIN,
                    source_component="mechanism.task_motion_evidence",
                    message=(
                        "reviewer mechanism adapter requires a reviewer "
                        "TaskAdapter window policy"
                    ),
                    evidence_validity=EvidenceValidity.NOT_APPLICABLE,
                    plugin_kind="task",
                    registration_source="reviewer_task_adapter",
                    details=details,
                ),
            )
        raise MechanismValidationError(
            "TASK_PROGRAM_POLICY_UNREGISTERED",
            "ACTIVE mechanism adapter has no exact task-window policy for "
            f"{load_case.program!r} with phases {actual_phase_ids!r}",
        )
    (
        powered,
        returned,
        return_policy,
        adaptive,
        minimum_powered_samples,
        maximum_gap_cadences,
        extension_keys,
    ) = match
    return TaskProgramEvidenceSpec(
        program=load_case.program,
        powered_phase_ids=powered,
        return_phase_ids=returned,
        return_input_policy=return_policy,
        adaptive_phase_ids=adaptive,
        minimum_powered_sample_count=minimum_powered_samples,
        maximum_sample_gap_cadences=maximum_gap_cadences,
        executed_phase_extension_keys=extension_keys,
    )


def _normalized_executed_schedule(
    trace: Mapping[str, Any],
    load_case: LoadCase,
    *,
    program_spec: TaskProgramEvidenceSpec,
) -> tuple[list[dict[str, Any]], tuple[float, ...]]:
    raw_schedule = trace.get("executed_phases")
    if (
        not isinstance(raw_schedule, list)
        or len(raw_schedule) != len(load_case.phases)
    ):
        raise MechanismValidationError(
            "TASK_EXECUTED_SCHEDULE_INVALID",
            "trace must contain every validator load phase exactly once",
        )
    normalized: list[dict[str, Any]] = []
    ends: list[float] = []
    cumulative = 0.0
    for index, (raw, phase) in enumerate(
        zip(raw_schedule, load_case.phases)
    ):
        if not isinstance(raw, Mapping):
            raise MechanismValidationError(
                "TASK_EXECUTED_SCHEDULE_INVALID",
                f"executed phase {index} is not an object",
            )
        base_keys = {
            "phase_id",
            "wrench_kind",
            "direction_abs",
            "magnitude_si",
            "duration_s",
        }
        if phase.end_magnitude_si is not None:
            base_keys.add("end_magnitude_si")
        if phase.stop_condition is not None:
            if phase.phase_id not in program_spec.adaptive_phase_ids:
                raise MechanismValidationError(
                    "TASK_ADAPTIVE_PHASE_UNREGISTERED",
                    f"phase {phase.phase_id!r} has no registered adaptive policy",
                )
            base_keys.add("termination")
        allowed_keys = base_keys | set(
            program_spec.executed_phase_extension_keys
        )
        if set(raw) != allowed_keys:
            raise MechanismValidationError(
                "TASK_EXECUTED_PHASE_SCHEMA_MISMATCH",
                f"executed phase {index} does not use its closed schema",
            )
        expected = {
            "phase_id": phase.phase_id,
            "wrench_kind": phase.wrench_kind,
            "direction_abs": (
                None
                if phase.direction_abs is None
                else list(phase.direction_abs)
            ),
            "magnitude_si": phase.magnitude_si,
        }
        if phase.end_magnitude_si is not None:
            expected["end_magnitude_si"] = phase.end_magnitude_si
        for key, value in expected.items():
            if raw.get(key) != value:
                raise MechanismValidationError(
                    "TASK_EXECUTED_SCHEDULE_MISMATCH",
                    f"executed phase {index}.{key} differs from the load case",
                )
        duration = _finite(raw.get("duration_s"), f"phase {index} duration")
        if duration <= 0.0:
            raise MechanismValidationError(
                "TASK_EXECUTED_SCHEDULE_INVALID",
                f"executed phase {index} has an invalid duration",
            )
        # Integration arithmetic may introduce only machine-scale drift here.
        # A physics step is not a duration tolerance: accepting one would let a
        # producer silently delete the final step of every fixed phase.
        duration_tolerance = max(
            1.0e-12,
            abs(float(phase.duration_s)) * 1.0e-12,
        )
        full_duration = (
            abs(duration - phase.duration_s) <= duration_tolerance
        )
        if phase.stop_condition is None:
            if not full_duration:
                raise MechanismValidationError(
                    "TASK_EXECUTED_SCHEDULE_DURATION_MISMATCH",
                    f"nonadaptive phase {index} changes its registered duration",
                )
        else:
            termination = raw.get("termination")
            termination_keys = {
                "schema_version",
                "stop_condition",
                "target_coordinate_si",
                "actual_coordinate_si",
                "reason",
                "branch_termination_policy",
            }
            if (
                not isinstance(termination, Mapping)
                or set(termination) != termination_keys
                or termination.get("schema_version")
                != "registered_adaptive_stop_v1"
                or termination.get("stop_condition")
                != phase.stop_condition
                or termination.get("target_coordinate_si")
                != phase.target_coordinate
                or termination.get("branch_termination_policy")
                != phase.branch_termination_policy
                or termination.get("reason")
                not in {"duration_limit", "registered_stop_condition_reached"}
            ):
                raise MechanismValidationError(
                    "TASK_ADAPTIVE_TERMINATION_EVIDENCE_INVALID",
                    f"adaptive phase {index} lacks exact termination evidence",
                )
            coordinate = _finite(
                termination.get("actual_coordinate_si"),
                f"phase {index} termination coordinate",
            )
            target = float(phase.target_coordinate)
            reached = (
                phase.stop_condition
                in {"position_at_least", "progress_at_least"}
                and coordinate >= target
            ) or (
                phase.stop_condition
                in {"position_at_most", "progress_at_most"}
                and coordinate <= target
            )
            if (
                termination.get("reason")
                == "registered_stop_condition_reached"
                and not reached
            ):
                raise MechanismValidationError(
                    "TASK_ADAPTIVE_TERMINATION_EVIDENCE_INVALID",
                    f"adaptive phase {index} did not reach its registered target",
                )
            if (
                not full_duration
                and termination.get("reason")
                != "registered_stop_condition_reached"
            ):
                raise MechanismValidationError(
                    "TASK_EXECUTED_SCHEDULE_DURATION_MISMATCH",
                    f"adaptive phase {index} shortened without a reached target",
                )
            if duration > phase.duration_s + duration_tolerance:
                raise MechanismValidationError(
                    "TASK_EXECUTED_SCHEDULE_DURATION_MISMATCH",
                    f"adaptive phase {index} exceeds its registered duration",
                )
        item = dict(raw)
        item["duration_s"] = duration
        normalized.append(item)
        cumulative += duration
        ends.append(cumulative)
    digest = canonical_sha256(raw_schedule)
    if trace.get("executed_load_sha256") != digest:
        raise MechanismValidationError(
            "TASK_EXECUTED_SCHEDULE_HASH_MISMATCH",
            "trace executed-load hash does not match its schedule",
        )
    return normalized, tuple(ends)


def validate_trace_program(
    trace: Mapping[str, Any],
    load_case: LoadCase,
    *,
    program_spec: TaskProgramEvidenceSpec,
) -> ValidatedTraceProgram:
    """Fail closed unless a trace completely and densely covers its program."""

    if trace.get("status") != "COMPLETED":
        raise MechanismValidationError(
            "TASK_TRACE_NOT_COMPLETED",
            "task motion evidence requires a completed trace",
        )
    samples = trace.get("samples")
    if not isinstance(samples, list) or len(samples) < 2:
        raise MechanismValidationError(
            "TASK_TRACE_SAMPLES_MISSING",
            "task motion evidence requires at least two samples",
        )
    if trace.get("sample_count") != len(samples):
        raise MechanismValidationError(
            "TASK_TRACE_SAMPLE_COUNT_MISMATCH",
            "trace sample_count must equal the saved sample inventory",
        )
    if not all(isinstance(sample, Mapping) for sample in samples):
        raise MechanismValidationError(
            "TASK_TRACE_SAMPLE_INVALID",
            "task motion evidence contains a non-object sample",
        )
    spec = program_spec
    if spec.program != load_case.program:
        raise MechanismValidationError(
            "TASK_PROGRAM_MISMATCH",
            "task program evidence does not match the load case",
        )
    schedule, phase_ends = _normalized_executed_schedule(
        trace,
        load_case,
        program_spec=spec,
    )

    times = [
        _finite(sample.get("time_s"), f"samples[{index}].time_s")
        for index, sample in enumerate(samples)
    ]
    if abs(times[0]) > 1.0e-12 or samples[0].get("phase_id") != "initial":
        raise MechanismValidationError(
            "TASK_TRACE_INITIAL_BOUNDARY_MISSING",
            "trace must begin at time zero with the initial phase",
        )
    gaps = [right - left for left, right in zip(times, times[1:])]
    if any(gap <= 1.0e-12 for gap in gaps):
        raise MechanismValidationError(
            "TASK_TRACE_TIME_NOT_STRICT",
            "trace sample times must be strictly increasing",
        )
    max_allowed_gap = (
        load_case.sample_interval_s
        * spec.maximum_sample_gap_cadences
        + load_case.step_s
        + 1.0e-12
    )
    if any(gap > max_allowed_gap for gap in gaps):
        raise MechanismValidationError(
            "TASK_TRACE_SAMPLING_SPARSE",
            "trace contains a gap larger than the validator sampling cadence",
        )
    expected_end = phase_ends[-1]
    if abs(times[-1] - expected_end) > load_case.step_s + 1.0e-9:
        raise MechanismValidationError(
            "TASK_TRACE_FINAL_BOUNDARY_MISSING",
            "trace does not end at the executed schedule boundary",
        )

    phase_ids = tuple(phase.phase_id for phase in load_case.phases)
    phase_index = {phase_id: index for index, phase_id in enumerate(phase_ids)}
    observed_order: list[int] = []
    boundary_tolerance = max(1.0e-9, load_case.step_s * 1.0e-6)
    for sample_index, sample in enumerate(samples[1:], start=1):
        sample_phase = sample.get("phase_id")
        if sample_phase not in phase_index:
            raise MechanismValidationError(
                "TASK_TRACE_PHASE_UNKNOWN",
                f"sample {sample_index} uses an unknown phase",
            )
        time_s = times[sample_index]
        expected_phase_index = next(
            (
                index
                for index, end_time in enumerate(phase_ends)
                if time_s <= end_time + boundary_tolerance
            ),
            None,
        )
        if (
            expected_phase_index is None
            or sample_phase != phase_ids[expected_phase_index]
        ):
            raise MechanismValidationError(
                "TASK_TRACE_PHASE_TIME_MISMATCH",
                "sample phase label does not match the registered "
                "left-open/right-closed cumulative phase interval",
            )
        observed_order.append(phase_index[str(sample_phase)])
    if any(right < left for left, right in zip(observed_order, observed_order[1:])):
        raise MechanismValidationError(
            "TASK_TRACE_PHASE_ORDER_INVALID",
            "trace phase labels do not follow the executed load order",
        )
    tolerance = boundary_tolerance
    for index, (phase_id, end_time) in enumerate(zip(phase_ids, phase_ends)):
        if not any(
            sample.get("phase_id") == phase_id
            and abs(time_s - end_time) <= tolerance
            for sample, time_s in zip(samples, times)
        ):
            raise MechanismValidationError(
                "TASK_TRACE_PHASE_BOUNDARY_MISSING",
                f"trace has no saved end boundary for phase {phase_id!r}",
            )
        if index not in observed_order:
            raise MechanismValidationError(
                "TASK_TRACE_PHASE_COVERAGE_MISSING",
                f"trace has no sample for phase {phase_id!r}",
            )

    powered_set = set(spec.powered_phase_ids)
    return_set = set(spec.return_phase_ids)
    powered_indices = tuple(
        index
        for index, sample in enumerate(samples)
        if sample.get("phase_id") in powered_set
    )
    return_indices = tuple(
        index
        for index, sample in enumerate(samples)
        if sample.get("phase_id") in return_set
    )
    if not powered_indices:
        raise MechanismValidationError(
            "TASK_POWERED_WINDOW_EMPTY",
            "trace contains no sample in the designated powered window",
        )
    if len(powered_indices) < spec.minimum_powered_sample_count:
        raise MechanismValidationError(
            "TASK_TRACE_POWERED_SAMPLE_COUNT_INSUFFICIENT",
            "trace does not meet the registered powered-sample threshold",
        )
    first_powered = powered_indices[0]
    if first_powered <= 0:
        raise MechanismValidationError(
            "TASK_POWERED_BASELINE_MISSING",
            "powered motion requires a saved pre-powered baseline",
        )
    if spec.return_phase_ids and not return_indices:
        raise MechanismValidationError(
            "TASK_RETURN_WINDOW_EMPTY",
            "trace contains no sample in the designated return window",
        )
    return ValidatedTraceProgram(
        program_spec=spec,
        powered_sample_indices=powered_indices,
        return_sample_indices=return_indices,
        powered_baseline_index=first_powered - 1,
        home_sample_index=0,
        final_sample_index=len(samples) - 1,
        executed_phase_ids=tuple(item["phase_id"] for item in schedule),
        executed_load_sha256=str(trace["executed_load_sha256"]),
        max_sample_gap_s=max(gaps),
    )


def validate_aligned_trace_timeline(
    trace: Mapping[str, Any],
    physical_trace: Mapping[str, Any],
) -> None:
    """Require a derived branch to use the exact validated physical timeline."""

    samples = trace.get("samples")
    physical_samples = physical_trace.get("samples")
    if (
        not isinstance(samples, list)
        or not isinstance(physical_samples, list)
        or trace.get("sample_count") != len(samples)
        or len(samples) != len(physical_samples)
    ):
        raise MechanismValidationError(
            "TASK_TRACE_TIMELINE_MISMATCH",
            "derived trace sample inventory differs from the physical trace",
        )
    for index, (sample, physical) in enumerate(zip(samples, physical_samples)):
        if not isinstance(sample, Mapping) or not isinstance(physical, Mapping):
            raise MechanismValidationError(
                "TASK_TRACE_SAMPLE_INVALID",
                "derived trace contains a non-object sample",
            )
        if (
            sample.get("time_s") != physical.get("time_s")
            or sample.get("phase_id") != physical.get("phase_id")
        ):
            raise MechanismValidationError(
                "TASK_TRACE_TIMELINE_MISMATCH",
                f"derived trace timeline differs at sample {index}",
            )


def _body_inventory(
    family_spec: Mapping[str, Any],
) -> tuple[dict[str, str], tuple[str, ...]]:
    group_to_body: dict[str, str] = {}
    body_ids: list[str] = []
    raw_bodies = family_spec.get("bodies")
    if raw_bodies is None:
        return group_to_body, tuple(body_ids)
    if not isinstance(raw_bodies, list):
        raise MechanismValidationError(
            "TASK_BODY_INVENTORY_INVALID",
            "runtime family_spec.bodies must be a JSON array",
        )
    for index, raw in enumerate(raw_bodies):
        if not isinstance(raw, Mapping):
            raise MechanismValidationError(
                "TASK_BODY_INVENTORY_INVALID",
                f"runtime family_spec.bodies[{index}] must be an object",
            )
        body_id = raw.get("body_id")
        groups = raw.get("group_ids")
        rigid_group_id = raw.get("rigid_group_id")
        if groups is None and isinstance(rigid_group_id, str):
            groups = [rigid_group_id]
        elif (
            isinstance(groups, list)
            and rigid_group_id is not None
            and (
                not isinstance(rigid_group_id, str)
                or groups != [rigid_group_id]
            )
        ):
            raise MechanismValidationError(
                "TASK_GROUP_BODY_ALIAS",
                "body group_ids and rigid_group_id declarations disagree",
            )
        if not isinstance(body_id, str) or not body_id:
            raise MechanismValidationError(
                "TASK_BODY_INVENTORY_INVALID",
                f"runtime family_spec.bodies[{index}].body_id "
                "must be a non-empty string",
            )
        if not isinstance(groups, list) or not groups:
            raise MechanismValidationError(
                "TASK_BODY_INVENTORY_INVALID",
                f"runtime family_spec.bodies[{index}].group_ids "
                "must be a non-empty JSON array",
            )
        body_ids.append(body_id)
        for group_index, group_id in enumerate(groups):
            if not isinstance(group_id, str) or not group_id:
                raise MechanismValidationError(
                    "TASK_BODY_INVENTORY_INVALID",
                    "runtime family_spec.bodies"
                    f"[{index}].group_ids[{group_index}] "
                    "must be a non-empty string",
                )
            previous = group_to_body.setdefault(group_id, body_id)
            if previous != body_id:
                raise MechanismValidationError(
                    "TASK_GROUP_BODY_ALIAS",
                    f"rigid group {group_id!r} maps to multiple runtime bodies",
                )
    return group_to_body, tuple(body_ids)


def _runtime_channel_layout(
    plan: MechanismPlan,
) -> tuple[
    Any,
    dict[str, str],
    set[str],
    set[str],
    dict[str, str],
    dict[str, tuple[float, float, float]],
]:
    """Return an exact layout selected by the current ACTIVE plugin."""

    from .family_plugins import (
        mechanism_adapter_for_plan,
        plugin_registration_for,
        registration_allows_runtime_execution,
    )

    plugin, adapter = mechanism_adapter_for_plan(plan)
    registration = plugin_registration_for(plugin.plugin_id)
    if not registration_allows_runtime_execution(registration):
        raise MechanismValidationError(
            "TASK_CHANNEL_PLUGIN_NOT_ACTIVE",
            "task channel registry requires the current ACTIVE plugin or its "
            "live exact-one-test dispatch",
        )
    family_spec = (
        adapter.family_spec
        if isinstance(adapter.family_spec, Mapping)
        else {}
    )
    group_to_body, body_ids = _body_inventory(family_spec)
    input_ids: set[str] = set()
    output_ids: set[str] = set()
    fields: dict[str, str] = {}
    axes: dict[str, tuple[float, float, float]] = {}

    if not body_ids:
        for group_id in plan.fixed_group_ids:
            group_to_body[group_id] = "fixed_group"
        for group_id in plan.moving_group_ids:
            group_to_body[group_id] = "moving_group"

    layout = invoke_plugin_callback(
        lambda: plugin.task_channel_layout(plan, family_spec, body_ids),
        registration=registration,
        plugin_id=str(plugin.plugin_id),
        plugin_kind="mechanism",
        callback="task_channel_layout",
        source_component="mechanism.task_motion_evidence.channel_layout",
        unexpected_code="MECHANISM_ADAPTER_TASK_CHANNEL_LAYOUT_FAILED",
        details={"capsule_id": plan.capsule_id},
    )
    if layout is None:
        if body_ids:
            message = (
                f"plugin {plugin.plugin_id!r} has no exact task channel layout"
            )
            raise plugin_boundary_error(
                TypeError(message),
                registration=registration,
                plugin_id=str(plugin.plugin_id),
                plugin_kind="mechanism",
                callback="task_channel_layout",
                source_component=(
                    "mechanism.task_motion_evidence.channel_layout"
                ),
                code="MECHANISM_ADAPTER_TASK_CHANNEL_LAYOUT_REQUIRED",
                message=message,
                details={"capsule_id": plan.capsule_id},
                contract_violation=True,
            )
        input_ids.add("moving_group")
        output_ids.add("moving_group")
        fields["moving_group"] = plugin.coordinate.position_field
        axes["moving_group"] = tuple(plan.axis_abs)
    else:
        if (
            not isinstance(layout, tuple)
            or len(layout) != 4
            or not isinstance(layout[0], set)
            or not isinstance(layout[1], set)
            or not isinstance(layout[2], dict)
            or not isinstance(layout[3], dict)
        ):
            message = "task_channel_layout() returned an invalid layout"
            raise plugin_boundary_error(
                TypeError(message),
                registration=registration,
                plugin_id=str(plugin.plugin_id),
                plugin_kind="mechanism",
                callback="task_channel_layout",
                source_component=(
                    "mechanism.task_motion_evidence.channel_layout"
                ),
                code="MECHANISM_ADAPTER_TASK_CHANNEL_LAYOUT_INVALID",
                message=message,
                details={"capsule_id": plan.capsule_id},
                contract_violation=True,
            )
        input_ids, output_ids, fields, axes = layout

    input_ids.discard("")
    output_ids.discard("")
    if not input_ids or not output_ids:
        raise MechanismValidationError(
            "TASK_RUNTIME_IO_UNRESOLVED",
            "ACTIVE plugin does not expose exact runtime input/output bodies",
        )
    return (
        plugin,
        group_to_body,
        input_ids,
        output_ids,
        fields,
        axes,
    )


def build_task_channel_registry(
    plan: MechanismPlan,
    capsule: Mapping[str, Any],
) -> dict[str, Any]:
    """Capture exact capsule IDs/frames onto the ACTIVE runtime layout."""

    (
        plugin,
        group_to_body,
        input_ids,
        output_ids,
        fields,
        axes,
    ) = _runtime_channel_layout(plan)
    plugin_id = plugin.plugin_id
    inputs: dict[str, ResolvedTaskChannel] = {}
    observables: dict[str, ResolvedTaskChannel] = {}
    for raw in capsule.get("input_ports", []):
        if not isinstance(raw, Mapping):
            continue
        port_id = str(raw.get("port_id") or "")
        group_id = str(raw.get("rigid_group_id") or "")
        frame_id = str(raw.get("frame_id") or "")
        body_id = group_to_body.get(group_id)
        direction = raw.get("direction_abs")
        runtime_axis = axes.get(body_id) if body_id is not None else None
        if (
            not port_id
            or not frame_id
            or body_id is None
            or not isinstance(direction, (list, tuple))
            or len(direction) != 3
            or runtime_axis is None
            or len(runtime_axis) != 3
            or any(
                abs(float(actual) - float(expected)) > 1.0e-9
                for actual, expected in zip(direction, runtime_axis)
            )
        ):
            raise MechanismValidationError(
                "TASK_CHANNEL_BINDING_INCOMPLETE",
                "capsule input port cannot be bound to the runtime layout",
            )
        inputs[port_id] = ResolvedTaskChannel(
            semantic_id=port_id,
            rigid_group_id=group_id,
            body_id=body_id,
            frame_id=frame_id,
            canonical_field=fields.get(body_id),
            axis_abs=tuple(float(item) for item in runtime_axis),
            wrench_type=str(raw.get("wrench_type") or ""),
            actuation_body_ids=(
                tuple(sorted(input_ids))
                if plugin.task_input_actuation_spans_all_inputs
                else (body_id,)
            ),
        )
    for raw in capsule.get("observables", []):
        if not isinstance(raw, Mapping):
            continue
        observable_id = str(raw.get("observable_id") or "")
        group_id = str(raw.get("rigid_group_id") or "")
        frame_id = str(raw.get("frame_id") or "")
        body_id = group_to_body.get(group_id)
        if not observable_id or not frame_id or body_id is None:
            raise MechanismValidationError(
                "TASK_CHANNEL_BINDING_INCOMPLETE",
                "capsule observable cannot be bound to the runtime layout",
            )
        observables[observable_id] = ResolvedTaskChannel(
            semantic_id=observable_id,
            rigid_group_id=group_id,
            body_id=body_id,
            frame_id=frame_id,
            canonical_field=fields.get(body_id),
            axis_abs=axes.get(body_id),
            quantity=str(raw.get("quantity") or ""),
        )
    registry = TaskChannelRegistry(
        plugin_id=plugin_id,
        input_body_ids=tuple(sorted(input_ids)),
        output_body_ids=tuple(sorted(output_ids)),
        inputs=inputs,
        observables=observables,
    )
    return registry.to_dict()


def _channel_from_dict(value: Mapping[str, Any]) -> ResolvedTaskChannel:
    axis = value.get("axis_abs")
    return ResolvedTaskChannel(
        semantic_id=str(value.get("semantic_id") or ""),
        rigid_group_id=str(value.get("rigid_group_id") or ""),
        body_id=str(value.get("body_id") or ""),
        frame_id=str(value.get("frame_id") or ""),
        canonical_field=(
            str(value["canonical_field"])
            if isinstance(value.get("canonical_field"), str)
            else None
        ),
        axis_abs=(
            tuple(float(item) for item in axis)
            if isinstance(axis, list) and len(axis) == 3
            else None
        ),
        wrench_type=(
            str(value["wrench_type"])
            if isinstance(value.get("wrench_type"), str)
            else None
        ),
        quantity=(
            str(value["quantity"])
            if isinstance(value.get("quantity"), str)
            else None
        ),
        actuation_body_ids=tuple(
            str(item) for item in value.get("actuation_body_ids") or ()
        ),
    )


def task_channel_registry_for_plan(plan: MechanismPlan) -> TaskChannelRegistry:
    """Load and revalidate the exact discovery-time channel inventory."""

    raw = getattr(plan, "task_channel_registry", None)
    if not isinstance(raw, Mapping):
        raise MechanismValidationError(
            "TASK_CHANNEL_REGISTRY_MISSING",
            "mechanism plan has no discovery-time task channel registry",
        )
    payload = dict(raw)
    recorded = payload.pop("channel_registry_sha256", None)
    if (
        raw.get("schema_version") != TASK_CHANNEL_REGISTRY_SCHEMA
        or recorded != canonical_sha256(payload)
    ):
        raise MechanismValidationError(
            "TASK_CHANNEL_REGISTRY_HASH_MISMATCH",
            "saved task channel registry is invalid or changed",
        )
    inputs_raw = raw.get("inputs")
    observables_raw = raw.get("observables")
    if not isinstance(inputs_raw, Mapping) or not isinstance(
        observables_raw, Mapping
    ):
        raise MechanismValidationError(
            "TASK_CHANNEL_REGISTRY_INVALID",
            "saved task channel registry lacks exact channel inventories",
        )
    if (
        not all(isinstance(value, Mapping) for value in inputs_raw.values())
        or not all(
            isinstance(value, Mapping) for value in observables_raw.values()
        )
    ):
        raise MechanismValidationError(
            "TASK_CHANNEL_REGISTRY_INVALID",
            "saved channel inventories contain non-object entries",
        )
    registry = TaskChannelRegistry(
        plugin_id=str(raw.get("plugin_id") or ""),
        input_body_ids=tuple(str(item) for item in raw.get("input_body_ids") or ()),
        output_body_ids=tuple(str(item) for item in raw.get("output_body_ids") or ()),
        inputs={
            str(key): _channel_from_dict(value)
            for key, value in inputs_raw.items()
        },
        observables={
            str(key): _channel_from_dict(value)
            for key, value in observables_raw.items()
        },
    )
    (
        current_plugin,
        group_to_body,
        input_ids,
        output_ids,
        fields,
        axes,
    ) = _runtime_channel_layout(plan)
    if (
        registry.plugin_id != current_plugin.plugin_id
        or set(registry.input_body_ids) != input_ids
        or set(registry.output_body_ids) != output_ids
    ):
        raise MechanismValidationError(
            "TASK_CHANNEL_REGISTRY_RUNTIME_MISMATCH",
            "saved channel registry differs from the current ACTIVE adapter",
        )
    # Materialization may compact an adapter's body ``group_ids`` down to
    # collision-owning groups while leaving the discovery-time semantic
    # channels unchanged.  The registry is already hash-bound into the plan,
    # so restore only missing, non-conflicting aliases whose saved body still
    # belongs to the corresponding current runtime role.  Existing mappings,
    # fields, axes, and actuation roles remain strict below.
    for channel, allowed_body_ids in (
        *((item, input_ids) for item in registry.inputs.values()),
        *((item, output_ids) for item in registry.observables.values()),
    ):
        if (
            channel.rigid_group_id not in group_to_body
            and channel.body_id in allowed_body_ids
        ):
            group_to_body[channel.rigid_group_id] = channel.body_id
    expected_actuation = {
        key: (
            tuple(sorted(input_ids))
            if current_plugin.task_input_actuation_spans_all_inputs
            else (channel.body_id,)
        )
        for key, channel in registry.inputs.items()
    }
    for key, channel in registry.inputs.items():
        if (
            key != channel.semantic_id
            or channel.body_id not in input_ids
            or group_to_body.get(channel.rigid_group_id) != channel.body_id
            or fields.get(channel.body_id) != channel.canonical_field
            or axes.get(channel.body_id) != channel.axis_abs
            or channel.actuation_body_ids != expected_actuation[key]
        ):
            raise MechanismValidationError(
                "TASK_CHANNEL_REGISTRY_RUNTIME_MISMATCH",
                "saved input body, field, axis, or actuation mapping changed",
            )
    for key, channel in registry.observables.items():
        if (
            key != channel.semantic_id
            or group_to_body.get(channel.rigid_group_id) != channel.body_id
            or fields.get(channel.body_id) != channel.canonical_field
            or axes.get(channel.body_id) != channel.axis_abs
        ):
            raise MechanismValidationError(
                "TASK_CHANNEL_REGISTRY_RUNTIME_MISMATCH",
                "saved observable body, field, or axis changed",
            )
    return registry


def resolve_and_validate_task_channels(
    requirement: Mapping[str, Any],
    binding: Mapping[str, Any],
    registry: TaskChannelRegistry,
) -> dict[str, Any]:
    """Resolve semantic roles and reject body/channel alias or runtime swaps."""

    input_raw = binding.get("input")
    target_raw = binding.get("target")
    if not isinstance(input_raw, Mapping) or not isinstance(
        target_raw, Mapping
    ):
        raise MechanismValidationError(
            "TASK_CHANNEL_BINDING_INCOMPLETE",
            "task acceptance requires both input and target channels",
        )
    selected_input = registry.resolve(input_raw, role="input")
    target = registry.resolve(target_raw, role="target")
    if selected_input.body_id not in registry.input_body_ids:
        raise MechanismValidationError(
            "TASK_SEMANTIC_RUNTIME_INPUT_MISMATCH",
            "semantic input does not map to a runtime input body",
        )
    if (
        requirement.get("allow_direct_output_actuation") is False
        and target.body_id
        in (
            selected_input.actuation_body_ids
            or (selected_input.body_id,)
        )
    ):
        raise MechanismValidationError(
            "TASK_DIRECT_OUTPUT_BODY_FORBIDDEN",
            "semantic input and target resolve to the same runtime body",
        )
    metric_id = str(requirement.get("metric_id") or "")
    if (
        metric_id in MOTION_WINDOW_METRICS
        and target.body_id not in registry.output_body_ids
    ):
        raise MechanismValidationError(
            "TASK_SEMANTIC_RUNTIME_OUTPUT_MISMATCH",
            "semantic target does not map to a runtime output body",
        )
    if metric_id in MOTION_WINDOW_METRICS and not target.canonical_field:
        raise MechanismValidationError(
            "TASK_OBSERVABLE_CHANNEL_UNRESOLVED",
            "target body has no adapter-owned canonical trace channel",
        )

    reference: ResolvedTaskChannel | None = None
    reference_raw = binding.get("reference")
    if metric_id in RATIO_METRICS:
        if not isinstance(reference_raw, Mapping):
            raise MechanismValidationError(
                "TASK_REFERENCE_BINDING_MISSING",
                "ratio metrics require a reference channel",
            )
        reference = registry.resolve(reference_raw, role="reference")
        task_suite_binding = requirement.get("task_suite_binding")
        reference_policy = (
            task_suite_binding.get("reference_policy")
            if isinstance(task_suite_binding, Mapping)
            else "default"
        )
        if reference_policy == "distinct_output":
            if reference.body_id not in registry.output_body_ids:
                raise MechanismValidationError(
                    "TASK_RATIO_REFERENCE_RUNTIME_MISMATCH",
                    "task-suite ratio reference is not a registered output",
                )
        elif (
            reference.body_id != selected_input.body_id
            or reference.canonical_field != selected_input.canonical_field
            or reference.axis_abs != selected_input.axis_abs
        ):
            raise MechanismValidationError(
                "TASK_RATIO_REFERENCE_INPUT_MISMATCH",
                "ratio reference is not the exact semantically selected input",
            )
        if (
            reference.body_id == target.body_id
            or reference.canonical_field == target.canonical_field
        ):
            raise MechanismValidationError(
                "TASK_RATIO_CHANNEL_ALIAS",
                "ratio target and reference must be distinct physical coordinates",
            )
        if (
            reference_policy != "distinct_output"
            and reference.body_id not in registry.input_body_ids
        ):
            raise MechanismValidationError(
                "TASK_RATIO_REFERENCE_RUNTIME_MISMATCH",
                "ratio reference does not map to a runtime input body",
            )
        if not reference.canonical_field:
            raise MechanismValidationError(
                "TASK_OBSERVABLE_CHANNEL_UNRESOLVED",
                "ratio reference has no adapter-owned canonical trace channel",
            )

    payload = {
        "schema_version": TASK_CHANNEL_REGISTRY_SCHEMA,
        "registry": registry.to_dict(),
        "input": selected_input.to_dict(),
        "target": target.to_dict(),
        "reference": None if reference is None else reference.to_dict(),
    }
    payload["resolved_channels_sha256"] = canonical_sha256(payload)
    return payload
