from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .models import MechanismValidationError


DEFAULT_PHYSICAL_SAMPLE_INTERVAL_S = 0.05
DEFAULT_INTERPENETRATION_TOLERANCE_M = 0.00035


def _fail(code: str, message: str) -> MechanismValidationError:
    return MechanismValidationError(code, message)


def _vector(value: Any, size: int, field: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise _fail("SAMPLING_POSE_INVALID", f"{field} must be a numeric vector")
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise _fail("SAMPLING_POSE_INVALID", f"{field} must be numeric") from exc
    if len(result) != size or not all(math.isfinite(item) for item in result):
        raise _fail("SAMPLING_POSE_INVALID", f"{field} has invalid values")
    return result


def _quaternion_angle(left: Sequence[float], right: Sequence[float]) -> float:
    left_q = _vector(left, 4, "left quaternion")
    right_q = _vector(right, 4, "right quaternion")
    left_norm = math.sqrt(sum(item * item for item in left_q))
    right_norm = math.sqrt(sum(item * item for item in right_q))
    if left_norm <= 1.0e-12 or right_norm <= 1.0e-12:
        raise _fail("SAMPLING_POSE_INVALID", "sampling quaternion is zero")
    dot = abs(
        sum(a * b for a, b in zip(left_q, right_q)) / (left_norm * right_norm)
    )
    return 2.0 * math.acos(max(-1.0, min(1.0, dot)))


def maximum_pose_step_sweep_m(
    previous: Mapping[str, Mapping[str, Any]],
    current: Mapping[str, Mapping[str, Any]],
    *,
    body_sweep_radii_m: Mapping[str, float] | None = None,
) -> float:
    """Return a conservative translation plus rotational-sweep step bound."""

    if set(previous) != set(current):
        raise _fail(
            "SAMPLING_BODY_SET_CHANGED",
            "physical body identifiers changed within one trajectory",
        )
    radii = body_sweep_radii_m or {}
    maximum = 0.0
    for body_id in previous:
        before = previous[body_id]
        after = current[body_id]
        before_position = _vector(before.get("position_abs_m"), 3, "position_abs_m")
        after_position = _vector(after.get("position_abs_m"), 3, "position_abs_m")
        translation = math.sqrt(
            sum((a - b) * (a - b) for a, b in zip(after_position, before_position))
        )
        rotation = _quaternion_angle(
            before.get("rotation_abs_quat_wxyz"),
            after.get("rotation_abs_quat_wxyz"),
        )
        radius = max(0.0, float(radii.get(body_id, 0.0)))
        maximum = max(maximum, translation + radius * rotation)
    return maximum


@dataclass(frozen=True)
class SamplingDecision:
    sample: bool
    reasons: tuple[str, ...]
    max_pose_step_sweep_m: float


class AdaptiveSamplingTracker:
    """Cheap per-integrator-step trigger for a sparse physical trace.

    This tracker never performs mesh collision queries. It observes body poses
    and the simulator contact count, retaining the 20 Hz nominal cadence while
    adding evidence at contact transitions or an unexpectedly large single
    integrator step.
    """

    def __init__(
        self,
        *,
        step_s: float,
        nominal_interval_s: float = DEFAULT_PHYSICAL_SAMPLE_INTERVAL_S,
        interpenetration_tolerance_m: float = DEFAULT_INTERPENETRATION_TOLERANCE_M,
        body_sweep_radii_m: Mapping[str, float] | None = None,
        initial_poses: Mapping[str, Mapping[str, Any]],
        initial_contact_count: int = 0,
    ) -> None:
        self.step_s = float(step_s)
        self.nominal_interval_s = float(nominal_interval_s)
        self.motion_step_trigger_m = max(
            1.0e-5, 0.5 * float(interpenetration_tolerance_m)
        )
        if not 1.0e-6 <= self.step_s <= 0.01:
            raise _fail("SAMPLING_STEP_INVALID", "sampling step is outside bounds")
        if self.nominal_interval_s < self.step_s:
            raise _fail(
                "SAMPLING_INTERVAL_INVALID",
                "nominal sampling interval cannot be below the integrator step",
            )
        self.body_sweep_radii_m = dict(body_sweep_radii_m or {})
        self.previous_poses = {
            body_id: dict(pose) for body_id, pose in initial_poses.items()
        }
        self.previous_contact_count = int(initial_contact_count)
        self.next_nominal_s = self.nominal_interval_s
        self.observed_step_count = 0
        self.nominal_sample_count = 0
        self.contact_transition_sample_count = 0
        self.fast_pose_step_sample_count = 0
        self.max_observed_pose_step_sweep_m = 0.0

    def observe(
        self,
        *,
        time_s: float,
        poses: Mapping[str, Mapping[str, Any]],
        contact_count: int,
    ) -> SamplingDecision:
        current_time = float(time_s)
        if not math.isfinite(current_time) or current_time < 0.0:
            raise _fail("SAMPLING_TIME_INVALID", "sampling time is invalid")
        sweep = maximum_pose_step_sweep_m(
            self.previous_poses,
            poses,
            body_sweep_radii_m=self.body_sweep_radii_m,
        )
        self.max_observed_pose_step_sweep_m = max(
            self.max_observed_pose_step_sweep_m, sweep
        )
        reasons: list[str] = []
        if current_time + 1.0e-12 >= self.next_nominal_s:
            reasons.append("nominal_interval")
            self.nominal_sample_count += 1
            while self.next_nominal_s <= current_time + 1.0e-12:
                self.next_nominal_s += self.nominal_interval_s
        current_contacts = int(contact_count)
        if current_contacts != self.previous_contact_count:
            reasons.append("contact_count_transition")
            self.contact_transition_sample_count += 1
        if sweep > self.motion_step_trigger_m:
            reasons.append("fast_pose_step")
            self.fast_pose_step_sample_count += 1
        self.previous_poses = {body_id: dict(pose) for body_id, pose in poses.items()}
        self.previous_contact_count = current_contacts
        self.observed_step_count += 1
        return SamplingDecision(bool(reasons), tuple(reasons), sweep)

    def disclosure(self, samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        times = sorted(float(item.get("time_s", 0.0)) for item in samples)
        gaps = [right - left for left, right in zip(times, times[1:])]
        return {
            "schema_version": "0.1",
            "mode": "nominal_time_grid_with_validator_owned_event_samples",
            "nominal_interval_s": self.nominal_interval_s,
            "integrator_step_s": self.step_s,
            "motion_step_trigger_m": self.motion_step_trigger_m,
            "mandatory_samples": [
                "initial",
                "phase_boundary",
                "stop_condition",
                "final",
            ],
            "adaptive_triggers": [
                "contact_count_transition",
                "fast_pose_step",
            ],
            "observed_integrator_step_count": self.observed_step_count,
            "saved_sample_count": len(samples),
            "nominal_trigger_count": self.nominal_sample_count,
            "contact_transition_trigger_count": self.contact_transition_sample_count,
            "fast_pose_step_trigger_count": self.fast_pose_step_sample_count,
            "max_observed_pose_step_sweep_m": self.max_observed_pose_step_sweep_m,
            "max_saved_sample_gap_s": max(gaps, default=0.0),
            "sample_times_non_uniform": len({round(item, 12) for item in gaps}) > 1,
            "every_integrator_step_mesh_audited": False,
        }


def select_saved_trace_samples(
    samples: Sequence[Mapping[str, Any]],
    *,
    nominal_interval_s: float = DEFAULT_PHYSICAL_SAMPLE_INTERVAL_S,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select a deterministic 20 Hz legacy audit set plus risk boundaries."""

    if not samples:
        raise _fail("SAMPLING_SAMPLES_MISSING", "saved physical trace has no samples")
    interval = float(nominal_interval_s)
    if not 0.001 <= interval <= 1.0:
        raise _fail("SAMPLING_INTERVAL_INVALID", "legacy audit interval is invalid")
    times = [float(item.get("time_s", 0.0)) for item in samples]
    if any(not math.isfinite(item) for item in times) or any(
        right < left for left, right in zip(times, times[1:])
    ):
        raise _fail("SAMPLING_TIME_INVALID", "saved trace times are not monotonic")

    selected_indices = {0, len(samples) - 1}
    next_nominal = times[0] + interval
    previous_phase = samples[0].get("phase_id")
    previous_contacts = int(samples[0].get("contact_count", 0))
    phase_boundary_count = 0
    contact_transition_count = 0
    nominal_count = 0
    for index, (sample, time_s) in enumerate(zip(samples[1:], times[1:]), start=1):
        if time_s + 1.0e-12 >= next_nominal:
            selected_indices.add(index)
            nominal_count += 1
            while next_nominal <= time_s + 1.0e-12:
                next_nominal += interval
        phase = sample.get("phase_id")
        if phase != previous_phase:
            selected_indices.update((index - 1, index))
            phase_boundary_count += 1
        contacts = int(sample.get("contact_count", 0))
        if contacts != previous_contacts:
            selected_indices.update((index - 1, index))
            contact_transition_count += 1
        previous_phase = phase
        previous_contacts = contacts

    indices = sorted(selected_indices)
    selected = []
    for index in indices:
        item = dict(samples[index])
        # The legality audit enumerates this compact list, while the dashboard
        # replays the original saved trace. Preserve the source frame so visual
        # evidence can jump back to the exact pose instead of treating the two
        # index spaces as interchangeable.
        item["_audit_source_sample_index"] = index
        selected.append(item)
    selected_times = [times[index] for index in indices]
    gaps = [right - left for left, right in zip(selected_times, selected_times[1:])]
    return selected, {
        "schema_version": "0.1",
        "mode": "legacy_saved_trace_20hz_plus_boundaries_and_contact_transitions",
        "nominal_interval_s": interval,
        "original_saved_sample_count": len(samples),
        "selected_sample_count": len(selected),
        "nominal_trigger_count": nominal_count,
        "phase_boundary_trigger_count": phase_boundary_count,
        "contact_transition_trigger_count": contact_transition_count,
        "maximum_selected_sample_gap_s": max(gaps, default=0.0),
        "selected_source_indices": indices,
        "every_integrator_step_mesh_audited": False,
    }

def legacy_sampling_disclosure(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    times = sorted(float(item.get("time_s", 0.0)) for item in samples)
    gaps = [right - left for left, right in zip(times, times[1:])]
    positive = [item for item in gaps if item > 1.0e-12]
    return {
        "schema_version": "0.1",
        "mode": "historical_saved_trace_samples",
        "saved_sample_count": len(samples),
        "minimum_sample_gap_s": min(positive, default=0.0),
        "maximum_sample_gap_s": max(positive, default=0.0),
        "sample_times_non_uniform": len({round(item, 12) for item in positive}) > 1,
        "adaptive_policy_recorded": False,
        "every_integrator_step_mesh_audited": False,
    }


