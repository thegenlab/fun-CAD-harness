from __future__ import annotations

import math
from typing import Any, Mapping

from .clock_gear_train_spec import clock_gear_train_from_dict
from .io import canonical_sha256
from .models import LoadCase, MechanismPlan, MechanismValidationError


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MechanismValidationError('CLOCK_TRACE_NUMBER_REQUIRED', f'{label} must be numeric')
    result = float(value)
    if not math.isfinite(result):
        raise MechanismValidationError('CLOCK_TRACE_NUMBER_INVALID', f'{label} must be finite')
    return result


def _validate_trace(
    trace: dict[str, Any],
    *,
    branch: str,
    plan: MechanismPlan,
    load_case: LoadCase,
    spec: Any,
) -> list[dict[str, Any]]:
    if trace.get('status') != 'COMPLETED':
        raise MechanismValidationError('TRACE_NOT_COMPLETED', f'{branch} clock trace did not complete')
    expected = {
        'branch': branch,
        'package_binding_sha256': plan.package_binding_sha256,
        'capsule_id': plan.capsule_id,
        'load_case_id': load_case.load_case_id,
        'plan_sha256': canonical_sha256(plan.to_dict()),
        'load_case_sha256': canonical_sha256(load_case.to_dict()),
    }
    mismatches = [key for key, value in expected.items() if trace.get(key) != value]
    if mismatches:
        raise MechanismValidationError(
            'TRACE_IDENTITY_MISMATCH',
            f'{branch} clock trace identity mismatch: {", ".join(mismatches)}',
        )
    samples = trace.get('samples')
    if not isinstance(samples, list) or len(samples) < 2:
        raise MechanismValidationError('TRACE_SAMPLES_MISSING', f'{branch} clock trace has too few samples')
    body_ids = {item.body_id for item in spec.bodies}
    shaft_body_ids = {item.body_id for item in spec.shafts}
    mesh_ids = {
        *(item.mesh_id for item in spec.gear_meshes),
        *(item.transfer_id for item in spec.shaft_transfers),
    }
    shaft_ids = {item.shaft_id for item in spec.shafts}
    follower_ids = {item.follower_id for item in spec.cam_followers}
    allowed_phases = {'initial', *(phase.phase_id for phase in load_case.phases)}
    previous_time = -math.inf
    for index, sample in enumerate(samples):
        label = f'{branch}.samples[{index}]'
        if not isinstance(sample, dict) or sample.get('phase_id') not in allowed_phases:
            raise MechanismValidationError('TRACE_SAMPLE_INVALID', f'{label} has an invalid phase')
        time_s = _finite(sample.get('time_s'), f'{label}.time_s')
        if time_s < previous_time - 1.0e-12:
            raise MechanismValidationError('TRACE_TIME_REVERSED', f'{branch} trace time is not monotonic')
        previous_time = time_s
        for field in (
            'clock_output_angle_rad',
            'clock_output_speed_rad_s',
            'clock_input_angle_rad',
            'clock_input_speed_rad_s',
            'barrel_input_angle_rad',
            'setting_input_angle_rad',
            'hour_hand_angle_rad',
            'hour_hand_speed_rad_s',
            'second_hand_angle_rad',
            'second_hand_speed_rad_s',
            'alarm_angle_rad',
            'alarm_speed_rad_s',
            'max_mesh_error_rad',
            'max_support_drift_m',
            'max_off_axis_rotation_rad',
            'setting_clutch_error_rad',
        ):
            _finite(sample.get(field), f'{label}.{field}')
        if spec.alarm_setting_clutch is not None:
            for field in (
                'alarm_setting_target_angle_rad',
                'alarm_setting_clutch_error_rad',
            ):
                _finite(sample.get(field), f'{label}.{field}')
        inventories = (
            ('shaft_angles_rad', shaft_body_ids),
            ('shaft_speeds_rad_s', shaft_body_ids),
            ('mesh_errors_rad', mesh_ids),
            ('support_drifts_m', shaft_ids),
            ('off_axis_rotations_rad', shaft_ids),
            ('cam_follower_errors_rad', follower_ids),
            ('body_poses', body_ids),
        )
        for field, expected_ids in inventories:
            value = sample.get(field)
            if not isinstance(value, dict) or set(value) != expected_ids:
                raise MechanismValidationError(
                    'CLOCK_TRACE_INVENTORY_MISMATCH', f'{label}.{field} changed'
                )
    return samples


def _metrics(samples: list[dict[str, Any]], load_case: LoadCase) -> dict[str, Any]:
    output = [float(sample['clock_output_angle_rad']) for sample in samples]
    alarm = [float(sample['alarm_angle_rad']) for sample in samples]
    input_angles = [float(sample['clock_input_angle_rad']) for sample in samples]
    barrel_angles = [float(sample['barrel_input_angle_rad']) for sample in samples]
    setting_angles = [float(sample['setting_input_angle_rad']) for sample in samples]
    alarm_target_angles = [
        float(
            sample.get(
                'alarm_setting_target_angle_rad',
                sample['setting_input_angle_rad'],
            )
        )
        for sample in samples
    ]
    hour = [float(sample['hour_hand_angle_rad']) for sample in samples]
    second = [float(sample['second_hand_angle_rad']) for sample in samples]
    phase_fields = {
        'output': 'clock_output_angle_rad',
        'hour': 'hour_hand_angle_rad',
        'second': 'second_hand_angle_rad',
        'alarm': 'alarm_angle_rad',
        'setting': 'setting_input_angle_rad',
        'alarm_target': 'alarm_setting_target_angle_rad',
    }
    phase_travel: dict[str, dict[str, float]] = {
        label: {} for label in phase_fields
    }
    phase_start: dict[str, dict[str, float]] = {
        label: {} for label in phase_fields
    }
    phase_end: dict[str, dict[str, float]] = {
        label: {} for label in phase_fields
    }
    phase_span: dict[str, dict[str, float]] = {
        label: {} for label in phase_fields
    }
    for phase in load_case.phases:
        phase_samples = [
            sample for sample in samples if sample['phase_id'] == phase.phase_id
        ]
        for label, field in phase_fields.items():
            values = [
                float(
                    sample.get(field, sample['setting_input_angle_rad'])
                )
                for sample in phase_samples
            ]
            if values:
                phase_start[label][phase.phase_id] = values[0]
                phase_end[label][phase.phase_id] = values[-1]
                phase_travel[label][phase.phase_id] = values[-1] - values[0]
                phase_span[label][phase.phase_id] = max(values) - min(values)
    output_span = max(output) - min(output)
    hour_span = max(hour) - min(hour)
    second_span = max(second) - min(second)
    furthest_index = max(
        range(len(output)),
        key=lambda index: abs(output[index] - output[0]),
    )
    minute_delta = output[furthest_index] - output[0]
    hour_delta = hour[furthest_index] - hour[0]
    second_delta = second[furthest_index] - second[0]
    return {
        'min_output_angle_rad': min(output),
        'max_output_angle_rad': max(output),
        'final_output_angle_rad': output[-1],
        'output_span_rad': output_span,
        'min_input_angle_rad': min(input_angles),
        'max_input_angle_rad': max(input_angles),
        'barrel_input_span_rad': max(barrel_angles) - min(barrel_angles),
        'setting_input_span_rad': max(setting_angles) - min(setting_angles),
        'alarm_setting_target_span_rad': (
            max(alarm_target_angles) - min(alarm_target_angles)
        ),
        'hour_hand_span_rad': hour_span,
        'second_hand_span_rad': second_span,
        'hour_to_minute_ratio': hour_span / output_span if output_span > 1.0e-8 else None,
        'second_to_minute_ratio': second_span / output_span if output_span > 1.0e-8 else None,
        'hand_direction_consistent': (
            abs(minute_delta) <= 1.0e-8
            or (minute_delta * hour_delta > 0 and minute_delta * second_delta > 0)
        ),
        'max_setting_clutch_error_rad': max(
            abs(float(item['setting_clutch_error_rad'])) for item in samples
        ),
        'alarm_swing_rad': max(alarm) - min(alarm),
        'max_mesh_error_rad': max(float(item['max_mesh_error_rad']) for item in samples),
        'max_support_drift_m': max(float(item['max_support_drift_m']) for item in samples),
        'max_joint_drift_m': max(float(item['max_support_drift_m']) for item in samples),
        'max_off_axis_rotation_rad': max(
            float(item['max_off_axis_rotation_rad']) for item in samples
        ),
        'max_off_plane_rotation_rad': max(
            float(item['max_off_axis_rotation_rad']) for item in samples
        ),
        'phase_output_travel_rad': phase_travel['output'],
        'phase_hour_travel_rad': phase_travel['hour'],
        'phase_second_travel_rad': phase_travel['second'],
        'phase_alarm_travel_rad': phase_travel['alarm'],
        'phase_setting_travel_rad': phase_travel['setting'],
        'phase_output_start_rad': phase_start['output'],
        'phase_output_end_rad': phase_end['output'],
        'phase_hour_start_rad': phase_start['hour'],
        'phase_hour_end_rad': phase_end['hour'],
        'phase_setting_end_rad': phase_end['setting'],
        'phase_alarm_target_travel_rad': phase_travel['alarm_target'],
        'phase_alarm_target_start_rad': phase_start['alarm_target'],
        'phase_alarm_target_end_rad': phase_end['alarm_target'],
        'phase_alarm_target_span_rad': phase_span['alarm_target'],
        'phase_alarm_span_rad': phase_span['alarm'],
        'contact_observed': any((item.get('contact_count') or 0) > 0 for item in samples),
        'catastrophic_termination': None,
    }


def _semantic_failures(
    plan: MechanismPlan,
    load_case: LoadCase,
    physical: dict[str, Any],
    reduced: dict[str, Any],
) -> list[str]:
    from .family_plugins import mechanism_adapter_for_plan

    plugin, adapter = mechanism_adapter_for_plan(plan)
    spec = clock_gear_train_from_dict(adapter.family_spec)
    expected_links = list(plugin.expected_reduced_ideal_links(plan, adapter))
    failures: list[str] = []
    physical_engine = physical.get('engine', {})
    reduced_engine = reduced.get('engine', {})
    if physical_engine.get('chrono_link_count') not in {0, None}:
        failures.append('physical_internal_ideal_link_present')
    if physical_engine.get('internal_ideal_joint_count') != 0:
        failures.append('physical_internal_ideal_joint_disclosure_invalid')
    if reduced_engine.get('ideal_links') != expected_links:
        failures.append('reduced_clock_topology_mismatch')
    if reduced_engine.get('internal_ideal_joint_count') != len(expected_links):
        failures.append('reduced_clock_link_count_mismatch')
    physical_collision = physical.get('collision_geometry', {})
    reduced_collision = reduced.get('collision_geometry', {})
    if physical_collision.get('contact_enabled') is not True:
        failures.append('physical_STEP_collision_disabled')
    if reduced_collision.get('contact_enabled') is not False:
        failures.append('reduced_collision_not_disabled')
    for label in ('whole_body_box_proxy_used', 'whole_body_convex_hull_used', 'convex_hull_used'):
        if physical_collision.get(label) is not False:
            failures.append(f'physical_{label}')
    physical_semantics = physical.get('branch_semantics', {})
    reduced_semantics = reduced.get('branch_semantics', {})
    for label in (
        'same_mass_kg',
        'same_inertia_xx_kg_m2',
        'same_initial_pose',
        'same_external_load_sha256',
        'family_spec_sha256',
    ):
        if physical_semantics.get(label) != reduced_semantics.get(label):
            failures.append(f'branch_{label}_mismatch')
    if physical_semantics.get('direct_hand_or_hammer_actuation') is not False:
        failures.append('physical_direct_output_actuation_detected')
    expected_direct_hour_actuation = (
        load_case.program == 'hour_hand_alarm_setting_clock'
        or (
            load_case.program == 'scheduled_alarm_clock'
            and spec.alarm_setting_clutch is not None
        )
    )
    if physical_semantics.get(
        'direct_hour_second_or_hammer_actuation'
    ) is not expected_direct_hour_actuation:
        failures.append('physical_direct_three_hand_or_alarm_actuation_detected')
    expected_policy = (
        'external_torque_on_setting_crown_only'
        if load_case.program == 'manual_time_setting_clock'
        else 'calibrate_on_setting_crown_then_release_and_run_time_barrel'
        if load_case.program == 'manual_zero_and_resume_clock'
        else 'turn_hour_hand_to_latch_alarm_target_only'
        if load_case.program == 'hour_hand_alarm_setting_clock'
        else 'turn_hour_hand_to_latch_target_then_run_time_barrel_and_release_independent_alarm_at_selected_position'
        if (
            load_case.program == 'scheduled_alarm_clock'
            and spec.alarm_setting_clutch is not None
        )
        else 'set_alarm_on_setting_crown_then_run_time_barrel_and_trigger_independent_alarm_barrel'
        if load_case.program == 'scheduled_alarm_clock'
        else 'external_torque_on_independent_alarm_barrel_only'
        if load_case.program == 'alarm_clock_cycle' and spec.alarm_isolated
        else 'external_torque_on_barrel_only'
    )
    if physical_semantics.get('input_actuation_policy') != expected_policy:
        failures.append('physical_clock_input_policy_invalid')
    return failures


def _behavior_failures(
    metrics: dict[str, Any],
    *,
    branch: str,
    load_case: LoadCase,
    spec: Any,
) -> list[str]:
    failures: list[str] = []
    if metrics['max_support_drift_m'] > spec.max_support_drift_m:
        failures.append(f'{branch}_shaft_support_drift_exceeded')
    if metrics['max_off_axis_rotation_rad'] > spec.max_off_axis_rotation_rad:
        failures.append(f'{branch}_off_axis_rotation_exceeded')
    mesh_limit = spec.max_mesh_error_rad if branch == 'physical' else 0.005
    if metrics['max_mesh_error_rad'] > mesh_limit:
        failures.append(f'{branch}_gear_mesh_error_exceeded')
    if load_case.program == 'zero_input_clock_train':
        if metrics['output_span_rad'] > spec.output_angle_tolerance_rad:
            failures.append(f'{branch}_zero_input_display_drift_exceeded')
        return failures
    if load_case.program == 'transverse_clock_probe':
        if metrics['output_span_rad'] > spec.output_angle_tolerance_rad:
            failures.append(f'{branch}_transverse_probe_changed_display')
        return failures
    if load_case.program == 'alarm_clock_cycle' and spec.alarm_isolated:
        if metrics['output_span_rad'] > spec.output_angle_tolerance_rad:
            failures.append(f'{branch}_alarm_changed_three_hand_display')
        if metrics['alarm_swing_rad'] < spec.minimum_alarm_swing_rad:
            failures.append(f'{branch}_alarm_hammer_swing_too_small')
        return failures
    if load_case.program == 'hour_hand_alarm_setting_clock':
        scheduled = spec.scheduled_alarm
        if scheduled is None or spec.alarm_setting_clutch is None:
            failures.append(f'{branch}_hour_hand_alarm_setting_spec_missing')
            return failures
        target_end = metrics['phase_alarm_target_end_rad'].get(
            'alarm_setting_latched',
            0.0,
        )
        target_latch_span = metrics['phase_alarm_target_span_rad'].get(
            'alarm_setting_latched',
            math.inf,
        )
        hour_travel = abs(
            metrics['phase_hour_travel_rad'].get(
                'turn_hour_hand_to_alarm_target',
                0.0,
            )
        )
        target_travel = abs(
            metrics['phase_alarm_target_travel_rad'].get(
                'turn_hour_hand_to_alarm_target',
                0.0,
            )
        )
        if abs(
            target_end - scheduled.setting_target_angle_rad
        ) > scheduled.setting_angle_tolerance_rad:
            failures.append(f'{branch}_hour_hand_alarm_target_not_latched')
        if target_latch_span > scheduled.setting_angle_tolerance_rad:
            failures.append(f'{branch}_hour_hand_alarm_target_did_not_hold')
        if hour_travel < 0.5 * scheduled.setting_target_angle_rad:
            failures.append(f'{branch}_hour_hand_did_not_set_alarm_target')
        if target_travel < 0.5 * scheduled.setting_target_angle_rad:
            failures.append(f'{branch}_alarm_target_cam_did_not_follow_hour_hand')
        return failures
    hour_ratio = metrics['hour_to_minute_ratio']
    second_ratio = metrics['second_to_minute_ratio']
    if hour_ratio is None or abs(
        hour_ratio / spec.expected_hour_to_minute_ratio - 1.0
    ) > spec.hand_ratio_relative_tolerance:
        failures.append(f'{branch}_hour_minute_ratio_invalid')
    if second_ratio is None or abs(
        second_ratio / spec.expected_second_to_minute_ratio - 1.0
    ) > spec.hand_ratio_relative_tolerance:
        failures.append(f'{branch}_second_minute_ratio_invalid')
    if not metrics['hand_direction_consistent']:
        failures.append(f'{branch}_hand_direction_inconsistent')
    if load_case.program == 'manual_time_setting_clock':
        if metrics['setting_input_span_rad'] < 0.02:
            failures.append(f'{branch}_setting_crown_did_not_move')
        if metrics['output_span_rad'] < 0.02:
            failures.append(f'{branch}_manual_setting_did_not_move_display')
        return failures
    if load_case.program == 'manual_zero_and_resume_clock':
        calibration_offset = metrics['phase_output_travel_rad'].get(
            'calibration_offset', 0.0
        )
        calibration_start = metrics['phase_output_start_rad'].get(
            'calibration_offset', 0.0
        )
        calibration_end = metrics['phase_output_end_rad'].get(
            'calibration_return', calibration_start
        )
        released_drift = metrics['phase_output_travel_rad'].get(
            'release_setting', 0.0
        )
        post_zero_travel = metrics['phase_output_travel_rad'].get(
            'post_zero_run', 0.0
        )
        if metrics['setting_input_span_rad'] < 0.02 or abs(calibration_offset) < 0.02:
            failures.append(f'{branch}_zero_calibration_knob_did_not_move_display')
        if abs(calibration_end - calibration_start) > spec.output_angle_tolerance_rad:
            failures.append(f'{branch}_zero_calibration_did_not_return_to_dial_zero')
        if abs(released_drift) > spec.output_angle_tolerance_rad:
            failures.append(f'{branch}_released_setting_crown_changed_display')
        if post_zero_travel < 0.02:
            failures.append(f'{branch}_clock_did_not_resume_after_zero_calibration')
        return failures
    if load_case.program == 'scheduled_alarm_clock':
        scheduled = spec.scheduled_alarm
        if scheduled is None:
            failures.append(f'{branch}_scheduled_alarm_spec_missing')
            return failures
        setting_end = (
            metrics['phase_alarm_target_end_rad'].get(
                'alarm_setting_latched',
                0.0,
            )
            if spec.alarm_setting_clutch is not None
            else metrics['phase_setting_end_rad'].get(
                'alarm_setting_latched',
                0.0,
            )
        )
        if abs(setting_end - scheduled.setting_target_angle_rad) > (
            scheduled.setting_angle_tolerance_rad
        ):
            failures.append(f'{branch}_one_minute_alarm_setting_not_latched')
        if spec.alarm_setting_clutch is not None:
            target_drift = abs(
                metrics['phase_alarm_target_travel_rad'].get(
                    'advance_to_alarm_time',
                    0.0,
                )
            )
            trigger_hour_end = metrics['phase_hour_end_rad'].get(
                'advance_to_alarm_time',
                0.0,
            )
            if target_drift > scheduled.setting_angle_tolerance_rad:
                failures.append(f'{branch}_alarm_target_cam_drifted')
            if abs(trigger_hour_end - setting_end) > max(
                scheduled.setting_angle_tolerance_rad,
                0.5 * scheduled.target_hour_hand_travel_rad,
            ):
                failures.append(
                    f'{branch}_hour_hand_did_not_reach_selected_alarm_position'
                )
        second_travel = abs(
            metrics['phase_second_travel_rad'].get('advance_to_alarm_time', 0.0)
        )
        minute_travel = abs(
            metrics['phase_output_travel_rad'].get('advance_to_alarm_time', 0.0)
        )
        hour_travel = abs(
            metrics['phase_hour_travel_rad'].get('advance_to_alarm_time', 0.0)
        )
        target_tolerance = scheduled.target_second_hand_tolerance_rad
        if abs(second_travel - scheduled.target_second_hand_travel_rad) > target_tolerance:
            failures.append(f'{branch}_alarm_did_not_trigger_after_one_displayed_minute')
        if abs(minute_travel - scheduled.target_minute_hand_travel_rad) > (
            spec.output_angle_tolerance_rad
        ):
            failures.append(f'{branch}_minute_hand_travel_at_alarm_invalid')
        if abs(hour_travel - scheduled.target_hour_hand_travel_rad) > max(
            0.01, spec.output_angle_tolerance_rad / 12.0
        ):
            failures.append(f'{branch}_hour_hand_travel_at_alarm_invalid')
        pretrigger_alarm_span = max(
            metrics['phase_alarm_span_rad'].get('set_alarm_for_one_minute', 0.0),
            metrics['phase_alarm_span_rad'].get('alarm_setting_latched', 0.0),
            metrics['phase_alarm_span_rad'].get('advance_to_alarm_time', 0.0),
        )
        if pretrigger_alarm_span > spec.output_angle_tolerance_rad:
            failures.append(f'{branch}_alarm_rang_before_one_minute')
        if metrics['phase_alarm_span_rad'].get('ring_with_timekeeping', 0.0) < (
            spec.minimum_alarm_swing_rad
        ):
            failures.append(f'{branch}_scheduled_alarm_hammer_swing_too_small')
        if abs(
            metrics['phase_output_travel_rad'].get('ring_with_timekeeping', 0.0)
        ) < scheduled.minimum_post_trigger_time_travel_rad:
            failures.append(f'{branch}_timekeeping_stopped_during_alarm')
        return failures
    outbound = metrics['phase_output_travel_rad'].get('wind_and_advance', 0.0)
    returned = metrics['phase_output_travel_rad'].get('controlled_return', 0.0)
    if outbound < 0.02:
        failures.append(f'{branch}_display_did_not_advance')
    if returned > -0.01:
        failures.append(f'{branch}_display_did_not_return')
    alarm_required = (
        load_case.program == 'alarm_clock_cycle'
        or (load_case.program == 'task_clock_cycle' and not spec.alarm_isolated)
    )
    if alarm_required and metrics['alarm_swing_rad'] < spec.minimum_alarm_swing_rad:
        failures.append(f'{branch}_alarm_hammer_swing_too_small')
    if load_case.program == 'task_clock_cycle':
        if metrics['max_output_angle_rad'] < (
            spec.outbound_output_angle_rad - spec.output_angle_tolerance_rad
        ):
            failures.append(f'{branch}_display_task_target_not_reached')
    return failures


def _clock_requirement_checks(
    trace: Mapping[str, Any],
    spec: Any,
    samples: list[dict[str, Any]],
    *,
    strict_non_claims: Mapping[str, bool],
) -> dict[str, dict[str, Any]]:
    semantics = trace.get('branch_semantics')
    semantics = semantics if isinstance(semantics, Mapping) else {}
    collision = trace.get('collision_geometry')
    collision = collision if isinstance(collision, Mapping) else {}
    wrench = trace.get('wrench_audit')
    wrench = wrench if isinstance(wrench, Mapping) else {}
    collision_bodies = [
        item
        for item in collision.get('bodies', [])
        if isinstance(item, Mapping) and item.get('body_id')
    ]
    expected_body_ids = {body.body_id for body in spec.bodies}
    collision_body_ids = {str(body['body_id']) for body in collision_bodies}
    collision_components_complete = bool(collision_bodies) and all(
        isinstance(body.get('components'), list) and bool(body['components'])
        for body in collision_bodies
    )
    saved_poses_complete = bool(samples) and all(
        isinstance(sample.get('body_poses'), Mapping)
        and expected_body_ids.issubset(sample['body_poses'])
        for sample in samples
    )
    declared_group_ids = [
        group_id for body in spec.bodies for group_id in body.group_ids
    ]
    single_source_passed = bool(
        collision.get('mode')
        == 'intake-bound STEP per-solid body-local collision components'
        and collision.get('package_binding_sha256')
        == trace.get('package_binding_sha256')
        and isinstance(collision.get('assembly_sha256'), str)
        and len(str(collision['assembly_sha256'])) == 64
        and isinstance(collision.get('assembly_contract_sha256'), str)
        and len(str(collision['assembly_contract_sha256'])) == 64
        and collision_components_complete
        and collision.get('whole_body_box_proxy_used') is False
        and collision.get('whole_body_convex_hull_used') is False
        and collision.get('convex_hull_used') is False
    )
    ownership_passed = bool(
        collision_body_ids == expected_body_ids
        and len(collision_bodies) == len(expected_body_ids)
        and collision_components_complete
        and len(set(declared_group_ids)) == len(declared_group_ids)
        and saved_poses_complete
    )

    edges: dict[str, set[str]] = {body_id: set() for body_id in expected_body_ids}
    for connection in (*spec.gear_meshes, *spec.shaft_transfers):
        driver = connection.driver_body_id
        driven = connection.driven_body_id
        edges.setdefault(driver, set()).add(driven)
        edges.setdefault(driven, set()).add(driver)

    def reachable(start: str) -> set[str]:
        observed: set[str] = set()
        pending = [start]
        while pending:
            body_id = pending.pop()
            if body_id in observed:
                continue
            observed.add(body_id)
            pending.extend(edges.get(body_id, set()) - observed)
        return observed

    time_path = reachable(spec.input_body_id)
    alarm_path = reachable(spec.effective_alarm_input_body_id)
    follower_path_passed = bool(spec.cam_followers) and all(
        follower.cam_body_id in alarm_path
        and follower.follower_body_id == spec.alarm_body_id
        for follower in spec.cam_followers
    )
    setting_path_passed = bool(
        spec.setting_clutch is not None
        and spec.setting_clutch.input_body_id == spec.setting_input_body_id
        and spec.setting_clutch.driven_body_id in time_path
    )
    topology_passed = bool(
        spec.alarm_isolated
        and spec.effective_alarm_input_body_id not in time_path
        and {spec.output_body_id, spec.hour_body_id, spec.second_body_id}
        <= time_path
        and follower_path_passed
        and setting_path_passed
        and semantics.get('alarm_train_isolated') is True
        and semantics.get('direct_hand_or_hammer_actuation') is False
        and semantics.get('direct_output_shaft_actuation') is False
        and wrench.get('passed') is True
        and wrench.get('direct_output_drive_detected') is False
    )
    boundary_passed = all(
        strict_non_claims.get(field) is False
        for field in (
            'real_time_chronometric_accuracy_certified',
            'spring_endurance_certified',
            'wear_or_fatigue_certified',
            'manufacturability_certified',
            'consumer_safety_certified',
        )
    )

    def check(passed: bool, *roles: str) -> dict[str, Any]:
        return {'passed': passed, 'evidence_roles': list(roles)}

    return {
        'validator.mechanical.single_source_cad': check(
            single_source_passed,
            'validation.assembly_contract',
            'validation.original_step_collision_geometry',
        ),
        'validator.mechanical.rigid_group_ownership': check(
            ownership_passed,
            'validation.assembly_contract',
            'validation.body_pose_trace',
        ),
        'validator.clock.independent_alarm_power_paths': check(
            topology_passed,
            'validation.wrench_audit',
            'validation.branch_semantics',
        ),
        'policy.clock.validation_boundaries': check(
            boundary_passed,
            'validation.terminal_verdict',
        ),
    }


def compare_clock_gear_trace_pair(
    plan: MechanismPlan,
    load_case: LoadCase,
    physical_trace: dict[str, Any],
    reduced_trace: dict[str, Any],
) -> dict[str, Any]:
    from .adapters import adapter_from_dict, get_adapter

    adapter = adapter_from_dict(plan.adapter_spec) if plan.adapter_spec else get_adapter(plan.adapter_id)
    spec = clock_gear_train_from_dict(adapter.family_spec)
    thresholds = {
        'max_support_drift_m': spec.max_support_drift_m,
        'max_off_axis_rotation_rad': spec.max_off_axis_rotation_rad,
        'max_mesh_error_rad': spec.max_mesh_error_rad,
        'minimum_alarm_swing_rad': spec.minimum_alarm_swing_rad,
        'expected_hour_to_minute_ratio': spec.expected_hour_to_minute_ratio,
        'expected_second_to_minute_ratio': spec.expected_second_to_minute_ratio,
        'hand_ratio_relative_tolerance': spec.hand_ratio_relative_tolerance,
        'output_angle_tolerance_rad': spec.output_angle_tolerance_rad,
        'home_output_angle_rad': spec.home_output_angle_rad,
        'outbound_output_angle_rad': spec.outbound_output_angle_rad,
        'source': 'validated_clock_gear_family_spec',
    }
    try:
        physical_samples = _validate_trace(
            physical_trace, branch='physical', plan=plan, load_case=load_case, spec=spec
        )
        reduced_samples = _validate_trace(
            reduced_trace, branch='reduced', plan=plan, load_case=load_case, spec=spec
        )
    except MechanismValidationError as exc:
        return {
            'schema_version': '0.1',
            'capsule_id': plan.capsule_id,
            'load_case_id': load_case.load_case_id,
            'purpose': load_case.purpose,
            'program': load_case.program,
            'required_for_certificate': load_case.required_for_certificate,
            'verdict': 'PHYSICS_INCONCLUSIVE',
            'failure_reasons': [exc.code],
            'detail': str(exc),
            'thresholds': thresholds,
        }
    physical_metrics = _metrics(physical_samples, load_case)
    reduced_metrics = _metrics(reduced_samples, load_case)
    physical_metrics['catastrophic_termination'] = physical_trace.get('catastrophic_termination')
    semantics = _semantic_failures(plan, load_case, physical_trace, reduced_trace)
    physical_failures = _behavior_failures(
        physical_metrics, branch='physical', load_case=load_case, spec=spec
    )
    reduced_failures = _behavior_failures(
        reduced_metrics, branch='reduced', load_case=load_case, spec=spec
    )
    if physical_trace.get('catastrophic_termination') is not None:
        physical_failures.append('physical_catastrophic_termination')
    range_delta = abs(physical_metrics['output_span_rad'] - reduced_metrics['output_span_rad'])
    cross_branch_limit = (
        max(0.20, 0.60 * spec.outbound_output_angle_rad)
        if load_case.program == 'manual_time_setting_clock'
        else max(0.12, 0.45 * spec.outbound_output_angle_rad)
    )
    if load_case.program not in {'zero_input_clock_train', 'transverse_clock_probe'} and range_delta > cross_branch_limit:
        reduced_failures.append('physical_reduced_display_range_mismatch')
    verdict = (
        'PHYSICS_INCONCLUSIVE' if semantics
        else 'PHYSICAL_FAILURE' if physical_failures
        else 'CLAIM_MISMATCH' if reduced_failures
        else 'PASS'
    )
    strict_non_claims = {
        'pointwise_animation_match_used_as_gate': False,
        'exact_tooth_impact_or_backlash_certified': False,
        'escapement_accuracy_or_alarm_acoustics_certified': False,
        'friction_or_efficiency_certified': False,
        'dynamic_response_certified': False,
        'real_time_chronometric_accuracy_certified': False,
        'spring_endurance_certified': False,
        'wear_or_fatigue_certified': False,
        'manufacturability_certified': False,
        'consumer_safety_certified': False,
    }
    return {
        'schema_version': '0.1',
        'capsule_id': plan.capsule_id,
        'load_case_id': load_case.load_case_id,
        'purpose': load_case.purpose,
        'program': load_case.program,
        'required_for_certificate': load_case.required_for_certificate,
        'verdict': verdict,
        'failure_reasons': [*semantics, *physical_failures, *reduced_failures],
        'thresholds': thresholds,
        'physical_metrics': physical_metrics,
        'reduced_metrics': reduced_metrics,
        'cross_branch_metrics': {'display_range_delta_rad': range_delta},
        'strict_non_claims': strict_non_claims,
        'requirement_checks': _clock_requirement_checks(
            physical_trace,
            spec,
            physical_samples,
            strict_non_claims=strict_non_claims,
        ),
    }
