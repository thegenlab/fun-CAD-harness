from __future__ import annotations

from typing import Any

from .adapters import AdapterSpec
from .clock_gear_train_spec import clock_gear_train_from_dict
from .io import canonical_sha256
from .models import LoadCase


def clock_gear_load_envelope(
    load_cases: tuple[LoadCase, ...],
    comparisons: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    masses = [item.moving_mass_kg for item in load_cases]
    torques = [
        phase.magnitude_si
        for item in load_cases
        for phase in item.phases
        if phase.wrench_kind == 'torque'
    ]
    metrics = [
        item['physical_metrics']
        for item in comparisons
        if isinstance(item.get('physical_metrics'), dict)
    ]
    angles = [
        value
        for item in metrics
        for value in (item['min_output_angle_rad'], item['max_output_angle_rad'])
    ]
    return {
        'moving_mass_kg': {'min': min(masses), 'max': max(masses)},
        'load_case_ids': [item.load_case_id for item in load_cases],
        'purposes': sorted({item.purpose for item in load_cases}),
        'gravity_abs_m_s2': None,
        'applied_torque_nm': {
            'min_nonzero': min(torques) if torques else 0.0,
            'max': max(torques) if torques else 0.0,
        },
        'generalized_output_angle_rad': (
            {
                'min': min(angles),
                'max': max(angles),
                'source': 'observed_physical_clock_display_angle',
            }
            if angles else None
        ),
        'max_observed_shaft_support_drift_m': (
            max(item['max_support_drift_m'] for item in metrics) if metrics else None
        ),
        'max_observed_off_axis_rotation_rad': (
            max(item['max_off_axis_rotation_rad'] for item in metrics) if metrics else None
        ),
        'max_observed_mesh_error_rad': (
            max(item['max_mesh_error_rad'] for item in metrics) if metrics else None
        ),
        'min_observed_alarm_swing_rad': (
            min(item['alarm_swing_rad'] for item in metrics) if metrics else None
        ),
    }


def clock_gear_certificate_model(adapter: AdapterSpec) -> dict[str, Any]:
    spec = clock_gear_train_from_dict(adapter.family_spec)
    return {
        'adapter_id': adapter.adapter_id,
        'train_id': spec.train_id,
        'mechanism_kind': spec.mechanism_kind,
        'topology_sha256': canonical_sha256(spec.to_dict()),
        'body_count': len(spec.bodies),
        'shaft_count': len(spec.shafts),
        'gear_mesh_count': len(spec.gear_meshes),
        'shaft_transfer_count': len(spec.shaft_transfers),
        'cam_follower_count': len(spec.cam_followers),
        'input_body_id': spec.input_body_id,
        'setting_input_body_id': spec.setting_input_body_id,
        'output_body_id': spec.output_body_id,
        'hour_body_id': spec.hour_body_id,
        'second_body_id': spec.second_body_id,
        'alarm_body_id': spec.alarm_body_id,
        'alarm_input_body_id': spec.effective_alarm_input_body_id,
        'alarm_train_isolated': spec.alarm_isolated,
        'expected_hand_ratios': {
            'hour_to_minute': spec.expected_hour_to_minute_ratio,
            'second_to_minute': spec.expected_second_to_minute_ratio,
            'relative_tolerance': spec.hand_ratio_relative_tolerance,
        },
        'physical_support': 'finite_ChLoadBodyBodyBushingGeneric',
        'physical_gear_mesh': 'finite_compliant_mesh_torque',
        'physical_seconds_transfer': 'finite_torsional_shaft_transfer',
        'reduced_support': 'declared_ChLinkLockRevolute_topology',
        'reduced_gear_mesh': 'declared_ChLinkLockGear_topology',
        'cam_followers': 'finite_analytical_follower_torque',
        'manual_setting_clutch': (
            None
            if spec.setting_clutch is None
            else {
                'clutch_id': spec.setting_clutch.clutch_id,
                'model': 'finite_torque_limited_pull_to_engage_clutch',
                'max_torque_nm': spec.setting_clutch.max_torque_nm,
            }
        ),
        'alarm_setting_clutch': (
            None
            if spec.alarm_setting_clutch is None
            else {
                'clutch_id': spec.alarm_setting_clutch.clutch_id,
                'model': 'finite_hour_hand_to_alarm_target_clutch',
                'max_torque_nm': (
                    spec.alarm_setting_clutch.max_torque_nm
                ),
            }
        ),
        'external_actuation': {
            'normal_run': 'time_barrel_input_torque_only',
            'alarm_cycle': (
                'independent_alarm_barrel_input_torque_only'
                if spec.alarm_isolated
                else 'shared_time_barrel_input_torque_only'
            ),
            'manual_time_setting': 'setting_crown_torque_only',
            'zero_and_resume': (
                'setting_crown_then_release_then_time_barrel'
                if spec.scheduled_alarm is not None
                else None
            ),
            'scheduled_alarm': (
                'turn_hour_hand_to_latch_target_then_run_time_barrel_until_selected_position'
                if spec.alarm_setting_clutch is not None
                else 'set_target_on_setting_crown_then_time_barrel_with_triggered_alarm_barrel'
                if spec.scheduled_alarm is not None
                else None
            ),
        },
        'scheduled_alarm': (
            None
            if spec.scheduled_alarm is None
            else {
                'target_time_seconds': spec.scheduled_alarm.target_time_seconds,
                'target_second_hand_travel_rad': (
                    spec.scheduled_alarm.target_second_hand_travel_rad
                ),
                'target_second_hand_tolerance_rad': (
                    spec.scheduled_alarm.target_second_hand_tolerance_rad
                ),
                'target_minute_hand_travel_rad': (
                    spec.scheduled_alarm.target_minute_hand_travel_rad
                ),
                'target_hour_hand_travel_rad': (
                    spec.scheduled_alarm.target_hour_hand_travel_rad
                ),
                'ringing_must_preserve_timekeeping': True,
            }
        ),
        'direct_hand_or_hammer_actuation': False,
        'exact_tooth_impact_or_backlash_certified': False,
    }


def clock_gear_replacement_topology(adapter: AdapterSpec) -> dict[str, Any]:
    spec = clock_gear_train_from_dict(adapter.family_spec)
    return {
        'train_id': spec.train_id,
        'mechanism_kind': spec.mechanism_kind,
        'family_spec_sha256': canonical_sha256(spec.to_dict()),
        'body_ids': [item.body_id for item in spec.bodies],
        'shaft_ids': [item.shaft_id for item in spec.shafts],
        'revolute_links': [
            {'shaft_id': item.shaft_id, 'body_id': item.body_id}
            for item in spec.shafts
        ],
        'gear_links': [
            {
                'mesh_id': item.mesh_id,
                'driver_body_id': item.driver_body_id,
                'driven_body_id': item.driven_body_id,
                'ratio': item.ratio,
            }
            for item in spec.gear_meshes
        ],
        'shaft_transfers': [
            {
                'transfer_id': item.transfer_id,
                'driver_body_id': item.driver_body_id,
                'driven_body_id': item.driven_body_id,
                'ratio': item.ratio,
                'mode': 'finite_torsional_transfer',
            }
            for item in spec.shaft_transfers
        ],
        'cam_followers': [
            {
                'follower_id': item.follower_id,
                'cam_body_id': item.cam_body_id,
                'follower_body_id': item.follower_body_id,
                'lobe_count': item.lobe_count,
            }
            for item in spec.cam_followers
        ],
        'setting_clutch': (
            None
            if spec.setting_clutch is None
            else {
                'clutch_id': spec.setting_clutch.clutch_id,
                'input_body_id': spec.setting_clutch.input_body_id,
                'driven_body_id': spec.setting_clutch.driven_body_id,
                'ratio': spec.setting_clutch.ratio,
                'mode': 'finite_selectable_clutch',
            }
        ),
        'alarm_setting_clutch': (
            None
            if spec.alarm_setting_clutch is None
            else {
                'clutch_id': spec.alarm_setting_clutch.clutch_id,
                'input_body_id': (
                    spec.alarm_setting_clutch.input_body_id
                ),
                'driven_body_id': (
                    spec.alarm_setting_clutch.driven_body_id
                ),
                'ratio': spec.alarm_setting_clutch.ratio,
                'mode': 'finite_selectable_alarm_target_clutch',
            }
        ),
        'input_body_id': spec.input_body_id,
        'setting_input_body_id': spec.setting_input_body_id,
        'output_body_id': spec.output_body_id,
        'hour_body_id': spec.hour_body_id,
        'second_body_id': spec.second_body_id,
        'alarm_body_id': spec.alarm_body_id,
        'alarm_input_body_id': spec.effective_alarm_input_body_id,
        'alarm_train_isolated': spec.alarm_isolated,
        'scheduled_alarm': (
            None
            if spec.scheduled_alarm is None
            else {
                'setting_body_id': spec.scheduled_alarm.setting_body_id,
                'target_time_seconds': spec.scheduled_alarm.target_time_seconds,
                'trigger_by_display_time': True,
            }
        ),
        'output_angle_range_rad': [
            spec.home_output_angle_rad,
            spec.outbound_output_angle_rad,
        ],
    }
