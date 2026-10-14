"""Harness-owned clock measurements, using complete solver records and explicit profiles.

This module never imports author test code, starts physics, or assigns a task
PASS. Its numerical algorithms retain the reviewed D r09 event/gap rules.
The caller must supply independently calibrated impulse/pre-impact semantics;
structural checks here cannot certify a contact collector's physical meaning.
"""
from __future__ import annotations

import hashlib
import copy
import json
import math
from pathlib import Path
from collections.abc import Iterable, Mapping
import numpy as np


class MeasurementCapabilityError(RuntimeError):
    """Required records or measurement semantics are unavailable."""


PROFILES = {
    'independent_alarm_setting': {'duration_s':75., 'step_s':.001},
    'automatic_alarm_and_stop': {'duration_s':195., 'step_s':.0005},
}
CHANNELS = (
    ('escape', 'escape_bearing.coordinate', math.pi/60),
    ('center', 'center_bearing.coordinate', math.pi/3600),
    ('minute', 'cannon_minute.angle.z_rad', math.pi/3600),
    ('hour', 'hour_sleeve.angle.z_rad', math.pi/43200),
)
PROFILE_SCHEMA = 'slim_clock_measurement_profile_v1'
# These are the two explicitly retained historical v2 measurement definitions.
# Their old joint signs are preserved for historical evidence, not reused as
# defaults for a new v3 source plan.
LEGACY_MEASUREMENT_PROFILES = {
    test_id: {
        'schema_version': PROFILE_SCHEMA, **timing,
        'channels': {name: {'signal': column, 'scale': 1,
                           'relative_to': 'frame.angle.z_rad' if name in ('minute', 'hour') else None}
                     for name, column, _increment in CHANNELS},
        'reference_window_s': [30., 60.], 'minimum_reference_intervals': 40,
        'operation_windows_s': {'alarm_setting': [25., 27.]} if test_id == 'independent_alarm_setting'
                               else {'OFF': [165., 166.]},
        **({} if test_id == 'independent_alarm_setting' else {
            'reader_reference_window_s': [30., 60.], 'release_search_window_s': [60., 165.],
            'armed_window_s': [10., 165.], 'ringing_window_s': [140., 160.],
            'before_off_window_s': [160., 164.8], 'after_off_window_s': [175., 190.],
            'phase_initial_rad': math.radians(1)}),
    } for test_id, timing in PROFILES.items()
}


def validate_measurement_profile(profile, test_id, test=None):
    """Validate only a bounded, reviewed data profile; never execute a source plan."""
    common = {'schema_version', 'duration_s', 'step_s', 'channels', 'reference_window_s',
              'minimum_reference_intervals', 'operation_windows_s'}
    alarm = {'reader_reference_window_s', 'release_search_window_s', 'armed_window_s',
             'ringing_window_s', 'before_off_window_s', 'after_off_window_s', 'phase_initial_rad',
             'release_boundary_phase_rad'}
    if test_id not in PROFILES or not isinstance(profile, Mapping):
        raise MeasurementCapabilityError('Unknown clock measurement/profile')
    expected = common | (alarm if test_id == 'automatic_alarm_and_stop' else set())
    if set(profile) != expected or profile['schema_version'] != PROFILE_SCHEMA:
        raise MeasurementCapabilityError('Explicit complete clock measurement profile is required')

    def finite(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)

    duration, step = profile['duration_s'], profile['step_s']
    if not finite(duration) or not finite(step) or not 0 < step <= duration:
        raise MeasurementCapabilityError('Invalid measurement duration/step')

    def window(value):
        if (not isinstance(value, (list, tuple)) or len(value) != 2
                or not all(finite(v) for v in value) or not 0 <= value[0] < value[1] <= duration
                or value[1]-value[0] < step):
            raise MeasurementCapabilityError('Measurement window must be nonempty and inside the full run')
        return value

    baseline = window(profile['reference_window_s'])
    intervals = profile['minimum_reference_intervals']
    if (not isinstance(intervals, int) or isinstance(intervals, bool) or intervals < 2
            or intervals >= math.ceil((baseline[1]-baseline[0])/step)):
        raise MeasurementCapabilityError('Reference needs a finite reviewed minimum of at least two intervals')
    permitted = {
        'escape': {('escape_bearing.coordinate', None), ('escape_arbor.angle.z_rad', 'frame.angle.z_rad')},
        'center': {('center_bearing.coordinate', None), ('center_arbor.angle.z_rad', 'frame.angle.z_rad')},
        'minute': {('cannon_minute.angle.z_rad', 'frame.angle.z_rad')},
        'hour': {('hour_sleeve.angle.z_rad', 'frame.angle.z_rad')},
    }
    channels = profile['channels']
    if not isinstance(channels, Mapping) or set(channels) != set(permitted):
        raise MeasurementCapabilityError('All four physical clock channels must be bound')
    for name, spec in channels.items():
        if (not isinstance(spec, Mapping) or set(spec) != {'signal', 'relative_to', 'scale'}
                or not isinstance(spec['signal'], str)
                or (spec['relative_to'] is not None and not isinstance(spec['relative_to'], str))
                or (spec['signal'], spec['relative_to']) not in permitted[name]
                or not finite(spec['scale']) or spec['scale'] not in (-1, 1)):
            raise MeasurementCapabilityError('Channel must use a permitted raw signal, reference and signed unit scale')
    operation = 'alarm_setting' if test_id == 'independent_alarm_setting' else 'OFF'
    operations = profile['operation_windows_s']
    if not isinstance(operations, Mapping) or set(operations) != {operation}:
        raise MeasurementCapabilityError('Explicit actual operation window is required')
    actual_window = window(operations[operation])
    if test_id == 'automatic_alarm_and_stop':
        for key in alarm - {'phase_initial_rad', 'release_boundary_phase_rad'}:
            window(profile[key])
        if not finite(profile['phase_initial_rad']):
            raise MeasurementCapabilityError('Actual initial hand/alarm phase must be bound')
        if not finite(profile['release_boundary_phase_rad']) or profile['release_boundary_phase_rad'] >= 0:
            raise MeasurementCapabilityError('An explicit negative source-clearance boundary must be bound')
        reader_ref = profile['reader_reference_window_s']
        search = profile['release_search_window_s']
        armed = profile['armed_window_s']
        ringing = profile['ringing_window_s']
        before = profile['before_off_window_s']
        after = profile['after_off_window_s']
        if (reader_ref[1] > search[0] or search[1] > actual_window[0]
                or armed[1] != actual_window[0] or armed[0] > search[0]
                or ringing[0] < armed[0] or ringing[1] > actual_window[0]
                or before[0] < armed[0] or before[1] > actual_window[0]
                or after[0] < actual_window[1]):
            raise MeasurementCapabilityError('Release/ringing/OFF windows do not follow the actual operation order')
    if test is not None:
        if (test.get('id') != test_id or test.get('duration_s') != duration or test.get('step_s') != step):
            raise MeasurementCapabilityError('Measurement profile changes the exact source test grid')
        joint = 'set_alarm_time' if operation == 'alarm_setting' else 'alarm_switch'
        inputs = test.get('inputs', [])
        passive_windows = [profile['reference_window_s']]
        if operation == 'OFF':
            passive_windows.append(profile['reader_reference_window_s'])
        if any(max(start, drive.get('start_s', 0)) < min(end, drive.get('end_s', duration))
               for start, end in passive_windows for drive in inputs):
            raise MeasurementCapabilityError('Passive reference window overlaps an actual source input')
        matches = [drive for drive in inputs if drive.get('joint') == joint
                   and [drive.get('start_s'), drive.get('end_s')] == list(actual_window)]
        if not matches:
            raise MeasurementCapabilityError('Measurement operation has no matching actual source input')
        if operation == 'OFF':
            if not any(drive.get('mode') == 'position' and drive.get('value', 0) == 0 for drive in matches):
                raise MeasurementCapabilityError('OFF window must bind the actual OFF position input')
            if not any(drive.get('joint') == 'alarm_switch' and drive.get('mode') == 'position'
                       and drive.get('value', 0) != 0 and drive.get('start_s') == profile['armed_window_s'][0]
                       and drive.get('end_s', duration+1) <= profile['release_search_window_s'][0]
                       for drive in inputs):
                raise MeasurementCapabilityError('ARM window must start at the actual ARM input')
    return profile


def measurement_profile_for_registration(registration, test=None):
    """v3 has no profile inference; v2 retains only its explicit old definition."""
    test_id = registration.get('test_id')
    if registration.get('schema_version') == 'slim_alarm_measurement_binding_v3':
        profile = validate_measurement_profile(registration.get('measurement_profile'), test_id, test)
        policy = registration.get('measurement_policy_bindings')
        if not isinstance(policy, Mapping) or not policy:
            raise MeasurementCapabilityError('An independently reviewed policy binding is required')
        plans = registration.get('source_plan_bindings', {})
        for field in ('source_plan', 'source_requirements'):
            reference = registration.get(field)
            if not isinstance(reference, Mapping) or reference.get('path') not in plans:
                raise MeasurementCapabilityError('Reviewed source plan and requirements must both be explicitly bound')
    elif registration.get('schema_version') == 'slim_alarm_measurement_binding_v2' and test_id in LEGACY_MEASUREMENT_PROFILES:
        profile = copy.deepcopy(LEGACY_MEASUREMENT_PROFILES[test_id])
    else:
        raise MeasurementCapabilityError('No explicit reviewed clock measurement registration')
    if any(registration.get(field) != profile[field] for field in ('duration_s', 'step_s')):
        raise MeasurementCapabilityError('Registration and measurement profile grids differ')
    return profile


def _channel_angle(sig, profile, name):
    spec = profile['channels'][name]
    angle = sig(spec['signal'])
    if spec['relative_to'] is not None:
        angle = angle - sig(spec['relative_to'])
    return spec['scale'] * angle
BELLS = ('left_bell', 'right_bell')
CONTACT_SCHEMA = {
    'sampling':'every_solver_step',
    'time_basis':'solver_step_end',
    'normal_impulse_unit':'N*s',
    'closing_speed_unit':'m/s',
    'closing_speed_timing':'before_same_contact_impulse',
    'normal_impulse_sign':'compression_positive',
    'closing_speed_sign':'approach_positive',
}


def implementation_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def registration_signature(registration):
    value={key:value for key,value in registration.items() if key!='signature'}
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode('utf-8')).hexdigest()


def _column(signals, name, shape=None):
    try:
        value = np.asarray(signals[name], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise MeasurementCapabilityError('Missing numeric solver column: '+name) from exc
    if value.ndim != 1 or (shape is not None and value.shape != shape) or not np.all(np.isfinite(value)):
        raise MeasurementCapabilityError('Incomplete/nonfinite solver column: '+name)
    return value


def _signals(signals, profile):
    t = _column(signals, 'time_s')
    duration, requested_step = profile['duration_s'], profile['step_s']
    steps = math.ceil(duration/requested_step)
    expected = np.arange(steps+1, dtype=float)*(duration/steps)
    # Require the actual complete run, not sparse replay frames or a clipped
    # interval whose absence of contact could be mistaken for silence.
    if t.shape != expected.shape or not np.allclose(t, expected, atol=1e-9, rtol=0):
        raise MeasurementCapabilityError(f'Complete 0..{duration:g} s solver-step grid is required')
    return t, float(np.max(np.diff(t))), lambda name:_column(signals, name, t.shape)


def advance_times(time, angle, advance):
    """Original forward-envelope half-beat rule; reversals do not count twice."""
    forward = np.maximum.accumulate(-(angle-angle[0]))
    levels = np.floor(forward/advance).astype(np.int64)
    changes = np.diff(levels)
    if np.any(changes > 1):
        raise MeasurementCapabilityError('More than one half-beat level crossed in a solver step')
    return time[1:][changes == 1]


def bracketing_max_gap(events, start, end):
    if len(events) < 2 or events[0] > start or events[-1] < end:
        return math.inf
    selected = np.diff(events)[(events[:-1] <= end) & (events[1:] >= start)]
    return float(np.max(selected)) if len(selected) else math.inf


def _continuity(t, dt, sig, operations, profile):
    esc = _channel_angle(sig, profile, 'escape')
    beats = advance_times(t, esc, math.pi/60)
    reference_start, reference_end = profile['reference_window_s']
    gaps = np.diff(beats[(beats >= reference_start) & (beats <= reference_end)])
    if len(gaps) < profile['minimum_reference_intervals']:
        return {'F5_sustained_reference':False}, {'reference_gaps':int(len(gaps))}
    beat = float(np.median(gaps))
    maximum = float(np.max(gaps))
    if maximum > 1.5*beat:
        return {'F5_reference_has_no_missed_beat':False}, {
            'reference_half_period_s':beat, 'reference_maximum_gap_s':maximum}
    limit = max(maximum, 1.5*beat)+2*dt
    assertions, measurements = {}, {'reference_half_period_s':beat, 'maximum_allowed_gap_s':limit}
    for name, _column_name, increment in CHANNELS:
        angle = _channel_angle(sig, profile, name)
        events = advance_times(t, angle, increment)
        for operation, (start, end) in operations.items():
            gap = bracketing_max_gap(events, start-4*beat, end+4*beat)
            assertions[f'F5_{name}_maximum_gap_at_{operation}'] = gap <= limit
            key = name+'_maximum_gap_s' if operation == 'release' else f'{name}_maximum_gap_at_{operation}_s'
            measurements[key] = gap if math.isfinite(gap) else None
    return assertions, measurements


def _strikes(t, contact_steps: Iterable[Mapping], contact_schema):
    """Count a manifold/persistent contact once; require explicit empty steps."""
    if not isinstance(contact_schema, Mapping) or any(contact_schema.get(k) != v for k,v in CONTACT_SCHEMA.items()):
        raise MeasurementCapabilityError('Contact impulse, timing, sign or all-step semantics are unbound')
    covered = {frozenset(pair) for pair in contact_schema.get('covered_body_pairs', [])}
    if not {frozenset(('alarm_hammer', bell)) for bell in BELLS} <= covered:
        raise MeasurementCapabilityError('Both hammer/bell pairs must be completely recorded')
    if contact_steps is None:
        raise MeasurementCapabilityError('Missing all-step contact record stream')
    active = {bell:False for bell in BELLS}
    classified = {bell:False for bell in BELLS}
    strikes, seen = [], 0
    for expected_step, row in enumerate(contact_steps, 1):
        if expected_step >= len(t):
            raise MeasurementCapabilityError('Contact stream extends beyond the recorded solver grid')
        if (row.get('step') != expected_step or not isinstance(row.get('contacts'), list)
                or not isinstance(row.get('time_s'), (int,float))
                or not math.isfinite(row['time_s']) or abs(row['time_s']-t[expected_step]) > 1e-9):
            raise MeasurementCapabilityError('Missing, duplicate or misaligned all-step contact row')
        pair_rows = {bell:[] for bell in BELLS}
        for contact in row['contacts']:
            pair = {contact.get('body_a'), contact.get('body_b')}
            bell = next((bell for bell in BELLS if pair == {'alarm_hammer',bell}), None)
            if bell is None: continue
            try:
                impulse = float(contact['normal_impulse_n_s'])
                approach = float(contact['closing_speed_before_impulse_m_s'])
            except (KeyError, TypeError, ValueError) as exc:
                raise MeasurementCapabilityError('Missing impulse or pre-impact closing speed for hammer/bell contact') from exc
            if not math.isfinite(impulse) or not math.isfinite(approach):
                raise MeasurementCapabilityError('Nonfinite hammer/bell impulse or closing speed')
            pair_rows[bell].append((impulse, approach))
        for bell, points in pair_rows.items():
            if not points:
                active[bell] = classified[bell] = False
                continue
            if not active[bell]: classified[bell] = False
            active[bell] = True
            # Classify at the first positive load in this contact episode.
            # A support episode cannot become an impact merely because a later
            # solver residual is positive; explicit separation starts a new one.
            if not classified[bell] and any(impulse > 0 for impulse,_approach in points):
                classified[bell] = True
                if any(impulse > 0 and approach > 0 for impulse,approach in points):
                    strikes.append((expected_step, float(t[expected_step]), bell))
        seen = expected_step
    if seen != len(t)-1:
        raise MeasurementCapabilityError('Contact stream is truncated; missing rows are not zero impacts')
    return sorted(strikes, key=lambda strike:(strike[1],strike[2]))


def _result(test_id, assertions, measurements):
    if not assertions:
        raise MeasurementCapabilityError('Registered measurement returned no numerical assertions')
    return {'schema_version':'slim_alarm_measurement_v2', 'test_id':test_id,
            'assertions':[{'id':name, 'criterion_ids':[name.split('_',1)[0]], 'passed':bool(value)}
                          for name,value in assertions.items()],
            'measurements':measurements, 'task_verdict':None,
            'authority':'Numerical measurements only; solver validity and independent calibration/review remain caller-owned'}


def evaluate_clock_measurement(test_id, signals, contact_steps=None, contact_schema=None, *, measurement_profile, legacy_v2=False):
    """Evaluate an explicit reviewed clock profile; no generic ID/parameter guessing.

    ``signals`` may contain NumPy memmap columns. ``contact_steps`` is a bounded
    iterator with one row per completed solver step, including empty contacts.
    It is not the 720-frame visualization/force-evidence channel.
    """
    if legacy_v2:
        # Only the caller's explicit historical v2 registration selects this
        # branch. Missing data in a v3 profile never invokes a legacy fallback.
        if measurement_profile != LEGACY_MEASUREMENT_PROFILES.get(test_id):
            raise MeasurementCapabilityError('Historical v2 evaluation requires its exact original profile')
        profile = measurement_profile
    else:
        profile = validate_measurement_profile(measurement_profile, test_id)
    t, dt, sig = _signals(signals, profile)
    # Validate the required columns even when an earlier prerequisite fails.
    for name in profile['channels']: _channel_angle(sig, profile, name)
    if test_id == 'independent_alarm_setting':
        assertions, measurements = _continuity(t, dt, sig, profile['operation_windows_s'], profile)
        return _result(test_id, assertions, measurements)
    reader = sig('trigger_slide.coordinate')
    phase = profile['phase_initial_rad']+sig('hour_sleeve.angle.z_rad')-sig('alarm_setting_ring.angle.z_rad')
    strikes = _strikes(t, contact_steps, contact_schema)
    reference_start, reference_end = profile['reader_reference_window_s']
    baseline = (t >= reference_start) & (t <= reference_end)
    search_start, search_end = profile['release_search_window_s']
    release_indices = np.flatnonzero((t >= search_start) & (t < search_end) & (reader-np.mean(reader[baseline]) >= .0006))
    if not len(release_indices):
        return _result(test_id, {'F3_measured_automatic_release':False}, {'release_s':None})
    release = float(t[release_indices[0]])
    continuity, measurements = _continuity(t, dt, sig,
        {'release':(release,release), **profile['operation_windows_s']}, profile)
    measurements['release_s'] = release
    if not legacy_v2:
        boundary = profile['release_boundary_phase_rad']
        omega = 2*math.pi/43200
        phase_allowance = 2*dt*omega
        release_phase = float(phase[release_indices[0]])
        release_clear = release_phase <= boundary+phase_allowance
        continuity['F3_measured_automatic_release'] = release_clear
        measurements.update(release_phase_rad=release_phase, blade_half_angle_rad=-boundary,
                            release_boundary_phase_rad=boundary, numerical_phase_allowance_rad=phase_allowance)
    if 'maximum_allowed_gap_s' not in measurements:
        return _result(test_id, continuity, measurements)
    beat = measurements['reference_half_period_s']
    latency = 4*beat
    if legacy_v2:
        half_blade = .12/(2*17)
        phase_limit = half_blade+latency*2*math.pi/43200
        maximum_phase = half_blade+2*dt*2*math.pi/43200
    else:
        minimum_phase = boundary-latency*omega
        maximum_phase = boundary+phase_allowance
    armed_start, armed_end = profile['armed_window_s']
    ringing_start, ringing_end = profile['ringing_window_s']
    before_start, before_end = profile['before_off_window_s']
    after_start, after_end = profile['after_off_window_s']
    armed = [strike for strike in strikes if armed_start <= strike[1] < armed_end]
    during = [strike for strike in strikes if ringing_start <= strike[1] <= ringing_end]
    before_off = [strike for strike in strikes if before_start <= strike[1] <= before_end]
    first_matches = bool(armed) and release-dt <= armed[0][1] <= release+latency
    if legacy_v2:
        first_matches = first_matches and abs(float(phase[armed[0][0]])) <= phase_limit
    else:
        first_matches = first_matches and release_clear and minimum_phase <= float(phase[armed[0][0]]) <= maximum_phase
    assertions = {
        'F3_repeated_actual_bell_impacts':all(sum(strike[2]==bell for strike in during) >= 2 for bell in BELLS),
        'F3_no_early_impact_since_ARM':bool(armed) and all(phase[strike[0]] <= maximum_phase for strike in armed),
        'F3_first_impact_matches_actual_setting':first_matches,
        'F4_actually_ringing_immediately_before_OFF':len(before_off) >= 2,
        'F4_no_actual_impacts_after_OFF':not any(after_start <= strike[1] <= after_end for strike in strikes),
    }
    if not legacy_v2:
        assertions['F3_no_early_impact_since_ARM'] &= all(strike[1] >= release-dt for strike in armed)
        assertions['F3_no_actual_impacts_in_passive_reference'] = not any(
            reference_start <= strike[1] <= reference_end for strike in strikes)
    assertions.update(continuity)
    measurements.update(first_impact_s=armed[0][1] if armed else None,
                        first_impact_phase_rad=float(phase[armed[0][0]]) if armed else None,
                        pre_OFF_impacts=len(before_off),
                        ringing_window_s=list(profile['ringing_window_s']),
                        impacts_by_bell_in_ringing_window={bell:sum(strike[2]==bell for strike in during) for bell in BELLS},
                        actual_impact_events=[{'step':step,'time_s':time,'bell':bell} for step,time,bell in strikes])
    if legacy_v2:
        measurements['phase_limit_rad'] = phase_limit
    else:
        measurements.update(first_impact_minimum_phase_rad=minimum_phase,
                            first_impact_maximum_phase_rad=maximum_phase,
                            first_impact_latency_s=latency, first_impact_pre_release_allowance_s=dt)
    return _result(test_id, assertions, measurements)


def evaluate_registered_trace(trace, registration):
    """Return scalar metadata to merge before any fresh/cached trace is accepted.

    ``trace`` is the mapping returned by ``trace_storage.open_trace(path)``.
    ``registration`` is the independent Harness binding, not a scene callback
    or an author script name. Installation/review/calibration are caller-owned.
    The result never promotes an existing native rejection to a physical pass.
    MeasurementCapabilityError must be routed to HARNESS_ERROR, not the author.
    """
    test_id = trace.get('test_id')
    if (registration.get('schema_version') not in ('slim_alarm_measurement_binding_v2', 'slim_alarm_measurement_binding_v3')
            or registration.get('implementation') != 'slim_harness.alarm_measurements'
            or registration.get('test_id') != test_id or test_id not in PROFILES):
        raise MeasurementCapabilityError('No matching Harness-owned measurement registration')
    if registration.get('impact_event_rule') != 'first_positive_impulse_classifies_episode':
        raise MeasurementCapabilityError('Registration does not bind the calibrated contact episode rule')
    profile = measurement_profile_for_registration(registration)
    criteria = ['F5'] if test_id == 'independent_alarm_setting' else ['F3','F4','F5']
    if registration.get('criterion_ids') != criteria:
        raise MeasurementCapabilityError('Registration does not preserve the reviewed criterion mapping')
    if trace.get('failure') is not None:
        return {'physical_pass':False, 'registered_measurements':{
            'status':'not_evaluated_after_native_failure', 'test_id':test_id}}
    signature=registration.get('signature')
    if (registration.get('implementation_sha256')!=implementation_sha256()
            or not signature or signature!=registration_signature(registration)
            or trace.get('measurement_registration_signature')!=signature):
        raise MeasurementCapabilityError('Measurement implementation/registration is not bound to this trace')
    if test_id=='automatic_alarm_and_stop':
        calibration=registration.get('contact_calibration_signature')
        if not calibration or trace.get('measurement_contact_schema',{}).get('calibration_signature')!=calibration:
            raise MeasurementCapabilityError('Actual contact calibration is not bound to this registration and trace')
    signals = trace.get('signals')
    samples = trace.get('samples')
    if not isinstance(signals, Mapping) or samples is None or len(samples) != len(signals.get('time_s', [])):
        raise MeasurementCapabilityError('Full signals and solver-step sample view are required')

    def contact_steps():
        iterator = iter(samples)
        initial = next(iterator, None)
        if not initial or initial.get('time_s') != 0:
            raise MeasurementCapabilityError('Missing unsolved initial sample')
        for sample in iterator:
            row = sample.get('contact_step')
            if not isinstance(row, Mapping) or row.get('time_s') != sample.get('time_s'):
                raise MeasurementCapabilityError('Missing/misaligned sample.contact_step; absence is not zero contact')
            yield row

    result = evaluate_clock_measurement(test_id, signals,
        contact_steps() if test_id == 'automatic_alarm_and_stop' else None,
        trace.get('measurement_contact_schema'), measurement_profile=profile,
        legacy_v2=registration['schema_version']=='slim_alarm_measurement_binding_v2')
    existing = trace.get('assertions')
    if not isinstance(existing, list) or not existing:
        raise MeasurementCapabilityError('Native numerical assertions must be retained with the supplement')
    native_ids = {row['id'] for row in existing}
    if any(row['id'] in native_ids for row in result['assertions']):
        raise MeasurementCapabilityError('Registered assertion IDs collide with native assertions')
    assertions = [*existing, *result['assertions']]
    return {'assertions':assertions,
            'physical_pass':trace.get('physical_pass') is True and all(row.get('passed') is True for row in assertions),
            'registered_measurements':{'status':'evaluated', 'registration_id':registration['id'], **result}}
