from __future__ import annotations

import math

from .adapters import adapter_from_dict, get_adapter
from .models import LoadCase, LoadPhase, MechanismPlan, MechanismValidationError


def _scale(
    vector: tuple[float, float, float],
    value: float,
) -> tuple[float, float, float]:
    return tuple(item * value for item in vector)


def _cross(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _unit(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    length = math.sqrt(sum(item * item for item in vector))
    if length <= 1e-12:
        raise MechanismValidationError(
            "ORTHOGONAL_DIRECTION_UNAVAILABLE",
            "cannot derive a transverse load direction",
        )
    return tuple(item / length for item in vector)


def _orthogonal(plan: MechanismPlan) -> tuple[float, float, float]:
    gravity = plan.gravity_abs_m_s2
    if math.sqrt(sum(item * item for item in gravity)) > 1e-9:
        candidate = _cross(plan.axis_abs, _unit(gravity))
        if math.sqrt(sum(item * item for item in candidate)) > 1e-9:
            return _unit(candidate)
    basis = min(
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        key=lambda item: abs(sum(a * b for a, b in zip(plan.axis_abs, item))),
    )
    return _unit(_cross(plan.axis_abs, basis))


def _axis_case(
    plan: MechanismPlan,
    *,
    load_case_id: str,
    magnitude_n: float,
    moving_mass_kg: float,
    force_duration_s: float,
    settle_duration_s: float,
    purpose: str,
    step_s: float,
) -> LoadCase:
    return LoadCase(
        load_case_id=load_case_id,
        family=plan.family,
        program="bidirectional_axis",
        purpose=purpose,
        moving_mass_kg=moving_mass_kg,
        step_s=step_s,
        phases=(
            LoadPhase(
                "forward_force",
                "force",
                plan.axis_abs,
                magnitude_n,
                force_duration_s,
            ),
            LoadPhase("forward_settle", "none", None, 0.0, settle_duration_s),
            LoadPhase(
                "reverse_force",
                "force",
                _scale(plan.axis_abs, -1.0),
                magnitude_n * 2.25,
                force_duration_s,
            ),
            LoadPhase("reverse_settle", "none", None, 0.0, settle_duration_s),
        ),
    )
def default_prismatic_load_cases(plan: MechanismPlan) -> tuple[LoadCase, ...]:
    if plan.family != "prismatic":
        raise MechanismValidationError(
            "LOAD_FAMILY_MISMATCH",
            "prismatic load generation requires a prismatic plan",
        )
    adapter = None
    if plan.adapter_spec is not None:
        adapter = adapter_from_dict(plan.adapter_spec)
    elif plan.adapter_id is not None:
        try:
            adapter = get_adapter(plan.adapter_id)
        except KeyError:
            adapter = None
    geometry = None if adapter is None else adapter.geometry
    step_s = (
        0.00025
        if geometry is not None and geometry.rolling_elements is not None
        else 0.001
    )
    transverse = _orthogonal(plan)
    cases = (
        _axis_case(
            plan,
            load_case_id="nominal_light",
            magnitude_n=0.08,
            moving_mass_kg=0.12,
            force_duration_s=0.035,
            settle_duration_s=0.12,
            purpose="nominal",
            step_s=step_s,
        ),
        _axis_case(
            plan,
            load_case_id="nominal_loaded",
            magnitude_n=0.12,
            moving_mass_kg=0.20,
            force_duration_s=0.04,
            settle_duration_s=0.14,
            purpose="nominal",
            step_s=step_s,
        ),
        LoadCase(
            load_case_id="zero_input_gravity",
            family=plan.family,
            program="zero_input_gravity",
            purpose="adversarial",
            moving_mass_kg=0.16,
            step_s=step_s,
            phases=(LoadPhase("zero_input", "none", None, 0.0, 0.18),),
        ),
        LoadCase(
            load_case_id="transverse_probe",
            family=plan.family,
            program="transverse_probe",
            purpose="adversarial",
            moving_mass_kg=0.16,
            step_s=step_s,
            phases=(
                LoadPhase("transverse_force", "force", transverse, 0.06, 0.03),
                LoadPhase("transverse_settle", "none", None, 0.0, 0.12),
                LoadPhase(
                    "transverse_reverse_force",
                    "force",
                    _scale(transverse, -1.0),
                    0.06,
                    0.03,
                ),
                LoadPhase(
                    "transverse_reverse_settle",
                    "none",
                    None,
                    0.0,
                    0.12,
                ),
            ),
        ),
        _axis_case(
            plan,
            load_case_id="held_out_mid",
            magnitude_n=0.10,
            moving_mass_kg=0.16,
            force_duration_s=0.045,
            settle_duration_s=0.13,
            purpose="held_out",
            step_s=step_s,
        ),
    )
    task = None if adapter is None else adapter.whole_machine_task
    if task is None:
        return cases
    task_travel_m = abs(task.outbound_progress_m - task.home_progress_m)
    if (
        geometry is not None
        and geometry.certifiable_progress_limit_m is not None
        and task_travel_m > geometry.certifiable_progress_limit_m
    ):
        return cases
    task_case = LoadCase(
        load_case_id="task_envelope_roundtrip",
        family=plan.family,
        program="task_range_roundtrip",
        purpose="nominal",
        moving_mass_kg=task.moving_mass_kg,
        step_s=step_s,
        phases=(
            LoadPhase(
                "task_outbound_force",
                "force",
                plan.axis_abs,
                task.outbound_force_n,
                task.max_force_phase_s,
                stop_condition="progress_at_least",
                target_progress_m=task.outbound_progress_m,
            ),
            LoadPhase(
                "task_outbound_settle",
                "none",
                None,
                0.0,
                task.settle_phase_s,
            ),
            LoadPhase(
                "task_return_force",
                "force",
                _scale(plan.axis_abs, -1.0),
                task.return_force_n,
                task.max_force_phase_s,
                stop_condition="progress_at_most",
                target_progress_m=task.home_progress_m,
            ),
            LoadPhase(
                "task_return_settle",
                "none",
                None,
                0.0,
                task.settle_phase_s,
            ),
        ),
    )
    return (*cases, task_case)


def _revolute_execution_step_s(adapter: object | None) -> float:
    family_spec = getattr(adapter, "family_spec", None)
    if not isinstance(family_spec, dict):
        return 0.0005
    if family_spec.get("mechanism_kind") != "axis_spring":
        return 0.0005
    from .axis_spring_spec import axis_spring_from_dict

    axis_spring_from_dict(family_spec, expected_family="revolute")
    return 0.0001


def default_revolute_load_cases(plan: MechanismPlan) -> tuple[LoadCase, ...]:
    if plan.family != "revolute":
        raise MechanismValidationError(
            "LOAD_FAMILY_MISMATCH",
            "revolute load generation requires a revolute plan",
        )
    adapter = None
    if plan.adapter_spec is not None:
        adapter = adapter_from_dict(plan.adapter_spec)
    elif plan.adapter_id is not None:
        try:
            adapter = get_adapter(plan.adapter_id)
        except KeyError:
            adapter = None
    transverse = _orthogonal(plan)
    step_s = _revolute_execution_step_s(adapter)

    cases = (
        LoadCase(
            load_case_id="zero_input_gravity",
            family=plan.family,
            program="zero_input_gravity",
            purpose="adversarial",
            moving_mass_kg=0.40,
            step_s=step_s,
            phases=(LoadPhase("zero_input", "none", None, 0.0, 0.22),),
        ),
        LoadCase(
            load_case_id="transverse_torque_probe",
            family=plan.family,
            program="transverse_torque_probe",
            purpose="adversarial",
            moving_mass_kg=0.40,
            step_s=step_s,
            phases=(
                LoadPhase("transverse_torque", "torque", transverse, 0.012, 0.05),
                LoadPhase("transverse_settle", "none", None, 0.0, 0.10),
                LoadPhase(
                    "transverse_reverse_torque", "torque",
                    _scale(transverse, -1.0), 0.012, 0.05,
                ),
                LoadPhase(
                    "transverse_reverse_settle", "none", None, 0.0, 0.10
                ),
            ),
        ),
    )
    task = None if adapter is None else adapter.whole_machine_angular_task
    spring = None
    if adapter is not None and adapter.family_spec is not None:
        from .axis_spring_spec import axis_spring_from_dict

        spring = axis_spring_from_dict(
            adapter.family_spec,
            expected_family="revolute",
        )
    geometry = None if adapter is None else adapter.geometry
    if task is None:
        return cases
    travel_rad = abs(task.outbound_angle_rad - task.home_angle_rad)
    if (
        geometry is not None
        and geometry.certifiable_angle_limit_rad is not None
        and travel_rad > geometry.certifiable_angle_limit_rad
    ):
        return cases
    if (
        spring is not None
        and spring.contact_fixture is not None
        and spring.contact_fixture.shape == "dynamic_grip_box"
    ):
        grip_torque_nm = (
            spring.contact_fixture.grip_torque_nm or task.outbound_torque_nm
        )
        close_phase = lambda phase_id: LoadPhase(
            phase_id,
            "torque",
            plan.axis_abs,
            grip_torque_nm,
            task.max_torque_phase_s,
            stop_condition="position_at_least",
            target_position_si=task.outbound_angle_rad,
        )
        grip_case = LoadCase(
            load_case_id="tong_grip_object",
            family=plan.family,
            program="tong_grip_object",
            purpose="nominal",
            moving_mass_kg=task.moving_mass_kg,
            step_s=step_s,
            phases=(
                close_phase("tong_grip_close"),
                LoadPhase(
                    "tong_grip_verify",
                    "torque",
                    plan.axis_abs,
                    grip_torque_nm,
                    0.12,
                ),
            ),
        )
        hold_case = LoadCase(
            load_case_id="tong_hold_object",
            family=plan.family,
            program="tong_hold_object",
            purpose="nominal",
            moving_mass_kg=task.moving_mass_kg,
            step_s=step_s,
            phases=(
                close_phase("tong_hold_close"),
                LoadPhase(
                    "tong_hold_verify",
                    "torque",
                    plan.axis_abs,
                    grip_torque_nm,
                    spring.contact_fixture.hold_duration_s,
                ),
            ),
        )
        release_case = LoadCase(
            load_case_id="tong_release_object",
            family=plan.family,
            program="tong_release_object",
            purpose="nominal",
            moving_mass_kg=task.moving_mass_kg,
            step_s=step_s,
            phases=(
                close_phase("tong_release_close"),
                LoadPhase(
                    "tong_release_preload",
                    "torque",
                    plan.axis_abs,
                    grip_torque_nm,
                    0.12,
                ),
                LoadPhase(
                    "tong_release_passive_return",
                    "none",
                    None,
                    0.0,
                    0.60,
                ),
                LoadPhase(
                    "tong_release_observe",
                    "none",
                    None,
                    0.0,
                    0.08,
                ),
            ),
        )
        return (*cases, grip_case, hold_case, release_case)
    if (
        spring is not None
        and spring.contact_fixture is not None
        and spring.contact_fixture.shape == "tabletop_pickup_box"
    ):
        grip_torque_nm = (
            spring.contact_fixture.grip_torque_nm or task.outbound_torque_nm
        )
        close_phase = lambda phase_id: LoadPhase(
            phase_id,
            "torque",
            plan.axis_abs,
            grip_torque_nm,
            task.max_torque_phase_s,
            stop_condition="position_at_least",
            target_position_si=task.outbound_angle_rad,
        )

        def passive(phase_id: str, duration_s: float) -> LoadPhase:
            return LoadPhase(phase_id, "none", None, 0.0, duration_s)

        def driven(phase_id: str, duration_s: float) -> LoadPhase:
            return LoadPhase(
                phase_id,
                "torque",
                plan.axis_abs,
                grip_torque_nm,
                duration_s,
            )

        pickup_case = LoadCase(
            load_case_id="tong_tabletop_pickup",
            family=plan.family,
            program="tong_tabletop_pickup",
            purpose="nominal",
            moving_mass_kg=task.moving_mass_kg,
            step_s=step_s,
            phases=(
                passive("tong_pickup_initial_table_settle", 0.15),
                passive("tong_pickup_approach", 0.45),
                close_phase("tong_pickup_close"),
                driven("tong_pickup_grip_verify", 0.25),
                driven(
                    "tong_pickup_lift",
                    spring.contact_fixture.lift_duration_s or 0.40,
                ),
                driven("tong_pickup_verify", 0.15),
            ),
        )
        hold_case = LoadCase(
            load_case_id="tong_tabletop_hold",
            family=plan.family,
            program="tong_tabletop_hold",
            purpose="nominal",
            moving_mass_kg=task.moving_mass_kg,
            step_s=step_s,
            phases=(
                passive("tong_hold_initial_table_settle", 0.15),
                passive("tong_hold_approach", 0.45),
                close_phase("tong_hold_close"),
                driven("tong_hold_grip_verify", 0.25),
                driven(
                    "tong_hold_lift",
                    spring.contact_fixture.lift_duration_s or 0.40,
                ),
                driven(
                    "tong_hold_verify",
                    spring.contact_fixture.hold_duration_s,
                ),
            ),
        )
        release_case = LoadCase(
            load_case_id="tong_tabletop_release",
            family=plan.family,
            program="tong_tabletop_release",
            purpose="nominal",
            moving_mass_kg=task.moving_mass_kg,
            step_s=step_s,
            phases=(
                passive("tong_release_initial_table_settle", 0.15),
                passive("tong_release_approach", 0.45),
                close_phase("tong_release_close"),
                driven("tong_release_grip_verify", 0.25),
                driven(
                    "tong_release_lift",
                    spring.contact_fixture.lift_duration_s or 0.40,
                ),
                driven("tong_release_preload", 0.12),
                LoadPhase(
                    "tong_release_gradual_open",
                    "torque",
                    plan.axis_abs,
                    grip_torque_nm,
                    3.00,
                    end_magnitude_si=0.0,
                ),
                passive("tong_release_passive_return", 0.30),
                passive("tong_release_final_table_settle", 0.40),
                passive("tong_release_observe", 0.15),
            ),
        )
        return (*cases, pickup_case, hold_case, release_case)
    if spring is not None:
        hold_duration_s = (
            spring.contact_fixture.hold_duration_s
            if spring.contact_fixture is not None
            else max(0.05, task.settle_phase_s)
        )
        task_case = LoadCase(
            load_case_id="task_angle_roundtrip",
            family=plan.family,
            program="task_angle_roundtrip",
            purpose="nominal",
            moving_mass_kg=task.moving_mass_kg,
            step_s=step_s,
            phases=(
                LoadPhase(
                    "task_outbound_torque",
                    "torque",
                    plan.axis_abs,
                    task.outbound_torque_nm,
                    task.max_torque_phase_s,
                    stop_condition="position_at_least",
                    target_position_si=task.outbound_angle_rad,
                ),
                LoadPhase(
                    "task_outbound_hold",
                    "torque",
                    plan.axis_abs,
                    task.outbound_torque_nm,
                    hold_duration_s,
                ),
                LoadPhase(
                    "task_passive_spring_return",
                    "none",
                    None,
                    0.0,
                    task.max_torque_phase_s,
                ),
                LoadPhase(
                    "task_return_settle",
                    "none",
                    None,
                    0.0,
                    max(0.05, task.settle_phase_s),
                ),
            ),
        )
        return (*cases, task_case)
    task_case = LoadCase(
        load_case_id="task_angle_roundtrip",
        family=plan.family,
        program="task_angle_roundtrip",
        purpose="nominal",
        moving_mass_kg=task.moving_mass_kg,
        step_s=step_s,
        phases=(
            LoadPhase(
                "task_outbound_torque", "torque", plan.axis_abs,
                task.outbound_torque_nm, task.max_torque_phase_s,
                stop_condition="position_at_least",
                target_position_si=task.outbound_angle_rad,
            ),
            LoadPhase("task_outbound_settle", "none", None, 0.0, task.settle_phase_s),
            LoadPhase(
                "task_return_torque", "torque", _scale(plan.axis_abs, -1.0),
                task.return_torque_nm, task.max_torque_phase_s,
                stop_condition="position_at_most",
                target_position_si=task.home_angle_rad,
            ),
            LoadPhase("task_return_settle", "none", None, 0.0, task.settle_phase_s),
        ),
    )
    return (*cases, task_case)


def default_compound_load_cases(plan: MechanismPlan) -> tuple[LoadCase, ...]:
    if plan.family != "compound":
        raise MechanismValidationError(
            "LOAD_FAMILY_MISMATCH",
            "compound load generation requires a compound plan",
        )
    if plan.adapter_id is None:
        raise MechanismValidationError(
            "MECHANISM_ADAPTER_UNAVAILABLE",
            f"mechanism {plan.capsule_id!r} has no trusted execution data",
        )
    adapter = (
        adapter_from_dict(plan.adapter_spec)
        if plan.adapter_spec is not None
        else get_adapter(plan.adapter_id)
    )
    from .compound_spec import compound_linkage_from_dict

    linkage = compound_linkage_from_dict(adapter.family_spec)
    moving_mass = sum(body.mass_kg for body in linkage.bodies if not body.fixed)
    press_direction = linkage.input_actuation_direction_abs
    release_direction = _scale(press_direction, -1.0)
    transverse_reference = linkage.input_force_direction_abs
    basis = min(
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        key=lambda item: abs(sum(a * b for a, b in zip(transverse_reference, item))),
    )
    transverse_direction = _unit(_cross(transverse_reference, basis))

    def cycle(
        load_case_id: str,
        *,
        force_scale: float,
        purpose: str,
        target: bool = False,
    ) -> LoadCase:
        press = LoadPhase(
            "press_pedal",
            linkage.input_wrench_kind,
            press_direction,
            linkage.nominal_input_magnitude_si * force_scale,
            linkage.max_force_phase_s,
            stop_condition="position_at_least" if target else None,
            target_position_si=(linkage.outbound_output_angle_rad if target else None),
        )
        release_magnitude = max(
            linkage.return_input_magnitude_si,
            linkage.nominal_input_magnitude_si * 0.08,
        )
        return LoadCase(
            load_case_id=load_case_id,
            family=plan.family,
            program=("task_output_angle_roundtrip" if target else "bidirectional_linkage"),
            purpose=purpose,
            moving_mass_kg=moving_mass,
            step_s=0.0001,
            sample_interval_s=0.05,
            phases=(
                press,
                LoadPhase("open_settle", "none", None, 0.0, linkage.settle_phase_s),
                LoadPhase(
                    "release_pedal",
                    linkage.input_wrench_kind,
                    release_direction,
                    release_magnitude,
                    linkage.max_return_phase_s,
                ),
                LoadPhase("closed_settle", "none", None, 0.0, linkage.settle_phase_s),
            ),
        )

    zero = LoadCase(
        load_case_id="zero_input_gravity",
        family=plan.family,
        program="zero_input_gravity",
        purpose="adversarial",
        moving_mass_kg=moving_mass,
        step_s=0.0001,
        sample_interval_s=0.05,
        phases=(LoadPhase("zero_input", "none", None, 0.0, 0.35),),
    )
    transverse = LoadCase(
        load_case_id="transverse_linkage_probe",
        family=plan.family,
        program="transverse_linkage_probe",
        purpose="adversarial",
        moving_mass_kg=moving_mass,
        # The lateral probe excites the stiff bearing-seat directions, so it
        # deliberately uses a finer explicit integration step than free motion.
        step_s=0.00002,
        sample_interval_s=0.05,
        phases=(
            LoadPhase(
                "transverse_force",
                "force",
                transverse_direction,
                linkage.nominal_input_force_n * 0.20,
                0.20,
            ),
            LoadPhase("transverse_settle", "none", None, 0.0, 0.10),
        ),
    )
    return (
        cycle("nominal_light", force_scale=0.82, purpose="nominal"),
        cycle("nominal_loaded", force_scale=1.18, purpose="nominal"),
        zero,
        transverse,
        cycle("held_out_mid", force_scale=0.96, purpose="held_out"),
        cycle(
            "task_output_angle_roundtrip",
            force_scale=1.0,
            purpose="nominal",
            target=True,
        ),
    )


def default_load_cases(plan: MechanismPlan) -> tuple[LoadCase, ...]:
    if plan.adapter_id is None:
        raise MechanismValidationError(
            "MECHANISM_ADAPTER_UNAVAILABLE",
            f"mechanism {plan.capsule_id!r} has no trusted execution data",
        )
    try:
        adapter = (
            adapter_from_dict(plan.adapter_spec)
            if plan.adapter_spec is not None
            else get_adapter(plan.adapter_id)
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise MechanismValidationError(
            "MECHANISM_ADAPTER_UNKNOWN", str(exc)
        ) from exc
    from .family_plugins import (
        invoke_mechanism_plugin_callback,
        mechanism_adapter_for,
        plugin_registration_or_none,
    )
    from .plugin_boundary import plugin_boundary_error

    plugin = mechanism_adapter_for(plan, adapter)
    load_cases = invoke_mechanism_plugin_callback(
        plugin,
        "load_cases",
        plan,
        source_component="mechanism.loads.load_cases",
        unexpected_code="MECHANISM_ADAPTER_LOAD_CASE_FAILED",
    )
    if not isinstance(load_cases, tuple) or any(
        not isinstance(item, LoadCase) for item in load_cases
    ):
        message = "mechanism adapter load_cases() returned an invalid inventory"
        raise plugin_boundary_error(
            TypeError(message),
            registration=plugin_registration_or_none(str(plugin.plugin_id)),
            plugin_id=str(plugin.plugin_id),
            plugin_kind="mechanism",
            callback="load_cases",
            source_component="mechanism.loads.load_cases",
            code="MECHANISM_ADAPTER_LOAD_CASE_INVALID",
            message=message,
            details={"capsule_id": plan.capsule_id},
            contract_violation=True,
        )
    return load_cases
