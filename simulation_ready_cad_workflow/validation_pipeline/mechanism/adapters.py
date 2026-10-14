from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class MeshAssetSpec:
    workspace_relative_path: str
    sha256: str
    units_to_m: float
    rotation_abs_from_source: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    convex: bool = False


@dataclass(frozen=True)
class RollingElementSpec:
    element_id: str
    evidence_mesh_path: str
    evidence_mesh_sha256: str
    center_source_units: tuple[float, float, float]


@dataclass(frozen=True)
class RollingElementSetSpec:
    elements: tuple[RollingElementSpec, ...]
    units_to_m: float
    rotation_abs_from_source: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    radius_m: float
    density_kg_m3: float
    pre_settle_s: float
    minimum_rolling_count: int
    minimum_phase_rotation_rad: float
    max_lateral_shift_m: float
    return_path_resolved: bool


@dataclass(frozen=True)
class TwoBodyGeometrySpec:
    fixed_mesh: MeshAssetSpec
    moving_mesh: MeshAssetSpec
    fixed_origin_abs_m: tuple[float, float, float]
    moving_origin_abs_m: tuple[float, float, float]
    fixed_collision_meshes: tuple[MeshAssetSpec, ...] = ()
    moving_collision_meshes: tuple[MeshAssetSpec, ...] = ()
    fixed_mass_kg: float = 1.0
    friction: float = 0.02
    collision_radius_m: float = 0.0
    collision_envelope_m: float = 12e-6
    collision_margin_m: float = 2e-6
    rolling_elements: RollingElementSetSpec | None = None
    certifiable_progress_limit_m: float | None = None
    certifiable_angle_limit_rad: float | None = None
    collision_scope: tuple[str, ...] = ()
    omitted_contact_geometry: tuple[str, ...] = ()


@dataclass(frozen=True)
class WholeMachineTaskSpec:
    outbound_task_id: str
    return_task_id: str
    outbound_progress_m: float
    home_progress_m: float
    moving_mass_kg: float
    outbound_force_n: float
    return_force_n: float
    max_force_phase_s: float
    settle_phase_s: float


@dataclass(frozen=True)
class WholeMachineAngularTaskSpec:
    outbound_task_id: str
    return_task_id: str
    outbound_angle_rad: float
    home_angle_rad: float
    moving_mass_kg: float
    outbound_torque_nm: float
    return_torque_nm: float
    max_torque_phase_s: float
    settle_phase_s: float


@dataclass(frozen=True)
class AdapterSpec:
    adapter_id: str
    family: str
    required_source_hashes: frozenset[str]
    supported_programs: frozenset[str]
    geometry: TwoBodyGeometrySpec | None = None
    required_requirements_sha256: str | None = None
    whole_machine_task: WholeMachineTaskSpec | None = None
    whole_machine_angular_task: WholeMachineAngularTaskSpec | None = None
    # Family-owned execution data. Core code preserves and hash-binds this
    # object; the selected mechanism plugin owns its schema and validation.
    family_spec: dict[str, Any] | None = None
    # Trusted report binding required before new physical dispatch.
    collider_fidelity: dict[str, Any] | None = None


DERIVED_PLAIN_PRISMATIC_ADAPTER_ID = "derived_plain_prismatic_v1"
DERIVED_PLAIN_REVOLUTE_ADAPTER_ID = "derived_plain_revolute_v1"
DERIVED_COMPOUND_ADAPTER_ID = "derived_compound_linkage_v1"


def adapter_to_dict(adapter: AdapterSpec) -> dict[str, Any]:
    payload = asdict(adapter)
    payload["required_source_hashes"] = sorted(adapter.required_source_hashes)
    payload["supported_programs"] = sorted(adapter.supported_programs)
    return payload


def _mesh_asset_from_dict(value: Mapping[str, Any]) -> MeshAssetSpec:
    return MeshAssetSpec(
        workspace_relative_path=str(value["workspace_relative_path"]),
        sha256=str(value["sha256"]),
        units_to_m=float(value["units_to_m"]),
        rotation_abs_from_source=tuple(
            tuple(float(item) for item in row)
            for row in value["rotation_abs_from_source"]
        ),
        convex=bool(value.get("convex", False)),
    )


def adapter_from_dict(value: Mapping[str, Any]) -> AdapterSpec:
    geometry_value = value.get("geometry")
    geometry = None
    if isinstance(geometry_value, Mapping):
        rolling_value = geometry_value.get("rolling_elements")
        rolling = None
        if isinstance(rolling_value, Mapping):
            rolling = RollingElementSetSpec(
                elements=tuple(
                    RollingElementSpec(
                        element_id=str(item["element_id"]),
                        evidence_mesh_path=str(item["evidence_mesh_path"]),
                        evidence_mesh_sha256=str(item["evidence_mesh_sha256"]),
                        center_source_units=tuple(
                            float(component)
                            for component in item["center_source_units"]
                        ),
                    )
                    for item in rolling_value.get("elements", [])
                ),
                units_to_m=float(rolling_value["units_to_m"]),
                rotation_abs_from_source=tuple(
                    tuple(float(item) for item in row)
                    for row in rolling_value["rotation_abs_from_source"]
                ),
                radius_m=float(rolling_value["radius_m"]),
                density_kg_m3=float(rolling_value["density_kg_m3"]),
                pre_settle_s=float(rolling_value["pre_settle_s"]),
                minimum_rolling_count=int(rolling_value["minimum_rolling_count"]),
                minimum_phase_rotation_rad=float(
                    rolling_value["minimum_phase_rotation_rad"]
                ),
                max_lateral_shift_m=float(rolling_value["max_lateral_shift_m"]),
                return_path_resolved=bool(rolling_value["return_path_resolved"]),
            )
        geometry = TwoBodyGeometrySpec(
            fixed_mesh=_mesh_asset_from_dict(geometry_value["fixed_mesh"]),
            moving_mesh=_mesh_asset_from_dict(geometry_value["moving_mesh"]),
            fixed_origin_abs_m=tuple(
                float(item) for item in geometry_value["fixed_origin_abs_m"]
            ),
            moving_origin_abs_m=tuple(
                float(item) for item in geometry_value["moving_origin_abs_m"]
            ),
            fixed_collision_meshes=tuple(
                _mesh_asset_from_dict(item)
                for item in geometry_value.get("fixed_collision_meshes", [])
            ),
            moving_collision_meshes=tuple(
                _mesh_asset_from_dict(item)
                for item in geometry_value.get("moving_collision_meshes", [])
            ),
            fixed_mass_kg=float(geometry_value.get("fixed_mass_kg", 1.0)),
            friction=float(geometry_value.get("friction", 0.02)),
            collision_radius_m=float(geometry_value.get("collision_radius_m", 0.0)),
            collision_envelope_m=float(
                geometry_value.get("collision_envelope_m", 12e-6)
            ),
            collision_margin_m=float(
                geometry_value.get("collision_margin_m", 2e-6)
            ),
            rolling_elements=rolling,
            certifiable_progress_limit_m=(
                None
                if geometry_value.get("certifiable_progress_limit_m") is None
                else float(geometry_value["certifiable_progress_limit_m"])
            ),
            certifiable_angle_limit_rad=(
                None
                if geometry_value.get("certifiable_angle_limit_rad") is None
                else float(geometry_value["certifiable_angle_limit_rad"])
            ),
            collision_scope=tuple(geometry_value.get("collision_scope", [])),
            omitted_contact_geometry=tuple(
                geometry_value.get("omitted_contact_geometry", [])
            ),
        )
    task_value = value.get("whole_machine_task")
    task = (
        WholeMachineTaskSpec(**task_value)
        if isinstance(task_value, Mapping)
        else None
    )
    angular_task_value = value.get("whole_machine_angular_task")
    angular_task = (
        WholeMachineAngularTaskSpec(**angular_task_value)
        if isinstance(angular_task_value, Mapping)
        else None
    )
    return AdapterSpec(
        adapter_id=str(value["adapter_id"]),
        family=str(value["family"]),
        required_source_hashes=frozenset(value.get("required_source_hashes", [])),
        supported_programs=frozenset(value.get("supported_programs", [])),
        geometry=geometry,
        required_requirements_sha256=value.get("required_requirements_sha256"),
        whole_machine_task=task,
        whole_machine_angular_task=angular_task,
        family_spec=(
            dict(value["family_spec"])
            if isinstance(value.get("family_spec"), Mapping)
            else None
        ),
        collider_fidelity=(
            dict(value["collider_fidelity"])
            if isinstance(value.get("collider_fidelity"), Mapping)
            else None
        ),
    )


MCM_RAIL_SHA256 = (
    "be1e5ab3f8dedf98d1efb4274c80734889ffbd827d6f3a5f1906ce7c0ea964e4"
)
MCM_CARRIAGE_SHA256 = (
    "de59d42fc4a04b7d66860cde0dde000faa6048a98b553fb0dff90600c4985c62"
)
_RX_POS_90 = (
    (1.0, 0.0, 0.0),
    (0.0, 0.0, -1.0),
    (0.0, 1.0, 0.0),
)
_BALL_PARTS_ROOT = (
    "simulation_ready_cad_workflow/joint_template_catalog/unit_tests/"
    "prismatic_slider/rolling_element_evidence"
)


def _ball(
    solid: int,
    center_source_mm: tuple[float, float, float],
    sha256: str,
) -> RollingElementSpec:
    return RollingElementSpec(
        element_id=f"ball_solid_{solid:03d}",
        evidence_mesh_path=(
            f"{_BALL_PARTS_ROOT}/carriage_solid_{solid:03d}.obj"
        ),
        evidence_mesh_sha256=sha256,
        center_source_units=center_source_mm,
    )


_MCM_ACTIVE_BALLS = RollingElementSetSpec(
    elements=(
        _ball(
            9,
            (-0.000974, -1.14, -6.195244),
            "0408a4b633f747025f5363189d93452761e3f1a0aec76a9d47d3170c5d6a65ab",
        ),
        _ball(
            10,
            (2.473326, -1.14, -6.195244),
            "3b4716f28d6180c02bd343961aa843a0ae825342b14ca8e377934b98b9c1fbed",
        ),
        _ball(
            11,
            (4.947627, -1.14, -6.195244),
            "1f8420e5501aa0e569d148060722e219e6644f2e5b7c08a575b2fa11c41a387e",
        ),
        _ball(
            12,
            (7.421927, -1.14, -6.195244),
            "b91df1d9c2b491329566a506aaba2b2b03377495daf685f2b686b5c821a15f66",
        ),
        _ball(
            13,
            (9.896227, -1.14, -6.195244),
            "3d8c02ea94c75d0ce66d0aa25f8522a283f66e9ee2ebada018e063cdd84829ab",
        ),
        _ball(
            24,
            (-9.898175, -1.14, -6.195244),
            "6837959fafa1d30e3e5e3f127efb4d0ac371e9c8beb4bafb1a7c56cfee4adaef",
        ),
        _ball(
            25,
            (-7.423875, -1.14, -6.195244),
            "94b7821702dc32f520664c4016aac9d38ea1419033444063d683ef8c95566f8c",
        ),
        _ball(
            26,
            (-4.949574, -1.14, -6.195244),
            "b91c708422680b960c7f7e7ccf7e63f12366aebd62bce935ccf8358c6ab8f93f",
        ),
        _ball(
            45,
            (-2.475274, -1.14, -6.195244),
            "9766e1578d4a8461afb278ef13e2b2099360d523ae6a58f80a1a4d2fe6285d37",
        ),
        _ball(
            27,
            (0.000974, -1.14, 6.195244),
            "2b23e4e7e2da97dcc7ddd9328df252eae4df44e1523a505959adf8f40444f398",
        ),
        _ball(
            28,
            (-2.473326, -1.14, 6.195244),
            "1edebc62a2b861583ae2a1e4d9166b758cf3e8bf470817afe970b6c27c546c0d",
        ),
        _ball(
            29,
            (-4.947627, -1.14, 6.195244),
            "ad2c99c6eb50c1809f42a9ba24fb2cc870b80f786729b872b1ad56f2cf10ecc6",
        ),
        _ball(
            30,
            (-7.421927, -1.14, 6.195244),
            "0cb29b71105f50699fb205b143fc613af9f308bf704cf685743663c1719982e7",
        ),
        _ball(
            31,
            (-9.896227, -1.14, 6.195244),
            "b2d0f4fa183010a19b9c7d3cafd14a6a43fb8a47150849c762368cca0d3d777a",
        ),
        _ball(
            42,
            (9.898175, -1.14, 6.195244),
            "1ef89a2b97a2c6fc4e5cceaba0a682b27a6512b1cca124e6c0ad889d22d5cb23",
        ),
        _ball(
            43,
            (7.423875, -1.14, 6.195244),
            "90115e74cc252b8f8189ff79003e99e4587a4ee84b99557a6781f8eb5a92259d",
        ),
        _ball(
            44,
            (4.949574, -1.14, 6.195244),
            "d7741e599d96936adc2d4c2266c00e65cda89dd8a4f9d8a4be5c703f35a4658c",
        ),
        _ball(
            46,
            (2.475274, -1.14, 6.195244),
            "f17643326c18d085f8aa702d787d1671b94ab77dab4c74091071b37540bdc724",
        ),
    ),
    units_to_m=0.001,
    rotation_abs_from_source=_RX_POS_90,
    radius_m=0.0010906102,
    density_kg_m3=7850.0,
    pre_settle_s=0.08,
    minimum_rolling_count=12,
    minimum_phase_rotation_rad=0.13962634015954636,
    max_lateral_shift_m=0.0007,
    return_path_resolved=False,
)


ADAPTERS = (
    AdapterSpec(
        adapter_id="mcmaster_6725k41_8438k4_v1",
        family="prismatic",
        required_source_hashes=frozenset(
            {MCM_RAIL_SHA256, MCM_CARRIAGE_SHA256}
        ),
        supported_programs=frozenset(
            {
                "bidirectional_axis",
                "zero_input_gravity",
                "transverse_probe",
                "task_range_roundtrip",
            }
        ),
        required_requirements_sha256=(
            "e962c417aa97ba4fbc4c219f745a0467eb6dd5f37bab95c2f5865ae14985b325"
        ),
        whole_machine_task=WholeMachineTaskSpec(
            outbound_task_id="transport_payload_outbound",
            return_task_id="return_payload_home",
            outbound_progress_m=0.04,
            home_progress_m=0.0,
            moving_mass_kg=0.16,
            outbound_force_n=0.12,
            return_force_n=0.27,
            max_force_phase_s=0.8,
            settle_phase_s=0.15,
        ),
        geometry=TwoBodyGeometrySpec(
            fixed_mesh=MeshAssetSpec(
                workspace_relative_path=(
                    "simulation_ready_cad_workflow/joint_template_catalog/"
                    "unit_tests/prismatic_slider/generated_colliders/"
                    "6725K41_rail_contact.collider.reduced_nonconvex.obj"
                ),
                sha256=(
                    "891e61557f2b7d7c2f672eee37db23487820e1ad1794b337565fdc7ba5299d78"
                ),
                units_to_m=0.001,
                rotation_abs_from_source=_RX_POS_90,
            ),
            moving_mesh=MeshAssetSpec(
                workspace_relative_path=(
                    "simulation_ready_cad_workflow/joint_template_catalog/"
                    "unit_tests/prismatic_slider/generated_colliders/"
                    "8438K4_carriage_raceways.collider.reduced_nonconvex.obj"
                ),
                sha256=(
                    "aedf00a59d9795872e092d1a84c581bb24e6afd79f31bf0d1ee85f29f705b75d"
                ),
                units_to_m=0.001,
                rotation_abs_from_source=_RX_POS_90,
            ),
            fixed_origin_abs_m=(0.0, 0.0, -0.0025),
            moving_origin_abs_m=(0.0, 0.0, 0.0015),
            friction=0.12,
            collision_envelope_m=8e-6,
            collision_margin_m=1e-6,
            rolling_elements=_MCM_ACTIVE_BALLS,
            certifiable_progress_limit_m=0.004,
            collision_scope=(
                "rail raceway surfaces",
                "moving carriage raceway and retaining surfaces",
                "18 independently rotating STEP-derived active balls",
            ),
            omitted_contact_geometry=(
                "base frame outside the tested rail-carriage contact path",
                "payload tray and force tab rigidly represented by mass and inertia",
                "return-channel balls and recirculation path",
                "end stops outside the certified local progress envelope",
            ),
        ),
    ),
)


def select_adapter(
    family: str,
    artifact_hashes: Mapping[str, str],
    requirements_sha256: str | None = None,
) -> AdapterSpec | None:
    available_hashes = {digest.casefold() for digest in artifact_hashes.values()}
    matches = [
        adapter
        for adapter in ADAPTERS
        if adapter.family == family
        and adapter.required_source_hashes.issubset(available_hashes)
        and (
            adapter.required_requirements_sha256 is None
            or adapter.required_requirements_sha256 == requirements_sha256
        )
    ]
    if len(matches) > 1:
        raise RuntimeError(f"multiple adapters match family {family!r}")
    return matches[0] if matches else None


def get_adapter(adapter_id: str) -> AdapterSpec:
    if adapter_id == DERIVED_COMPOUND_ADAPTER_ID:
        return AdapterSpec(
            adapter_id=DERIVED_COMPOUND_ADAPTER_ID,
            family="compound",
            required_source_hashes=frozenset(),
            supported_programs=frozenset(
                {
                    "bidirectional_linkage",
                    "zero_input_gravity",
                    "transverse_linkage_probe",
                    "task_output_angle_roundtrip",
                    "bidirectional_rack_pinion",
                    "transverse_rack_probe",
                    "task_rack_roundtrip",
                    "task_bicycle_powered_roundtrip",
                    "task_bicycle_ground_roll",
                    "bicycle_transmission_held_output_load",
                    "bicycle_zero_input_home_drift",
                    "bicycle_reverse_crank_and_resistance",
                    "bicycle_bounded_rear_resistance_backdrive",
                    "bicycle_passive_release_return",
                    "bidirectional_clock_train",
                    "zero_input_clock_train",
                    "transverse_clock_probe",
                    "task_clock_cycle",
                    "manual_time_setting_clock",
                    "alarm_clock_cycle",
                    "manual_zero_and_resume_clock",
                    "hour_hand_alarm_setting_clock",
                    "scheduled_alarm_clock",
                    "bidirectional_multi_output_shaft",
                    "zero_input_multi_output_shaft",
                    "transverse_multi_output_shaft",
                    "task_egg_beater_cycle",
                    "select_detent_position",
                    "hold_detent_under_axial_load",
                    "zero_input_detent_slider",
                    "task_detent_extend",
                    "task_detent_hold_under_load",
                    "task_detent_second_position",
                    "task_detent_cycle",
                    "pen_first_click",
                    "pen_two_click_cycle",
                    "zero_input_click_pen",
                    "task_pen_first_click",
                    "task_pen_second_click",
                    "task_pen_repeat_cycles",
                    # [kimi-scissor-lift]
                    "bidirectional_scissor_lift",
                    "transverse_scissor_probe",
                    "task_platform_roundtrip",
                }
            ),
        )
    if adapter_id == DERIVED_PLAIN_REVOLUTE_ADAPTER_ID:
        return AdapterSpec(
            adapter_id=DERIVED_PLAIN_REVOLUTE_ADAPTER_ID,
            family="revolute",
            required_source_hashes=frozenset(),
            supported_programs=frozenset(
                {
                    "bidirectional_rotation",
                    "zero_input_gravity",
                    "transverse_torque_probe",
                    "task_angle_roundtrip",
                }
            ),
        )
    if adapter_id == DERIVED_PLAIN_PRISMATIC_ADAPTER_ID:
        return AdapterSpec(
            adapter_id=DERIVED_PLAIN_PRISMATIC_ADAPTER_ID,
            family="prismatic",
            required_source_hashes=frozenset(),
            supported_programs=frozenset(
                {
                    "bidirectional_axis",
                    "zero_input_gravity",
                    "transverse_probe",
                    "task_range_roundtrip",
                }
            ),
        )
    for adapter in ADAPTERS:
        if adapter.adapter_id == adapter_id:
            return adapter
    raise KeyError(f"unknown mechanism adapter: {adapter_id}")
