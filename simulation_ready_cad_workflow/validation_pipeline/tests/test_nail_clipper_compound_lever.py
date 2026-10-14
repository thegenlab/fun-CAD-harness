from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from simulation_ready_cad_workflow.validation_pipeline.mechanism.compound_spec import (
    CompoundBodySpec,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.family_plugins import (
    plugin_registration_for,
    public_simulation_interface_catalog,
    wrench_policies_for_plugin,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.nail_clipper_compound_lever import (
    NailClipperCompoundLeverAdapter,
    _build_nail_clipper_topology,
    _completed_nail_clipper_trace,
    _completed_trace_payload,
    _require_fixed_duration_phases,
    _transition_task_input_phase,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.io import (
    canonical_sha256,
    lane_execution_source_sha256,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.models import (
    MechanismValidationError,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.wrench_ledger import (
    WrenchLedger,
    activate_wrench_ledger,
    create_body_torque,
    set_current_wrench_phase,
    standard_wrench_declaration,
    transition_current_wrench_phase,
)


def _family_spec() -> dict[str, object]:
    return {
        "mechanism_kind": "nail_clipper_compound_lever",
        "bodies": [
            {
                "body_id": "stationary_cutting_jaw",
                "group_ids": ["group_stationary_cutting_jaw"],
                "fixed": True,
                "mass_kg": 0.03,
                "center_abs_m": [0.032, 0.0, 0.004],
                "size_m": [0.058, 0.011, 0.008],
            },
            {
                "body_id": "actuating_lever",
                "group_ids": ["group_actuating_lever"],
                "fixed": False,
                "mass_kg": 0.01,
                "center_abs_m": [0.015, 0.0, 0.018],
                "size_m": [0.026, 0.011, 0.006],
            },
            {
                "body_id": "moving_cutting_jaw",
                "group_ids": ["group_moving_cutting_jaw"],
                "fixed": False,
                "mass_kg": 0.02,
                "center_abs_m": [0.04, 0.0, 0.012],
                "size_m": [0.04, 0.011, 0.006],
            },
        ],
        "revolute_joints": [
            {
                "joint_id": "jaw_revolute",
                "parent_body_id": "stationary_cutting_jaw",
                "child_body_id": "moving_cutting_jaw",
                "origin_abs_m": [0.048, 0.0, 0.007],
                "axis_abs": [0.0, 1.0, 0.0],
            },
            {
                "joint_id": "lever_revolute",
                "parent_body_id": "moving_cutting_jaw",
                "child_body_id": "actuating_lever",
                "origin_abs_m": [0.022, 0.0, 0.012],
                "axis_abs": [0.0, 1.0, 0.0],
            },
        ],
        "restoring_spring": {
            "body_a_id": "stationary_cutting_jaw",
            "body_b_id": "moving_cutting_jaw",
            "origin_abs_m": [0.048, 0.0, 0.007],
            "axis_abs": [0.0, 1.0, 0.0],
            "stiffness_n_m_rad": 0.08,
            "damping_n_m_s_rad": 0.002,
            "rest_angle_rad": 0.0,
        },
        "input_body_id": "actuating_lever",
        "output_body_id": "moving_cutting_jaw",
        "contact_point_abs_m": [0.008, 0.0, 0.007],
        "cutting_edge_radius_m": 0.01,
    }


class NailClipperInstalledAdapterTest(unittest.TestCase):
    def test_family_spec_requires_the_two_canonical_joint_ids(self) -> None:
        family_spec = _family_spec()
        family_spec["revolute_joints"][0]["joint_id"] = "renamed_jaw"

        with self.assertRaises(MechanismValidationError) as raised:
            NailClipperCompoundLeverAdapter().parse_family_spec(family_spec)

        self.assertEqual(raised.exception.code, "FAMILY_SPEC_INVALID")
        self.assertIn("jaw_revolute", raised.exception.message)

    def test_worker_catalog_exposes_required_geometry_proof_kind(self) -> None:
        catalog = public_simulation_interface_catalog()
        plugin = next(
            item
            for item in catalog["plugins"]
            if item["plugin_id"] == "nail_clipper_compound_lever_v1"
        )

        self.assertEqual(
            plugin["geometry_proof_kinds"],
            ["revolute.pin_bore.v1"],
        )

    def test_adapter_requires_geometry_proofs_for_both_revolutes(self) -> None:
        plugin = NailClipperCompoundLeverAdapter()
        adapter = SimpleNamespace(family_spec=_family_spec())

        requirements = plugin.required_geometry_proofs(
            adapter,
            capsule_id="nail_clipper_compound_lever",
        )

        self.assertEqual(
            [item.joint_id for item in requirements],
            ["jaw_revolute", "lever_revolute"],
        )
        self.assertEqual(
            {item.proof_kind for item in requirements},
            {"revolute.pin_bore.v1"},
        )
        self.assertEqual(
            [item.child_body_id for item in requirements],
            ["moving_cutting_jaw", "actuating_lever"],
        )

    def test_adapter_consumes_both_geometry_proofs_before_simulation(self) -> None:
        plugin = NailClipperCompoundLeverAdapter()
        calls: list[tuple[str, str]] = []
        execution = SimpleNamespace(
            plan=SimpleNamespace(capsule_id="nail_clipper_compound_lever"),
            adapter=SimpleNamespace(family_spec=_family_spec()),
            require_geometry_proof=lambda joint_id, proof_kind: calls.append(
                (joint_id, proof_kind)
            ),
        )

        plugin.validate_execution_geometry_proofs(execution)

        self.assertEqual(
            calls,
            [
                ("jaw_revolute", "revolute.pin_bore.v1"),
                ("lever_revolute", "revolute.pin_bore.v1"),
            ],
        )

    def test_completed_trace_matches_worker_identity_contract(self) -> None:
        class _Document:
            def __init__(self, payload: dict[str, object]) -> None:
                self.payload = payload

            def to_dict(self) -> dict[str, object]:
                return dict(self.payload)

        plugin = NailClipperCompoundLeverAdapter()
        execution = SimpleNamespace(
            artifact_branch="physical",
            adapter=SimpleNamespace(adapter_id="derived_compound_linkage_v1"),
            mechanism_plugin=plugin,
            plan=SimpleNamespace(
                package_binding_sha256="p" * 64,
                capsule_id="nail_clipper_compound_lever",
                to_dict=_Document({"capsule_id": "nail_clipper_compound_lever"}).to_dict,
            ),
            load_case=SimpleNamespace(
                load_case_id="task_output_angle_roundtrip",
                to_dict=_Document({"load_case_id": "task_output_angle_roundtrip"}).to_dict,
            ),
        )
        samples = [{"time_s": 0.0, "phase_id": "initial"}]

        executed_phases = [
            {
                "phase_id": "close_press",
                "wrench_kind": "torque",
                "direction_abs": [0.0, -1.0, 0.0],
                "magnitude_si": 0.35,
                "duration_s": 0.45,
            }
        ]
        payload = _completed_trace_payload(
            execution, samples, executed_phases
        )
        expected = {
            "status": "COMPLETED",
            "validator_source_sha256": lane_execution_source_sha256(),
            "branch": execution.artifact_branch,
            "adapter_id": execution.adapter.adapter_id,
            "mechanism_plugin_id": plugin.plugin_id,
            "package_binding_sha256": execution.plan.package_binding_sha256,
            "capsule_id": execution.plan.capsule_id,
            "load_case_id": execution.load_case.load_case_id,
            "plan_sha256": canonical_sha256(execution.plan.to_dict()),
            "load_case_sha256": canonical_sha256(execution.load_case.to_dict()),
        }
        mismatches = [
            key for key, value in expected.items() if payload.get(key) != value
        ]

        self.assertEqual(mismatches, [])
        self.assertEqual(payload["plugin_id"], plugin.plugin_id)
        self.assertEqual(payload["samples"], samples)
        self.assertEqual(payload["executed_phases"], executed_phases)
        self.assertEqual(
            payload["executed_load_sha256"],
            canonical_sha256(executed_phases),
        )

    def test_adaptive_stop_is_rejected_without_termination_evidence(self) -> None:
        load_case = SimpleNamespace(
            phases=(
                SimpleNamespace(
                    phase_id="adaptive_close",
                    stop_condition="position_at_least",
                ),
            )
        )

        with self.assertRaises(MechanismValidationError) as caught:
            _require_fixed_duration_phases(load_case)
        self.assertEqual(
            caught.exception.code,
            "NAIL_CLIPPER_ADAPTIVE_STOP_UNSUPPORTED",
        )

    def test_exact_legacy_project_adapter_migration_is_declared(self) -> None:
        plugin = NailClipperCompoundLeverAdapter()

        self.assertEqual(
            plugin.legacy_project_adapter_migrations,
            (
                {
                    "basename": "nail_clipper_compound_lever.py",
                    "sha256": (
                        "65166c438501231c57f58288495894a22"
                        "bee9e7ad68fecb484da18343dc2a4a8"
                    ),
                    "size_bytes": 12612,
                    "legacy_mechanism_kind": (
                        "project_nail_clipper_compound_lever"
                    ),
                    "installed_mechanism_kind": (
                        "nail_clipper_compound_lever"
                    ),
                    "template_id": "joint.nail_clipper_compound_lever",
                    "variant_id": "clipper.compound_lever",
                },
            ),
        )

    def test_materialize_routes_through_step_family_collision(self) -> None:
        plugin = NailClipperCompoundLeverAdapter()
        adapter = SimpleNamespace(family_spec=_family_spec())
        plan = SimpleNamespace(capsule_id="nail_clipper_compound_lever")
        materialized = SimpleNamespace(collider_fidelity={"passed": True})

        with patch(
            "simulation_ready_cad_workflow.validation_pipeline.mechanism."
            "family_collision.materialize_family_collision_adapter",
            return_value=materialized,
        ) as materialize:
            result = plugin.materialize(
                plan,
                adapter,
                workspace_root=Path("/workspace"),
                output_dir=Path("/workspace/output"),
            )

        self.assertIs(result, materialized)
        self.assertEqual(plugin.plugin_id, "nail_clipper_compound_lever_v1")
        self.assertIn("step_collider_binding", plugin.extra_family_spec_fields)
        self.assertNotIn("project_adapter_file", plugin.extra_family_spec_fields)
        body_specs = materialize.call_args.args[2]
        self.assertTrue(all(isinstance(item, CompoundBodySpec) for item in body_specs))
        self.assertEqual(
            tuple(item.body_id for item in body_specs),
            (
                "stationary_cutting_jaw",
                "actuating_lever",
                "moving_cutting_jaw",
            ),
        )

    def test_physical_topology_uses_two_bushings_and_no_ideal_links(self) -> None:
        class _Vector:
            def __init__(self, x: float, y: float, z: float) -> None:
                self.x = x
                self.y = y
                self.z = z

        class _Quaternion:
            def __init__(self, e0: float, e1: float, e2: float, e3: float) -> None:
                self.e0 = e0
                self.e1 = e1
                self.e2 = e2
                self.e3 = e3

        class _CollisionModel:
            def __init__(self) -> None:
                self.family = None

            def SetFamily(self, family: int) -> None:
                self.family = family

            def DisallowCollisionsWith(self, _family: int) -> None:
                raise AssertionError("physical topology disabled a body pair")

        class _Body:
            def __init__(self, body_spec) -> None:
                self.body_spec = body_spec
                self.collision = _CollisionModel()

            def GetCollisionModel(self):
                return self.collision

            def GetPos(self):
                return _Vector(*self.body_spec.center_abs_m)

            def GetRot(self):
                return _Quaternion(1.0, 0.0, 0.0, 0.0)

        class _Solver:
            def SetMaxIterations(self, value: int) -> None:
                self.max_iterations = value

            def SetTolerance(self, value: float) -> None:
                self.tolerance = value

        class _System:
            def __init__(self) -> None:
                self.solver = None
                self.items = []

            def SetSolver(self, solver) -> None:
                self.solver = solver

            def SetGravitationalAcceleration(self, _value) -> None:
                pass

            def Add(self, value) -> None:
                self.items.append(value)

            def AddLink(self, _value) -> None:
                raise AssertionError("physical topology constructed an ideal link")

            def GetNumLinks(self) -> int:
                return 0

        class _Matrix:
            def __init__(self) -> None:
                self.diagonal = {}

            def SetZero(self) -> None:
                self.diagonal.clear()

            def SetItem(self, row: int, column: int, value: float) -> None:
                if row == column:
                    self.diagonal[row] = value

        class _Material:
            def SetFriction(self, _value) -> None:
                pass

        class _LoadContainer:
            def __init__(self) -> None:
                self.loads = []

            def Add(self, value) -> None:
                self.loads.append(value)

        class _Bushing:
            def __init__(self, child, parent, frame, stiffness, damping) -> None:
                self.child = child
                self.parent = parent
                self.frame = frame
                self.stiffness = stiffness
                self.damping = damping

        def forbidden_ideal_link():
            raise AssertionError("physical topology constructed an ideal link")

        chrono = SimpleNamespace(
            ChSystemNSC=_System,
            ChContactMaterialNSC=_Material,
            ChVector3d=_Vector,
            ChQuaterniond=_Quaternion,
            ChSolverADMM=_Solver,
            ChMatrix66d=_Matrix,
            ChLoadContainer=_LoadContainer,
            ChLoadBodyBodyBushingGeneric=_Bushing,
            ChFramed=lambda position, rotation: (position, rotation),
            ChLinkLockRevolute=forbidden_ideal_link,
            ChLinkLockPointPlane=forbidden_ideal_link,
        )
        plugin = NailClipperCompoundLeverAdapter()
        execution = SimpleNamespace(
            branch="physical",
            artifact_branch="physical",
            adapter=SimpleNamespace(adapter_id="derived_compound_linkage_v1"),
            mechanism_plugin=plugin,
            plan=SimpleNamespace(
                package_binding_sha256="p" * 64,
                capsule_id="nail_clipper_compound_lever",
                to_dict=lambda: {
                    "capsule_id": "nail_clipper_compound_lever"
                },
            ),
            load_case=SimpleNamespace(
                load_case_id="task_output_angle_roundtrip",
                step_s=0.0005,
                to_dict=lambda: {
                    "load_case_id": "task_output_angle_roundtrip"
                },
            ),
        )
        spec = plugin.parse_family_spec(_family_spec())
        observed: list[_Body] = []

        def make_body(_chrono, *, body_spec, **_kwargs):
            body = _Body(body_spec)
            observed.append(body)
            return body, {"body_id": body_spec.body_id}

        with patch(
            "simulation_ready_cad_workflow.validation_pipeline.mechanism."
            "family_collision.make_family_body",
            side_effect=make_body,
        ):
            topology = _build_nail_clipper_topology(
                chrono,
                execution,
                Path("/workspace"),
                spec,
                {"package_binding_sha256": "p" * 64},
            )

        self.assertEqual(len(observed), 3)
        self.assertEqual(len(topology["compound_bushings"]), 2)
        self.assertEqual(topology["ideal_links"], [])
        self.assertEqual(topology["link_count"], 0)
        self.assertEqual(
            len(topology["load_container"].loads),
            2,
        )
        self.assertEqual(
            [body.collision.family for body in observed],
            [1, 2, 3],
        )
        self.assertEqual(topology["system"].solver.max_iterations, 500)
        executed_phases = [
            {
                "phase_id": "close_press",
                "wrench_kind": "torque",
                "direction_abs": [0.0, -1.0, 0.0],
                "magnitude_si": 0.35,
                "duration_s": 0.45,
            }
        ]
        trace = _completed_nail_clipper_trace(
            execution,
            spec,
            [{"time_s": 0.0, "phase_id": "initial"}],
            executed_phases,
            topology,
            [{"support_id": "jaw-restoring-torsion-spring-v1"}],
            chrono_version="fake",
        )
        self.assertEqual(trace["engine"]["internal_ideal_joint_count"], 0)
        self.assertEqual(trace["engine"]["ideal_links"], [])
        self.assertEqual(len(trace["engine"]["compound_bushings"]), 2)
        self.assertEqual(trace["engine"]["compliant_supports"], [])
        self.assertEqual(len(trace["engine"]["constitutive_supports"]), 1)
        semantics = trace["branch_semantics"]
        self.assertFalse(semantics["replaced_contacts_disabled"])
        self.assertEqual(
            semantics["support_mode"],
            "finite_revolute_bushings_and_native_step_contact",
        )
        self.assertEqual(
            semantics["same_external_load_sha256"],
            canonical_sha256(executed_phases),
        )
        self.assertFalse(semantics["direct_output_actuation"])
        self.assertFalse(semantics["direct_output_actuation_used"])
        self.assertEqual(
            semantics["runtime_bindings"],
            {
                "fixed_body_id": "stationary_cutting_jaw",
                "input_body_id": "actuating_lever",
                "output_body_id": "moving_cutting_jaw",
            },
        )
        self.assertEqual(semantics["task_input_body_id"], "actuating_lever")
        self.assertEqual(
            semantics["unique_task_output_body_id"],
            "moving_cutting_jaw",
        )
        self.assertEqual(
            semantics["runtime_body_roles"],
            {
                "fixed_body_ids": ["stationary_cutting_jaw"],
                "input_body_id": "actuating_lever",
                "output_body_id": "moving_cutting_jaw",
            },
        )

    def test_prior_task_input_is_zero_before_phase_transition(self) -> None:
        events: list[tuple[str, object]] = []

        class _Torque:
            vector = (0.0, -0.35, 0.0)

            def SetTorque(self, vector, _local: bool) -> None:
                self.vector = tuple(vector)
                events.append(("set", self.vector))

        torque = _Torque()
        chrono = SimpleNamespace(ChVector3d=lambda *items: tuple(items))

        def transition(phase_id: str) -> None:
            events.append(("transition", (phase_id, torque.vector)))

        _transition_task_input_phase(
            torque,
            chrono,
            "close_hold",
            prior_phase_open=True,
            transition=transition,
        )

        self.assertEqual(
            events,
            [
                ("set", (0.0, 0.0, 0.0)),
                ("transition", ("close_hold", (0.0, 0.0, 0.0))),
            ],
        )

    def test_release_before_transition_satisfies_wrench_ledger(self) -> None:
        class _Vector:
            def __init__(self, x: float, y: float, z: float) -> None:
                self.x = x
                self.y = y
                self.z = z

        class _NativeTorque:
            def __init__(self, _body, value, _local: bool) -> None:
                self.value = value

            def SetTorque(self, value, _local: bool) -> None:
                self.value = value

        chrono = SimpleNamespace(
            ChVector3d=_Vector,
            ChLoadBodyTorque=_NativeTorque,
        )
        plugin_id = "nail_clipper_compound_lever_v1"
        ledger = WrenchLedger(
            plugin_id=plugin_id,
            capability_registration=plugin_registration_for(plugin_id),
            policies=wrench_policies_for_plugin(plugin_id),
            declared_input_body_ids=("actuating_lever",),
            declared_output_body_ids=("moving_cutting_jaw",),
            branch="physical",
            load_case_id="task_output_angle_roundtrip",
            program_phases=(
                {
                    "phase_id": "close_press",
                    "wrench_kind": "torque",
                    "direction_abs": [0.0, -1.0, 0.0],
                    "magnitude_si": 0.35,
                    "duration_s": 0.45,
                },
                {
                    "phase_id": "close_hold",
                    "wrench_kind": "none",
                    "direction_abs": None,
                    "magnitude_si": 0.0,
                    "duration_s": 0.10,
                },
            ),
        )

        with activate_wrench_ledger(ledger):
            torque = create_body_torque(
                chrono,
                object(),
                _Vector(0.0, 0.0, 0.0),
                False,
                declaration=standard_wrench_declaration(
                    plugin_id=plugin_id,
                    profile="task_input",
                    wrench_kind="torque",
                    load_id="declared-input-torque",
                    owner=plugin_id,
                    target_body="actuating_lever",
                    source_role="declared_task_input",
                    target_role="declared_input",
                    geometry_gate="actuating_lever_axis",
                    active_interval="active_load_phase",
                ),
            )
            transition_current_wrench_phase("close_press")
            torque.SetTorque(_Vector(0.0, -0.35, 0.0), False)
            _transition_task_input_phase(
                torque,
                chrono,
                "close_hold",
                prior_phase_open=True,
                transition=transition_current_wrench_phase,
            )
            set_current_wrench_phase(None)
            executed_phases = [
                {
                    "phase_id": "close_press",
                    "wrench_kind": "torque",
                    "direction_abs": [0.0, -1.0, 0.0],
                    "magnitude_si": 0.35,
                    "duration_s": 0.45,
                },
                {
                    "phase_id": "close_hold",
                    "wrench_kind": "none",
                    "direction_abs": None,
                    "magnitude_si": 0.0,
                    "duration_s": 0.10,
                },
            ]
            executed_load_sha256 = canonical_sha256(executed_phases)
            with self.assertRaises(MechanismValidationError) as wrong_order:
                ledger.bind_executed_schedule(list(reversed(executed_phases)))
            self.assertEqual(
                wrong_order.exception.code,
                "WRENCH_EXECUTED_SCHEDULE_MISMATCH",
            )
            with self.assertRaises(MechanismValidationError) as wrong_hash:
                ledger.bind_executed_schedule(
                    executed_phases,
                    executed_load_sha256="0" * 64,
                )
            self.assertEqual(
                wrong_hash.exception.code,
                "WRENCH_EXECUTED_SCHEDULE_HASH_MISMATCH",
            )
            ledger.bind_executed_schedule(
                executed_phases,
                executed_load_sha256=executed_load_sha256,
            )
            audit = ledger.verify_for_certification()
        self.assertEqual(audit["release_residual_count"], 0)
        self.assertTrue(audit["phase_barriers"][0]["exit_all_zero"])
        self.assertEqual(
            audit["executed_phase_ids"],
            ["close_press", "close_hold"],
        )
        self.assertEqual(audit["executed_load_sha256"], executed_load_sha256)


if __name__ == "__main__":
    unittest.main()
