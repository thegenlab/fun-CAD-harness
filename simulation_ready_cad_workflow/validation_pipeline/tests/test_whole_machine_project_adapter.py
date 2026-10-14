from __future__ import annotations

import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from simulation_ready_cad_workflow.validation_pipeline.mechanism import (
    family_plugins,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.adapters import (
    AdapterSpec,
    adapter_to_dict,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.io import (
    read_json,
    write_json_atomic,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.models import (
    MechanismPlan,
    MechanismValidationError,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.whole_machine import (
    build_execution_plan,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.whole_machine_worker import (
    run,
)


_HASH = "0" * 64
_PLUGIN_ID = "project.whole_machine_fixture_v1"
_BACKEND = "project_whole_machine_fixture_v1"
_PROJECT_ADAPTER_SOURCE = f'''\
from types import SimpleNamespace
from simulation_ready_cad_workflow.validation_pipeline.mechanism.family_plugins import MechanismAdapter, GeneralizedCoordinateSpec

class ProjectAdapter(MechanismAdapter):
    plugin_id = "{_PLUGIN_ID}"
    families = frozenset({{"project_family"}})
    coordinate = GeneralizedCoordinateSpec("position", "velocity", "m", "m/s", "force", "N")
    worker_backend = "project_worker_fixture_v1"
    whole_machine_backend = "{_BACKEND}"
    wrench_policy_profiles = frozenset()

    def load_cases(self, plan):
        return ()

    def expected_reduced_ideal_links(self, plan, adapter):
        return ()

    def load_whole_machine_plan(self, value):
        if value.get("coordinate_kind") != "project_fixture":
            return None
        return SimpleNamespace(**value)

    def simulate_whole_machine(self, plan, *, workspace_root):
        return {{
            "schema_version": "0.1",
            "status": "PROJECT_WHOLE_MACHINE_EXECUTED",
            "passed": True,
            "plugin_id": self.plugin_id,
        }}

MECHANISM_ADAPTER = ProjectAdapter()
'''


def _project_plan(package: Path) -> MechanismPlan:
    relative = "mechanism_templates/project_adapter.py"
    source = package / relative
    source.parent.mkdir(parents=True)
    source.write_text(_PROJECT_ADAPTER_SOURCE, encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    adapter = AdapterSpec(
        adapter_id="project_whole_machine_data_v1",
        family="project_family",
        required_source_hashes=frozenset(),
        supported_programs=frozenset(),
        family_spec={"project_adapter_file": relative},
    )
    return MechanismPlan(
        package_dir=str(package),
        intake_report="intake.json",
        intake_report_sha256=_HASH,
        package_id="whole_machine_project_adapter_test",
        design_version="1.0.0",
        requirements_sha256=_HASH,
        package_binding_sha256=_HASH,
        capsule_id="project_capsule",
        capsule_file="capsule.json",
        family="project_family",
        protocol_id="project_protocol",
        fixed_group_ids=("fixed",),
        moving_group_ids=("moving",),
        axis_abs=(0.0, 0.0, 1.0),
        joint_origin_abs_m=(0.0, 0.0, 0.0),
        gravity_abs_m_s2=(0.0, 0.0, -9.81),
        joint_frame_id="joint",
        input_port_id="input",
        input_frame_id="input_frame",
        input_point_abs_m=(0.0, 0.0, 0.0),
        observable_id="output",
        observable_frame_id="output_frame",
        observable_point_abs_m=(0.0, 0.0, 0.0),
        task_ids=("task",),
        characteristic_length_m=0.1,
        moving_extents_m=(0.1, 0.1, 0.1),
        adapter_id=adapter.adapter_id,
        artifact_hashes={relative: digest},
        adapter_spec=adapter_to_dict(adapter),
    )


class WholeMachineProjectAdapterTest(unittest.TestCase):
    def test_persisted_whole_machine_binding_error_is_harness_owned(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = _project_plan(root / "package")
            certificate_path = root / "certificate.json"
            replacement_path = root / "replacement.json"
            write_json_atomic(
                certificate_path,
                {
                    "verdict": "KINEMATIC_CERTIFIED",
                    "package": {"package_binding_sha256": "wrong-binding"},
                },
            )
            write_json_atomic(
                replacement_path,
                {"status": "ACTIVE"},
            )

            with self.assertRaises(MechanismValidationError) as caught:
                build_execution_plan(
                    plan,
                    workspace_root=root,
                    output_dir=root,
                    certificate_file=certificate_path,
                    replacement_file=replacement_path,
                )

        self.assertEqual(
            caught.exception.failure_envelope["owner"],
            "harness_code",
        )

    def test_worker_loads_and_runs_the_project_adapter_from_mechanism_plan(
        self,
    ) -> None:
        previous_adapters = dict(family_plugins._PROJECT_ADAPTERS)
        previous_cache = dict(family_plugins._PROJECT_ADAPTER_CACHE)
        try:
            with TemporaryDirectory() as temporary:
                root = Path(temporary)
                plan = _project_plan(root / "package")
                mechanism_plan_path = root / "mechanism_plan.json"
                execution_path = root / "execution_plan.json"
                certificate_path = root / "certificate.json"
                replacement_path = root / "replacement.json"
                output_path = root / "result.json"
                write_json_atomic(mechanism_plan_path, plan.to_dict())
                write_json_atomic(
                    certificate_path,
                    {"verdict": "KINEMATIC_CERTIFIED"},
                )
                write_json_atomic(
                    replacement_path,
                    {
                        "status": "ACTIVE",
                        "package_binding_sha256": plan.package_binding_sha256,
                    },
                )
                write_json_atomic(
                    execution_path,
                    {
                        "coordinate_kind": "project_fixture",
                        "package_binding_sha256": plan.package_binding_sha256,
                        "capsule_id": plan.capsule_id,
                        "certificate_file": certificate_path.name,
                        "certificate_sha256": hashlib.sha256(
                            certificate_path.read_bytes()
                        ).hexdigest(),
                        "replacement_file": replacement_path.name,
                        "replacement_sha256": hashlib.sha256(
                            replacement_path.read_bytes()
                        ).hexdigest(),
                        "mechanism_plugin_id": _PLUGIN_ID,
                        "worker_backend": _BACKEND,
                    },
                )

                payload = run(
                    execution_path,
                    mechanism_plan_path,
                    output_path,
                    workspace_root=root,
                )

                self.assertEqual(
                    payload["status"], "PROJECT_WHOLE_MACHINE_EXECUTED"
                )
                self.assertEqual(payload["plugin_id"], _PLUGIN_ID)
                self.assertEqual(read_json(output_path), payload)

                changed_execution = read_json(execution_path)
                changed_execution["certificate_sha256"] = "changed"
                write_json_atomic(execution_path, changed_execution)
                with self.assertRaises(MechanismValidationError) as caught:
                    run(
                        execution_path,
                        mechanism_plan_path,
                        output_path,
                        workspace_root=root,
                    )
                self.assertEqual(
                    caught.exception.failure_envelope["owner"],
                    "harness_code",
                )
        finally:
            family_plugins._PROJECT_ADAPTERS.clear()
            family_plugins._PROJECT_ADAPTERS.update(previous_adapters)
            family_plugins._PROJECT_ADAPTER_CACHE.clear()
            family_plugins._PROJECT_ADAPTER_CACHE.update(previous_cache)


if __name__ == "__main__":
    unittest.main()
