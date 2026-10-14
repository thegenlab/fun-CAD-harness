from __future__ import annotations

from copy import deepcopy
import unittest

from simulation_ready_cad_workflow.validation_pipeline.mechanism.adapters import (
    adapter_to_dict,
    get_adapter,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.ideal_comparison import (
    compare_physical_to_ideal,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.ideal_reference import (
    generate_ideal_reference_trace,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.io import (
    canonical_sha256,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.models import (
    LoadCase,
    LoadPhase,
    MechanismPlan,
    MechanismValidationError,
)


_HASH = "0" * 64


def _plan() -> MechanismPlan:
    adapter = get_adapter("mcmaster_6725k41_8438k4_v1")
    return MechanismPlan(
        package_dir="package",
        intake_report="intake.json",
        intake_report_sha256=_HASH,
        package_id="ideal_comparison_test",
        design_version="1.0.0",
        requirements_sha256=_HASH,
        package_binding_sha256=_HASH,
        capsule_id="test_capsule",
        capsule_file="capsule.json",
        family="prismatic",
        protocol_id="test_protocol",
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
        characteristic_length_m=0.2,
        moving_extents_m=(0.1, 0.1, 0.1),
        adapter_id=adapter.adapter_id,
        adapter_spec=adapter_to_dict(adapter),
    )


def _load() -> LoadCase:
    return LoadCase(
        load_case_id="ideal_case",
        family="prismatic",
        program="task_range_roundtrip",
        purpose="nominal",
        phases=(
            LoadPhase(
                phase_id="drive",
                wrench_kind="force",
                direction_abs=(0.0, 0.0, 1.0),
                magnitude_si=1.0,
                duration_s=1.0,
            ),
        ),
        moving_mass_kg=1.0,
        step_s=0.001,
        sample_interval_s=0.05,
    )


def _physical(plan: MechanismPlan, load: LoadCase) -> dict:
    return {
        "status": "COMPLETED",
        "branch": "physical",
        "package_binding_sha256": plan.package_binding_sha256,
        "capsule_id": plan.capsule_id,
        "load_case_id": load.load_case_id,
        "plan_sha256": canonical_sha256(plan.to_dict()),
        "load_case_sha256": canonical_sha256(load.to_dict()),
        "samples": [
            {
                "time_s": 0.0,
                "phase_id": "initial",
                "generalized_progress_m": 0.0,
            },
            {
                "time_s": 1.0,
                "phase_id": "drive",
                "generalized_progress_m": 0.04,
            },
        ],
    }


class IdealComparisonTests(unittest.TestCase):
    def test_same_saved_pose_reference_passes(self) -> None:
        plan = _plan()
        load = _load()
        physical = _physical(plan, load)
        ideal = generate_ideal_reference_trace(plan, load, physical)

        result = compare_physical_to_ideal(plan, load, physical, ideal)

        self.assertEqual(result["verdict"], "PASS")
        self.assertEqual(result["metrics"]["max_absolute_error"], 0.0)
        self.assertFalse(result["contact_evidence_authority"])

    def test_output_deviation_fails_with_worst_saved_sample(self) -> None:
        plan = _plan()
        load = _load()
        physical = _physical(plan, load)
        ideal = generate_ideal_reference_trace(plan, load, physical)
        changed = deepcopy(ideal)
        changed["samples"][-1]["generalized_progress_m"] = 0.06

        result = compare_physical_to_ideal(plan, load, physical, changed)

        self.assertEqual(result["verdict"], "CLAIM_MISMATCH")
        self.assertEqual(result["metrics"]["worst_sample"]["sample_index"], 1)
        self.assertGreater(
            result["metrics"]["max_absolute_error"],
            result["threshold"]["max_absolute_error"],
        )

    def test_sample_time_mismatch_is_rejected(self) -> None:
        plan = _plan()
        load = _load()
        physical = _physical(plan, load)
        ideal = generate_ideal_reference_trace(plan, load, physical)
        changed = deepcopy(ideal)
        changed["samples"][-1]["time_s"] = 1.1

        with self.assertRaises(MechanismValidationError) as caught:
            compare_physical_to_ideal(plan, load, physical, changed)

        self.assertEqual(
            caught.exception.code,
            "IDEAL_COMPARISON_SAMPLE_IDENTITY_MISMATCH",
        )


if __name__ == "__main__":
    unittest.main()
