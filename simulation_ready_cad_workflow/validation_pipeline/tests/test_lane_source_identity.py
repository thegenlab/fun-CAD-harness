from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from simulation_ready_cad_workflow.validation_pipeline.mechanism import (
    finalize,
    io,
    whole_machine_worker,
    worker,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.models import (
    MechanismValidationError,
)


class LaneExecutionSourceIdentityTest(unittest.TestCase):
    def test_real_closure_contains_lane_code_and_stops_at_stage_boundaries(
        self,
    ) -> None:
        source_root = Path(io.__file__).resolve().parent.parent
        sources = io._source_import_closure(
            source_root,
            seeds=io._LANE_EXECUTION_SOURCE_SEEDS,
            boundaries=io._LANE_EXECUTION_SOURCE_BOUNDARIES,
        )
        relative_sources = {
            source.relative_to(source_root).as_posix() for source in sources
        }

        self.assertTrue(
            {
                "mechanism/lane_worker.py",
                "mechanism/worker.py",
                "mechanism/ideal_reference.py",
                "mechanism/ideal_comparison.py",
                "mechanism/comparison.py",
                "mechanism/physical_ideal_comparison.py",
                "mechanism/bicycle_chain_runtime.py",
                "mechanism/bicycle_ground_roll_runtime.py",
                "mechanism/click_pen_runtime.py",
                "mechanism/clock_gear_runtime.py",
                "mechanism/compound_runtime.py",
                "mechanism/detent_slider_runtime.py",
                "mechanism/project_adapter_worker.py",
                "mechanism/rack_pinion_runtime_v10.py",
                "mechanism/scissor_lift_runtime.py",
                "contracts/failure_envelope.py",
                "runtime/bounded_subprocess.py",
                "runtime/system_probe.py",
            }.issubset(relative_sources)
        )
        self.assertTrue(
            {
                "mechanism/dispatch.py",
                "mechanism/finalize.py",
                "mechanism/reuse.py",
                "mechanism/whole_machine.py",
                "mechanism/whole_machine_worker.py",
            }.isdisjoint(relative_sources)
        )
        self.assertEqual(
            io._LANE_EXECUTION_SOURCE_RESOURCES,
            (
                "mechanism/templates/"
                "bicycle_chain_sprocket_drive_v2.json",
            ),
        )

    def test_digest_changes_only_for_files_inside_the_execution_closure(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source_root = Path(temporary)
            mechanism = source_root / "mechanism"
            runtime = source_root / "runtime"
            contracts = source_root / "contracts"
            templates = mechanism / "templates"
            for directory in (mechanism, runtime, contracts, templates):
                directory.mkdir(parents=True, exist_ok=True)
            (mechanism / "lane_worker.py").write_text(
                "from .worker import run\nfrom .reuse import adopt\n",
                encoding="utf-8",
            )
            (mechanism / "worker.py").write_text(
                "from ..runtime.bounded_subprocess import run\n"
                "from ..contracts.failure_envelope import wrap\n",
                encoding="utf-8",
            )
            bounded = runtime / "bounded_subprocess.py"
            bounded.write_text(
                "from .system_probe import probe\n",
                encoding="utf-8",
            )
            system_probe = runtime / "system_probe.py"
            system_probe.write_text("def probe(): return 1\n", encoding="utf-8")
            (contracts / "failure_envelope.py").write_text(
                "def wrap(): return 1\n",
                encoding="utf-8",
            )
            reuse = mechanism / "reuse.py"
            reuse.write_text("def adopt(): return 1\n", encoding="utf-8")
            dispatch = mechanism / "dispatch.py"
            dispatch.write_text("MODE = 1\n", encoding="utf-8")
            resource = templates / "bicycle_chain_sprocket_drive_v2.json"
            resource.write_text('{"mode":1}\n', encoding="utf-8")

            def digest() -> str:
                return io._source_import_closure_sha256(
                    source_root,
                    seeds=("mechanism.lane_worker",),
                    boundaries=frozenset({"mechanism.reuse"}),
                    resources=(
                        "mechanism/templates/"
                        "bicycle_chain_sprocket_drive_v2.json",
                    ),
                )

            original = digest()
            dispatch.write_text("MODE = 2\n", encoding="utf-8")
            reuse.write_text("def adopt(): return 2\n", encoding="utf-8")
            self.assertEqual(digest(), original)

            system_probe.write_text("def probe(): return 2\n", encoding="utf-8")
            self.assertNotEqual(digest(), original)
            system_probe.write_text("def probe(): return 1\n", encoding="utf-8")

            resource.write_text('{"mode":2}\n', encoding="utf-8")
            self.assertNotEqual(digest(), original)

    def test_public_identity_is_a_sha256(self) -> None:
        identity = io.lane_execution_source_sha256()
        self.assertEqual(len(identity), 64)
        int(identity, 16)

    def test_lane_and_whole_machine_verify_their_own_source_identity(
        self,
    ) -> None:
        lane_identity = "a" * 64
        whole_machine_identity = "b" * 64
        with patch.dict(
            os.environ,
            {io.MECHANISM_SOURCE_SHA256_ENV: lane_identity},
        ), patch.object(
            worker,
            "lane_execution_source_sha256",
            return_value=lane_identity,
        ):
            self.assertEqual(
                worker._verify_validator_source_binding(), lane_identity
            )

        with patch.dict(
            os.environ,
            {io.MECHANISM_SOURCE_SHA256_ENV: whole_machine_identity},
        ), patch.object(
            whole_machine_worker,
            "mechanism_source_sha256",
            return_value=whole_machine_identity,
        ):
            self.assertEqual(
                whole_machine_worker._verify_validator_source_binding(),
                whole_machine_identity,
            )

    def test_adapter_trace_must_carry_the_current_lane_identity(self) -> None:
        lane_identity = "a" * 64
        plan_document = {"capsule_id": "capsule"}
        load_document = {"load_case_id": "load"}
        payload = {
            "status": "COMPLETED",
            "validator_source_sha256": lane_identity,
            "branch": "physical",
            "adapter_id": "adapter",
            "mechanism_plugin_id": "project.example_v1",
            "package_binding_sha256": "b" * 64,
            "capsule_id": "capsule",
            "load_case_id": "load",
            "plan_sha256": io.canonical_sha256(plan_document),
            "load_case_sha256": io.canonical_sha256(load_document),
            "samples": [{"time_s": 0.0}],
        }
        plugin = SimpleNamespace(
            plugin_id="project.example_v1",
            geometry_proof_kinds=frozenset(),
            required_geometry_proofs=lambda *_args, **_kwargs: (),
            simulate=lambda *_args: dict(payload),
        )
        execution = SimpleNamespace(
            artifact_branch="physical",
            adapter=SimpleNamespace(adapter_id="adapter"),
            mechanism_plugin=plugin,
            plan=SimpleNamespace(
                package_binding_sha256="b" * 64,
                capsule_id="capsule",
                to_dict=lambda: dict(plan_document),
            ),
            load_case=SimpleNamespace(
                load_case_id="load",
                to_dict=lambda: dict(load_document),
            ),
        )

        with patch.object(
            worker,
            "lane_execution_source_sha256",
            return_value=lane_identity,
        ):
            self.assertEqual(
                worker._simulate(execution, Path.cwd()),
                payload,
            )
            for recorded in (None, "c" * 64):
                with self.subTest(recorded=recorded):
                    changed = dict(payload)
                    if recorded is None:
                        changed.pop("validator_source_sha256")
                    else:
                        changed["validator_source_sha256"] = recorded
                    plugin.simulate = lambda *_args, value=changed: dict(value)
                    with self.assertRaises(MechanismValidationError) as caught:
                        worker._simulate(execution, Path.cwd())
                    self.assertEqual(
                        caught.exception.code,
                        "MECHANISM_ADAPTER_SIMULATE_RESULT_INVALID",
                    )

    def test_finalize_checks_lane_trace_identity_but_not_ideal_reference(
        self,
    ) -> None:
        lane_identity = "a" * 64
        plan_document = {"capsule_id": "capsule"}
        load_document = {"load_case_id": "load"}
        plan = SimpleNamespace(
            package_binding_sha256="b" * 64,
            capsule_id="capsule",
            to_dict=lambda: dict(plan_document),
        )
        load_case = SimpleNamespace(
            load_case_id="load",
            to_dict=lambda: dict(load_document),
        )
        trace = {
            "status": "COMPLETED",
            "branch": "physical",
            "package_binding_sha256": plan.package_binding_sha256,
            "capsule_id": plan.capsule_id,
            "load_case_id": load_case.load_case_id,
            "plan_sha256": io.canonical_sha256(plan_document),
            "load_case_sha256": io.canonical_sha256(load_document),
            "validator_source_sha256": lane_identity,
            "samples": [{"time_s": 0.0}, {"time_s": 0.1}],
        }
        with tempfile.TemporaryDirectory() as temporary:
            trace_path = Path(temporary) / "trace.json"
            io.write_trace_json_atomic(trace_path, trace)
            self.assertEqual(
                finalize._read_bound_trace(
                    trace_path,
                    branch="physical",
                    plan=plan,
                    load_case=load_case,
                    expected_validator_source_sha256=lane_identity,
                ),
                trace,
            )
            with self.assertRaises(MechanismValidationError) as caught:
                finalize._read_bound_trace(
                    trace_path,
                    branch="physical",
                    plan=plan,
                    load_case=load_case,
                    expected_validator_source_sha256="c" * 64,
                )
            self.assertEqual(caught.exception.code, "TRACE_IDENTITY_MISMATCH")

            ideal = dict(trace)
            ideal["branch"] = "ideal_reference"
            ideal.pop("validator_source_sha256")
            io.write_trace_json_atomic(trace_path, ideal)
            self.assertEqual(
                finalize._read_bound_trace(
                    trace_path,
                    branch="ideal_reference",
                    plan=plan,
                    load_case=load_case,
                ),
                ideal,
            )


if __name__ == "__main__":
    unittest.main()
