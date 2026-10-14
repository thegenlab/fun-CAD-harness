from __future__ import annotations

import json
import copy
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ..mechanism.io import write_json_atomic
from ..slim_harness.cli import create_project, copy_delivery, start, resume_project
from ..slim_harness.contracts import DeliveryError, load_scene, compile_scene
from ..slim_harness.review import review_passes
from ..slim_harness.workflow import SlimWorkflow, digest, ActiveWorkflowError
from ..slim_harness.geometry import GEOMETRY_IMPLEMENTATION_SHA256
from ..slim_harness.audit import AUDIT_IMPLEMENTATION_SHA256
from ..slim_harness.implementation import native_implementation_binding
from ..slim_harness.numerical import NUMERICAL_POLICY, effective_initial_step_seconds
from ..slim_harness.trace_storage import SampleRows, TraceStorageError, TraceWriter, trace_artifacts


class SlimWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.workspace=Path(self.temporary.name)
        self.project=create_project(self.workspace,"machine","A recognizable machine must move.")
        self.run=self.project/"runs"/"r01"
        self.scene=self.run/"design"/"scene.json"
        self.source=self.run/"design"/"part.step"
        self.source.write_text("a geometry-file snapshot",encoding="utf-8")
        write_json_atomic(self.scene,{"schema_version":"slim_scene_v1","units":"mm",
            "bodies":[{"id":"moving","source":"part.step","fixed":False}],
            "tests":[{"id":"move","purpose":"Check movement","duration_s":1,
                      "assertions":[{"id":"moved","purpose":"Movement","expression":"last(signal('time_s')) >= 1"}]}]})
        criteria=[{"id":"appearance","kind":"appearance","statement":"Recognizable","source_quote":"recognizable","verification":"Image"},
                  {"id":"function","kind":"function","statement":"Moves","source_quote":"move","verification":"Physical test"}]
        frozen={"schema_version":"slim_frozen_criteria_v1","task_file":str(self.project/"task.md"),
                "task_text":"A recognizable machine must move.","criteria":criteria,"reviewer_summary":"Source-bound criteria"}
        write_json_atomic(self.project/"criteria.json",frozen)
        state=json.loads((self.project/"workflow.json").read_text())
        state["events"]=[{"kind":"criteria","result":{"status":"ready","summary":frozen["reviewer_summary"],"criteria":criteria}}]
        write_json_atomic(self.project/"workflow.json",state)

    def workflow(self,**kwargs):
        return SlimWorkflow(self.workspace,self.project,**kwargs)


    def _numerical_metadata(self,test=None,max_step_s=.001):
        initial=effective_initial_step_seconds(test or {},max_step_s)
        return {"numerical_policy":NUMERICAL_POLICY,
                "numerical_settings":{"requested_max_step_s":max_step_s,"effective_initial_step_s":initial},
                "numerical_attempts":[{"max_step_s":initial}]}

    def _implemented_test_summary(self):
        return {"test_id":"move","physical_pass":True,"execution_implementation":native_implementation_binding(),
                **self._numerical_metadata(),
                "geometry_audit":{"passed":True,"audit_implementation_sha256":AUDIT_IMPLEMENTATION_SHA256}}

    def _partial_physics_fixture(self,*,cached_max_step_s=.001,**workflow_kwargs):
        workflow=self.workflow(**workflow_kwargs)
        old=self.run/"attempts"/"a001"
        old.mkdir(parents=True)
        mesh=old/"body.obj";mesh.write_text("bound mesh bytes; no CAD is executed")
        geometry={"schema_version":"slim_geometry_v1","errors":[],"scene_file":str(self.scene),
                  "scene_sha256":digest(self.scene),"geometry_implementation_sha256":GEOMETRY_IMPLEMENTATION_SHA256,
                  "model_file":str(old/"model.glb"),"bodies":[{
                      "id":"moving","source_file":str(self.source),"source_sha256":digest(self.source),
                      "mesh_file":str(mesh),"mesh_sha256":digest(mesh),"position_m":[0,0,0],"rotation_wxyz":[1,0,0,0],
                  }]}
        write_json_atomic(old/"geometry"/"geometry.json",geometry)
        scene=json.loads(self.scene.read_text())
        prepared=compile_scene(scene,geometry)  # No joints: this fixture performs no geometry queries.
        request={"scene":prepared,"test":copy.deepcopy(prepared["tests"][0]),"output":str(old/"tests/move/physical.trace.json"),
                 "numerical_settings":{"max_step_s":cached_max_step_s}}
        trace={"test_id":"move","physical_pass":True,"failure":None,"execution_implementation":native_implementation_binding(),
               **self._numerical_metadata(request["test"],cached_max_step_s),
               "samples":[{"time_s":0,"body_poses":{}},{"time_s":1,"body_poses":{}}],
               "signals":{"time_s":[0,1]},"assertions":[{"id":"moved","passed":True}]}
        write_json_atomic(old/"tests/move/request.json",request)
        write_json_atomic(old/"tests/move/physical.trace.json",trace)
        review={"status":"ready","criteria_sha256":workflow.criteria_sha256,"observations":[
            {"criterion_id":"appearance","verdict":"met","observed_fact":"fixture","evidence_paths":["fixture.png"]}],
            "test_coverage":[{"criterion_id":"function","appropriate":True,"test_ids":["move"]}]}
        write_json_atomic(old/"preflight_review.json",review)
        workflow._preview=Mock()
        workflow._motion=Mock(return_value=({"fixture":True},{"id":"move"}))
        workflow._review=Mock(side_effect=AssertionError("No model is allowed in this fixture"))
        workflow._finish_validation=Mock(return_value={"status":"validation_only"})
        return workflow,old,geometry,request,trace

    def _exercise_partial_fixture(self,workflow,*,reject_motion=False):
        def audit(_prepared,samples,output):
            result={"passed":True,"violations":[],"audit_implementation_sha256":AUDIT_IMPLEMENTATION_SHA256,
                    "checked_solver_samples":len(samples)}
            if reject_motion and output.name=="geometry-audit.json":
                result.update(passed=False,violations=[{"kind":"original_solid_interpenetration","time_s":1}])
            write_json_atomic(output,result)
            return result
        def render(_geometry,output,*_args,**_kwargs):
            output.parent.mkdir(parents=True,exist_ok=True)
            output.write_text("isolated image fixture")
            return output
        prefix="simulation_ready_cad_workflow.validation_pipeline.slim_harness.workflow."
        with patch(prefix+"prepare_geometry",side_effect=AssertionError("No CAD is allowed in this fixture")), \
             patch(prefix+"audit_motion",side_effect=audit) as auditor, \
             patch(prefix+"audit_saved_motion",new=auditor), \
             patch(prefix+"render_views",side_effect=render), \
             patch(prefix+"run_bounded_subprocess",return_value=SimpleNamespace(returncode=1,stdout=b"",stderr=b"isolated execution sentinel")) as native:
            result=workflow.run_revision()
        return result,native,auditor

    def test_motion_audit_failure_preserves_evidence_and_continues_remaining_tests(self):
        scene=json.loads(self.scene.read_text())
        second=copy.deepcopy(scene["tests"][0]);second["id"]="move_again"
        scene["tests"].append(second)
        write_json_atomic(self.scene,scene)
        workflow,old,_geometry,_request,_trace=self._partial_physics_fixture()
        review=json.loads((old/"preflight_review.json").read_text())
        review["test_coverage"][0]["test_ids"].append("move_again")
        write_json_atomic(old/"preflight_review.json",review)
        prefix="simulation_ready_cad_workflow.validation_pipeline.slim_harness.workflow."
        with patch(prefix+"release_trace_cache") as release:
            result,native,auditor=self._exercise_partial_fixture(workflow,reject_motion=True)
        self.assertEqual("HARNESS_ERROR",result["status"])
        native.assert_called_once()  # First trace is reused; the second test now gets its own attempt.
        self.assertEqual(2,auditor.call_count)  # Initial geometry and the first complete motion.
        self.assertTrue(release.called)
        workflow._motion.assert_called_once()
        self.assertEqual([{"id":"move"}],workflow._preview.call_args.args[3])
        workflow._finish_validation.assert_not_called()
        attempt=Path(result["attempt"])
        folder=attempt/"tests"/"move"
        self.assertTrue((attempt/"tests"/"move_again"/"test-outcome.json").exists())
        self.assertEqual([],result["unrun_tests"])
        self.assertFalse((attempt/"validated-physics.json").exists())
        self.assertTrue((folder/"physical.trace.json").is_file())
        self.assertTrue((folder/"final-pose.png").is_file())
        audit=json.loads((folder/"geometry-audit.json").read_text())
        self.assertFalse(audit["passed"])
        self.assertEqual(audit,result["issues"][0]["geometry_audit"])
        self.assertEqual({"fixture":True},result["issues"][0]["replay"])

    def test_partial_physics_exact_request_reuses_trace_but_runs_current_audit(self):
        workflow,old,_geometry,_request,trace=self._partial_physics_fixture()
        result,native,auditor=self._exercise_partial_fixture(workflow)
        self.assertEqual("validation_only",result["status"])
        native.assert_not_called()
        self.assertEqual(2,auditor.call_count)  # Initial pose plus every recorded native pose.
        self.assertEqual(trace["samples"],auditor.call_args.args[1])
        copied=json.loads((self.run/"attempts/a002/tests/move/physical.trace.json").read_text())
        self.assertEqual(str(old/"tests/move/physical.trace.json"),copied["reused_from"])

    def _check_native_budget(self,**workflow_kwargs):
        workflow,old,_geometry,request,_trace=self._partial_physics_fixture(**workflow_kwargs)
        expected_scene=copy.deepcopy(request["scene"])
        expected_test=copy.deepcopy(request["test"])
        request["test"]["purpose"]="A stale cached test must not replace this native run"
        write_json_atomic(old/"tests/move/request.json",request)
        result,native,_auditor=self._exercise_partial_fixture(workflow)
        self.assertEqual("HARNESS_ERROR",result["status"])
        native.assert_called_once()
        limits={"timeout_seconds":workflow_kwargs.get("physics_timeout_seconds",600),"max_rss_bytes":"auto"}
        self.assertEqual(limits["timeout_seconds"],native.call_args.kwargs["timeout_seconds"])
        self.assertEqual(limits["max_rss_bytes"],native.call_args.kwargs["max_rss_bytes"])
        actual=json.loads(Path(native.call_args.args[0][-1]).read_text())
        self.assertEqual(limits,actual["execution_limits"])
        self.assertEqual(expected_scene,actual["scene"])
        self.assertEqual(expected_test,actual["test"])
        self.assertEqual({"max_step_s":.001,"solver_iterations":160},actual["numerical_settings"])

    def test_native_budget_defaults_to_600_seconds(self):
        self._check_native_budget()

    def test_native_budget_override_is_recorded_without_changing_physics(self):
        self._check_native_budget(physics_timeout_seconds=1800)

    def test_native_budget_rejects_invalid_values_before_state_writes(self):
        before=(self.project/"workflow.json").read_bytes()
        for value in (0,-1,float("nan"),float("inf"),-float("inf")):
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,"positive and finite"):
                self.workflow(physics_timeout_seconds=value)
        self.assertEqual(before,(self.project/"workflow.json").read_bytes())
        self.assertFalse((self.project/"criteria.lock.json").exists())

    def test_native_budget_override_preserves_successful_trace_cache(self):
        workflow,*_=self._partial_physics_fixture(physics_timeout_seconds=1800)
        result,native,_auditor=self._exercise_partial_fixture(workflow)
        self.assertEqual("validation_only",result["status"])
        native.assert_not_called()
        request=json.loads((self.run/"attempts/a002/tests/move/request.json").read_text())
        self.assertEqual(1800,request["execution_limits"]["timeout_seconds"])

    def test_finer_max_step_rejects_coarse_successful_trace_and_records_request(self):
        workflow,old,_geometry,request,_trace=self._partial_physics_fixture(physics_max_step_seconds=.0005)
        expected_scene=copy.deepcopy(request["scene"]);expected_test=copy.deepcopy(request["test"])
        result,native,_auditor=self._exercise_partial_fixture(workflow)
        self.assertEqual("HARNESS_ERROR",result["status"])
        native.assert_called_once()
        actual=json.loads(Path(native.call_args.args[0][-1]).read_text())
        self.assertEqual({"max_step_s":.0005,"solver_iterations":160},actual["numerical_settings"])
        self.assertEqual(expected_scene,actual["scene"]);self.assertEqual(expected_test,actual["test"])

    def test_matching_fine_step_keeps_successful_trace_cache(self):
        workflow,*_=self._partial_physics_fixture(cached_max_step_s=.0005,physics_max_step_seconds=.0005)
        result,native,_auditor=self._exercise_partial_fixture(workflow)
        self.assertEqual("validation_only",result["status"]);native.assert_not_called()

    def test_author_smaller_step_keeps_cache_when_effective_initial_step_is_same(self):
        scene=json.loads(self.scene.read_text());scene["tests"][0]["step_s"]=.0002
        write_json_atomic(self.scene,scene)
        workflow,*_=self._partial_physics_fixture(physics_max_step_seconds=.0005)
        result,native,_auditor=self._exercise_partial_fixture(workflow)
        self.assertEqual("validation_only",result["status"]);native.assert_not_called()

    def test_max_step_invalid_values_are_rejected_before_state_writes(self):
        before=(self.project/"workflow.json").read_bytes()
        for value in (0,-.001,.0011,float("nan"),float("inf"),-float("inf")):
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,"positive, finite and at most"):
                self.workflow(physics_max_step_seconds=value)
        self.assertEqual(before,(self.project/"workflow.json").read_bytes())
        self.assertFalse((self.project/"criteria.lock.json").exists())

    def test_finer_step_does_not_recover_an_old_coarse_failure(self):
        workflow,old,_geometry,_request,trace=self._partial_physics_fixture(physics_max_step_seconds=.0005)
        trace.update(physical_pass=False,failure={"kind":"contact_penetration"},numerical_failure_unresolved=True)
        trace["step_s"]=.00025  # Last retry is not the initial policy.
        trace["numerical_attempts"].extend([{"max_step_s":.0005},{"max_step_s":.00025}])
        write_json_atomic(old/"tests/move/physical.trace.json",trace)
        result,native,_auditor=self._exercise_partial_fixture(workflow)
        self.assertEqual("HARNESS_ERROR",result["status"])
        native.assert_called_once()
        self.assertIn("isolated execution sentinel",result["issues"][0]["failure"]["issues"])

    def test_eligible_pass_early_return_obeys_requested_initial_step(self):
        previous={"eligible_pass":True,"status":"PASS","tests":[self._implemented_test_summary()]}
        for step,expected_calls in ((.001,0),(.0005,1)):
            workflow=self.workflow(physics_max_step_seconds=step)
            workflow.run_revision=Mock(return_value={"status":"HARNESS_ERROR","issues":["new precision run"]})
            with patch("simulation_ready_cad_workflow.validation_pipeline.slim_harness.publication.read_result",return_value=previous), \
                 patch("simulation_ready_cad_workflow.validation_pipeline.slim_harness.workflow.RuntimeHeartbeat"):
                result=workflow.run(one_revision=True)
            self.assertEqual(expected_calls,workflow.run_revision.call_count)
            self.assertEqual("PASS" if step==.001 else "HARNESS_ERROR",result["status"])

    def _loop_after(self,status,*,one_revision=False,**kwargs):
        workflow=self.workflow(**kwargs)
        workflow.run_revision=Mock(return_value={"status":status,"issues":["fixture problem"]})
        workflow._budget_revision=Mock(return_value={"status":"author_turn"})
        with patch("simulation_ready_cad_workflow.validation_pipeline.slim_harness.publication.read_result",return_value={}), \
             patch("simulation_ready_cad_workflow.validation_pipeline.slim_harness.workflow.RuntimeHeartbeat"):
            return workflow,workflow.run(one_revision=one_revision)

    def test_budgeted_loop_returns_harness_and_binding_failures_to_the_author(self):
        for status in ("HARNESS_ERROR","EVIDENCE_REQUIRED"):
            with self.subTest(status=status):
                self.setUp()
                workflow,result=self._loop_after(status,tier="light")
                self.assertEqual(result["status"],"author_turn")
                feedback=workflow._budget_revision.call_args.args[1]
                self.assertEqual((feedback["status"],feedback["reported_status"]),("NEEDS_REVISION",status))
                self.assertTrue((self.project/"runs/r02/design").is_dir())

    def test_unbudgeted_or_single_revision_runs_still_stop_for_harness_repair(self):
        for kwargs in ({}, {"tier":"light","one_revision":True}):
            with self.subTest(**kwargs):
                self.setUp()
                workflow,result=self._loop_after("HARNESS_ERROR",**kwargs)
                self.assertEqual(result["status"],"HARNESS_ERROR")
                workflow._budget_revision.assert_not_called()

    def test_finer_step_does_not_reuse_coarse_validated_bundle(self):
        workflow,old,geometry,_request,_trace=self._partial_physics_fixture(physics_max_step_seconds=.0005)
        write_json_atomic(old/"validated-physics.json",{
            "criteria_sha256":workflow.criteria_sha256,"geometry":geometry,
            "source_bindings":{str(self.scene):digest(self.scene),str(self.source):digest(self.source)},
            "evidence_bindings":{},"images":[],"motions":[],"tests":[self._implemented_test_summary()],"preflight":{"status":"ready"}})
        result,native,_auditor=self._exercise_partial_fixture(workflow)
        self.assertEqual("HARNESS_ERROR",result["status"]);native.assert_called_once()
        workflow._finish_validation.assert_not_called()

    def test_disk_trace_resume_binds_every_side_file_and_rejects_changed_signal(self):
        workflow,old,_geometry,_request,trace=self._partial_physics_fixture()
        original=old/"tests/move/physical.trace.json"
        with TraceWriter(original,len(trace["samples"]),chunk_rows=1) as writer:
            for index,sample in enumerate(trace["samples"]):
                writer.append(sample,{name:values[index] for name,values in trace["signals"].items()})
            writer.commit({key:value for key,value in trace.items() if key not in {"samples","signals"}})
        result,native,auditor=self._exercise_partial_fixture(workflow)
        self.assertEqual("validation_only",result["status"])
        native.assert_not_called()
        self.assertIsInstance(auditor.call_args.args[1],SampleRows)
        self.assertEqual(trace["samples"],list(auditor.call_args.args[1]))
        copied=self.run/"attempts/a002/tests/move/physical.trace.json"
        bundle=json.loads((self.run/"attempts/a002/validated-physics.json").read_text())
        self.assertLessEqual({str(path) for path in trace_artifacts(copied)},set(bundle["evidence_bindings"]))
        self.assertEqual(set(trace_artifacts(original)[1:]),set(trace_artifacts(copied)[1:]))
        # Exercise validated-bundle adoption, not known-failure recovery. If a
        # side file is omitted from bindings, the early review shortcut wins.
        write_json_atomic(self.run/"result.json",{"status":"EVIDENCE_REQUIRED"})
        metadata=json.loads(copied.read_text())
        signal_path=(copied.parent/metadata["storage"]["files"]["signals"]["path"]).resolve()
        with signal_path.open("r+b") as stream:stream.write(b"changed!")
        workflow._finish_validation.reset_mock()
        rejected,_,_=self._exercise_partial_fixture(workflow)
        self.assertEqual("HARNESS_ERROR",rejected["status"])
        self.assertIn("binding mismatch",str(rejected["issues"]))
        workflow._finish_validation.assert_not_called()

    def test_partial_physics_rejects_changed_compiled_scene(self):
        workflow,old,_geometry,request,_trace=self._partial_physics_fixture()
        request["scene"]["gravity_m_s2"]=[0,-3,0]
        write_json_atomic(old/"tests/move/request.json",request)
        result,native,_auditor=self._exercise_partial_fixture(workflow)
        self.assertEqual("HARNESS_ERROR",result["status"])
        native.assert_called_once()  # The sentinel replaces, and prevents, actual native execution.

    def test_partial_physics_rejects_changed_test(self):
        workflow,old,_geometry,request,_trace=self._partial_physics_fixture()
        request["test"]["duration_s"]=2
        write_json_atomic(old/"tests/move/request.json",request)
        _result,native,_auditor=self._exercise_partial_fixture(workflow)
        native.assert_called_once()

    def test_partial_physics_rejects_changed_native_implementation(self):
        workflow,old,_geometry,_request,trace=self._partial_physics_fixture()
        self.assertEqual({"runtime.py","runtime_progress.py","spur_gears.py","measurements.py","contracts.py","trace_storage.py","fixed_component_filter.py",
            "source_contact_filter.py","source_contact_normals.py","source_capsule_contacts.py",
            "source_queries.py","source_support.py","numerical.py",
            "measurement_registry.py","alarm_measurements.py","clock_contacts/contact_collector.py",
            "clock_contacts/calibration_binding.py","clock_contacts/runtime_identity.py",
            "equivalent_elasticity.py","equivalent_elastic_compiler.py","equivalent_collision_scope.py","validation_scope.py","shaft_supports.py"},set(trace["execution_implementation"]))
        trace["execution_implementation"]["measurements.py"]="previous measurement implementation"
        write_json_atomic(old/"tests/move/physical.trace.json",trace)
        _result,native,_auditor=self._exercise_partial_fixture(workflow)
        native.assert_called_once()

    def test_partial_physics_does_not_guess_missing_native_implementation(self):
        workflow,old,_geometry,_request,trace=self._partial_physics_fixture()
        trace.pop("execution_implementation")
        write_json_atomic(old/"tests/move/physical.trace.json",trace)
        _result,native,_auditor=self._exercise_partial_fixture(workflow)
        native.assert_called_once()

    def test_geometry_cache_rejects_changed_or_missing_implementation(self):
        workflow,old,geometry,_request,_trace=self._partial_physics_fixture()
        for marker in (None,"previous geometry implementation"):
            with self.subTest(marker=marker):
                value=copy.deepcopy(geometry)
                if marker is None:value.pop("geometry_implementation_sha256")
                else:value["geometry_implementation_sha256"]=marker
                write_json_atomic(old/"geometry/geometry.json",value)
                with patch("simulation_ready_cad_workflow.validation_pipeline.slim_harness.workflow.prepare_geometry",return_value={"errors":["fresh geometry required"]}) as prepare:
                    self.assertEqual("NEEDS_REVISION",workflow.run_revision()["status"])
                    prepare.assert_called_once()

    def test_project_creation_preserves_existing_and_delivery_copies_only_references(self):
        with self.assertRaisesRegex(ValueError,"already exists"):
            create_project(self.workspace,"machine","different task")
        original=self.workspace/"delivery";original.mkdir()
        shared=self.workspace/"shared";shared.mkdir()
        (shared/"source.obj").write_text("referenced source")
        (original/"unrelated.txt").write_text("must not be copied")
        scene=json.loads(self.scene.read_text());scene["bodies"][0]["source"]="../shared/source.obj"
        write_json_atomic(original/"scene.json",scene)
        target=self.workspace/"copied";target.mkdir()
        copy_delivery(original/"scene.json",target)
        self.assertEqual((target/"source.obj").read_text(),"referenced source")
        self.assertFalse((target/"unrelated.txt").exists())
        self.assertEqual(json.loads((target/"scene.json").read_text())["bodies"][0]["source"],"source.obj")

    def test_missing_or_partial_scene_is_delivery_feedback(self):
        with self.assertRaises(DeliveryError): load_scene(self.workspace/"absent.json")
        self.scene.write_text('{"schema_version":',encoding="utf-8")
        with self.assertRaises(DeliveryError): load_scene(self.scene)

    def test_snapshot_fallback_name_does_not_overwrite_another_source(self):
        original=self.workspace/"collision-delivery";original.mkdir()
        paths=[original/"source_003.obj",original/"left"/"part.obj",original/"right"/"part.obj"]
        for index,path in enumerate(paths):path.parent.mkdir(exist_ok=True);path.write_text(f"source-{index}")
        scene=json.loads(self.scene.read_text());scene["bodies"]=[{"id":f"body{i}","source":str(path.relative_to(original)),"fixed":False} for i,path in enumerate(paths)]
        write_json_atomic(original/"scene.json",scene)
        target=self.workspace/"collision-copied";target.mkdir()
        copy_delivery(original/"scene.json",target)
        copied=json.loads((target/"scene.json").read_text())
        self.assertEqual([(target/body["source"]).read_text() for body in copied["bodies"]],["source-0","source-1","source-2"])


    def test_active_runner_prevents_a_second_execution(self):
        import os,psutil
        workflow=self.workflow()
        write_json_atomic(self.project/"workflow_runtime.json",{"schema_version":"workflow_runtime_v1","status":"running",
                          "pid":os.getpid(),"process_started_at_epoch":psutil.Process().create_time()})
        workflow.run_revision=Mock(side_effect=AssertionError("duplicate execution"))
        with self.assertRaises(ActiveWorkflowError):workflow.run()
        workflow.run_revision.assert_not_called()
        with self.assertRaises(ActiveWorkflowError):self.workflow()

    def test_original_criteria_lock_survives_resume(self):
        self.workflow()
        criteria=json.loads((self.project/"criteria.json").read_text())
        criteria["criteria"][0]["statement"]="Changed interpretation"
        write_json_atomic(self.project/"criteria.json",criteria)
        with self.assertRaisesRegex(ValueError,"Frozen criteria changed"):
            self.workflow()


    def test_duplicate_criterion_does_not_overwrite_a_violation(self):
        criteria=json.loads((self.project/"criteria.json").read_text())
        result={"status":"ready","observations":[
            {"criterion_id":"appearance","verdict":"violated","observed_fact":"missing","evidence_paths":["view.png"]},
            {"criterion_id":"appearance","verdict":"met","observed_fact":"present","evidence_paths":["view.png"]},
            {"criterion_id":"function","verdict":"met","observed_fact":"moved","evidence_paths":["trace.json"]}],
            "test_coverage":[{"criterion_id":"function","appropriate":True,"test_ids":["move"]}]}
        self.assertFalse(review_passes(result,criteria,json.loads(self.scene.read_text()),final=True))

    def test_validated_bundle_resumes_code_finalization_without_physics(self):
        workflow=self.workflow()
        old=self.run/"attempts"/"a001";old.mkdir(parents=True)
        record=old/"trace.json";record.write_text("previously generated physical evidence")
        geometry={"scene_file":str(self.scene),"model_file":str(old/"model.glb"),"bodies":[],
                  "geometry_implementation_sha256":GEOMETRY_IMPLEMENTATION_SHA256}
        write_json_atomic(old/"validated-physics.json",{
            "criteria_sha256":workflow.criteria_sha256,"geometry":geometry,
            "source_bindings":{str(self.scene):digest(self.scene),str(self.source):digest(self.source)},
            "evidence_bindings":{str(record):digest(record)},"images":[],"motions":[],"tests":[self._implemented_test_summary()],"preflight":{"status":"ready"}})
        workflow._preview=Mock()
        workflow._finish_validation=Mock(return_value={"status":"validation_only"})
        with patch("simulation_ready_cad_workflow.validation_pipeline.slim_harness.workflow.prepare_geometry",side_effect=AssertionError("geometry rerun")), \
             patch("simulation_ready_cad_workflow.validation_pipeline.slim_harness.workflow.run_bounded_subprocess",side_effect=AssertionError("physics rerun")), \
             patch("simulation_ready_cad_workflow.validation_pipeline.slim_harness.workflow.audit_motion",side_effect=AssertionError("audit rerun")):
            self.assertEqual(workflow.run_revision()["status"],"validation_only")
        self.assertEqual(workflow._finish_validation.call_args.kwargs["resumed_from"],str(old/"validated-physics.json"))

    def test_mutated_physical_evidence_invalidates_camera_resume(self):
        workflow=self.workflow()
        old=self.run/"attempts"/"a001";old.mkdir(parents=True)
        record=old/"trace.json";record.write_text("original")
        write_json_atomic(old/"validated-physics.json",{"criteria_sha256":workflow.criteria_sha256,
            "source_bindings":{str(self.scene):digest(self.scene)},"evidence_bindings":{str(record):digest(record)},
            "geometry":{"geometry_implementation_sha256":GEOMETRY_IMPLEMENTATION_SHA256},"tests":[self._implemented_test_summary()]})
        record.write_text("changed")
        with patch("simulation_ready_cad_workflow.validation_pipeline.slim_harness.workflow.prepare_geometry",return_value={"errors":["needs preparation"]}) as prepare:
            self.assertEqual(workflow.run_revision()["status"],"NEEDS_REVISION")
            prepare.assert_called_once()

    def test_validated_bundle_cannot_bypass_native_or_audit_implementation_change(self):
        workflow=self.workflow()
        old=self.run/"attempts/a001";old.mkdir(parents=True)
        record=old/"trace.json";record.write_text("bound native evidence")
        bundle={"criteria_sha256":workflow.criteria_sha256,
                "source_bindings":{str(self.scene):digest(self.scene),str(self.source):digest(self.source)},
                "evidence_bindings":{str(record):digest(record)},
                "geometry":{"geometry_implementation_sha256":GEOMETRY_IMPLEMENTATION_SHA256},
                "tests":[self._implemented_test_summary()]}
        for changed in ("native","audit","geometry"):
            with self.subTest(changed=changed):
                value=copy.deepcopy(bundle)
                if changed=="native":value["tests"][0]["execution_implementation"]["runtime.py"]="old runtime"
                elif changed=="audit":value["tests"][0]["geometry_audit"]["audit_implementation_sha256"]="old audit"
                else:value["geometry"]["geometry_implementation_sha256"]="old geometry"
                write_json_atomic(old/"validated-physics.json",value)
                with patch("simulation_ready_cad_workflow.validation_pipeline.slim_harness.workflow.prepare_geometry",return_value={"errors":["current pipeline required"]}) as prepare:
                    self.assertEqual("NEEDS_REVISION",workflow.run_revision()["status"])
                    prepare.assert_called_once()

    def test_recorded_functional_failure_does_not_wait_for_acceptance_audit(self):
        workflow,old,_geometry,_request,trace=self._partial_physics_fixture()
        trace["physical_pass"]=False
        trace["assertions"][0]["passed"]=False
        write_json_atomic(old/"tests/move/physical.trace.json",trace)
        result,native,auditor=self._exercise_partial_fixture(workflow)
        self.assertEqual("NEEDS_REVISION",result["status"])
        self.assertEqual("not_completed_for_rejected_test",result["issues"][0]["geometry_audit"]["status"])
        native.assert_not_called()
        self.assertEqual(1,auditor.call_count)  # Common initial geometry only; no acceptance-only motion audit.
        workflow._motion.assert_called_once()


    def test_selected_equivalent_mode_requires_actual_author_declaration_before_physics(self):
        write_json_atomic(self.project/'modeling_policy.json',{'selected_mode':'equivalent_elastic','explicit_scope':'Synthetic scope integration check, not clock acceptance.'})
        workflow,old,*_=self._partial_physics_fixture()
        old_review=json.loads((old/'preflight_review.json').read_text())
        workflow._review=Mock(return_value=old_review)
        result,native,auditor=self._exercise_partial_fixture(workflow)
        self.assertEqual('NEEDS_REVISION',result['status'])
        self.assertTrue(any(isinstance(x,str) and 'requires an author-supplied equivalent_elasticity declaration' in x for x in result['issues']))
        native.assert_not_called()
        self.assertEqual(1,auditor.call_count)
        workflow._review.assert_not_called()  # Compile feedback returns directly to the author.
        workflow._finish_validation.assert_not_called()

    def test_policy_change_cannot_reuse_prior_ready_review(self):
        write_json_atomic(self.project/'modeling_policy.json',{'selected_mode':'equivalent_elastic','explicit_scope':'New limited model.'})
        workflow,old,*_=self._partial_physics_fixture()
        review=json.loads((old/'preflight_review.json').read_text())
        review['modeling_policy_sha256']='old_policy_digest'
        write_json_atomic(old/'preflight_review.json',review)
        workflow._review=Mock(return_value={'status':'needs_revision','observations':[],'test_coverage':[]})
        result,native,_=self._exercise_partial_fixture(workflow)
        self.assertEqual('NEEDS_REVISION',result['status']);native.assert_not_called();workflow._review.assert_not_called()

    def test_changed_policy_must_not_deliver_old_model_failure_to_author(self):
        old_workflow, old, _, _, trace = self._partial_physics_fixture()
        trace["physical_pass"] = False
        trace["assertions"][0]["passed"] = False
        write_json_atomic(old / "tests/move/physical.trace.json", trace)
        # A supervisor interruption leaves a completed failed trace but no
        # result.json. The user's modeling choice changes before resume.
        self.assertFalse((self.run / "result.json").exists())
        write_json_atomic(self.project / "modeling_policy.json", {
            "selected_mode": "equivalent_elastic", "explicit_scope": "New user-selected equivalent scope."})
        current = self.workflow()
        current._preview = old_workflow._preview
        current._motion = old_workflow._motion
        current._finish_validation = old_workflow._finish_validation
        current._review = Mock(return_value={"status": "needs_revision", "observations": [], "test_coverage": []})
        result, native, auditor = self._exercise_partial_fixture(current)
        self.assertEqual("NEEDS_REVISION", result["status"])
        self.assertTrue(any(isinstance(issue, str) and "author-supplied equivalent_elasticity declaration" in issue
                            for issue in result["issues"]),
                        "Resume bypassed the current policy/declaration gate and returned the previous model's failure")
        self.assertEqual(1, auditor.call_count)
        current._review.assert_not_called()
        native.assert_not_called()


    def _compact_policy(self):
        policy={"selected_mode":"equivalent_elastic", "max_test_duration_s":30,
                "max_total_test_duration_s":90, "acceptance":"All six functions, real bell impact and continuous running remain required.",
                "test_duration_basis":"User authorized reasonable durations considering simulation speed; current planning limits are revisable with physical evidence."}
        write_json_atomic(self.project/"modeling_policy.json",policy)
        return policy

    def _set_test_durations(self,durations):
        scene=json.loads(self.scene.read_text(encoding="utf-8"))
        original=scene["tests"][0]
        scene["tests"]=[{**copy.deepcopy(original),"id":f"move_{i}","duration_s":duration}
                        for i,duration in enumerate(durations)]
        write_json_atomic(self.scene,scene)

    def test_compact_policy_rejects_old_450s_before_measurement_registration(self):
        policy=self._compact_policy();self._set_test_durations([450])
        workflow=self.workflow()
        with patch("simulation_ready_cad_workflow.validation_pipeline.slim_harness.workflow.load_registered_measurements") as registration:
            with self.assertRaises(DeliveryError) as raised:workflow.run_revision()
        registration.assert_not_called()
        self.assertIn('"overlong_tests": [{"id": "move_0", "duration_s": 450}]',str(raised.exception))
        self.assertIn(policy["acceptance"],str(raised.exception))
        self.assertIn("time-limited task-change advice and an author decision",str(raised.exception))
        self.assertIn("registered measurements still require actual re-registration",str(raised.exception))
        self.assertEqual(450,json.loads(self.scene.read_text())["tests"][0]["duration_s"])

    def test_compact_policy_allows_20_25_30s_to_reach_measurement_registration(self):
        self._compact_policy();self._set_test_durations([20,25,30])
        workflow=self.workflow()
        with patch("simulation_ready_cad_workflow.validation_pipeline.slim_harness.workflow.load_registered_measurements",
                   side_effect=RuntimeError("measurement boundary reached")) as registration:
            with self.assertRaisesRegex(RuntimeError,"measurement boundary reached"):workflow.run_revision()
        registration.assert_called_once()

    def test_compact_policy_rejects_total_over_90s_even_when_each_test_fits(self):
        self._compact_policy();self._set_test_durations([30,30,30,1])
        workflow=self.workflow()
        with patch("simulation_ready_cad_workflow.validation_pipeline.slim_harness.workflow.load_registered_measurements") as registration:
            with self.assertRaises(DeliveryError) as raised:workflow.run_revision()
        registration.assert_not_called()
        self.assertIn('"actual_total_test_duration_s": 91',str(raised.exception))
        self.assertIn('"overlong_tests": []',str(raised.exception))

    def test_absent_duration_policy_keeps_450s_measurement_path(self):
        self._set_test_durations([450]);workflow=self.workflow()
        with patch("simulation_ready_cad_workflow.validation_pipeline.slim_harness.workflow.load_registered_measurements",
                   side_effect=RuntimeError("measurement boundary reached")) as registration:
            with self.assertRaisesRegex(RuntimeError,"measurement boundary reached"):workflow.run_revision()
        registration.assert_called_once()



    def _bind_reviewed_source_feedback(self):
        policy=self._compact_policy()
        plan=self.scene.parent/"plan.json";write_json_atomic(plan,{"operation":"reviewed source plan"})
        feedback={"verdict":"REVISE", "source_bindings":{path.name:digest(path) for path in (self.scene,self.source,plan)},
                  "findings":[{"id":"source_review_fixture","reason":"Correct the reviewed physical trigger expression."}]}
        policy["source_review_feedback"]=feedback
        write_json_atomic(self.project/"modeling_policy.json",policy)
        return feedback

    def test_matching_source_review_rejection_reaches_author_revision_before_registration(self):
        feedback=self._bind_reviewed_source_feedback();workflow=self.workflow()
        with patch("simulation_ready_cad_workflow.validation_pipeline.slim_harness.workflow.load_registered_measurements",
                   side_effect=AssertionError("Old registration must not hide source review feedback")) as registration:
            result=workflow.run(one_revision=True)
        self.assertEqual("NEEDS_REVISION",result["status"])
        registration.assert_not_called()
        self.assertIn(json.dumps(feedback,ensure_ascii=False),result["issues"][0])
        self.assertIn("Independent source review",result["issues"][0])
        self.assertEqual(1,workflow.state["revision"])

    def test_any_changed_reviewed_source_does_not_reuse_old_rejection(self):
        feedback=self._bind_reviewed_source_feedback();workflow=self.workflow()
        for relative in feedback["source_bindings"]:
            with self.subTest(changed_source=relative):
                path=self.scene.parent/relative;original=path.read_bytes()
                path.write_bytes(original+b"\n")
                try:
                    with patch("simulation_ready_cad_workflow.validation_pipeline.slim_harness.workflow.load_registered_measurements",
                               side_effect=RuntimeError("new source reaches current registration")) as registration:
                        with self.assertRaisesRegex(RuntimeError,"new source reaches current registration"):
                            workflow.run_revision()
                    registration.assert_called_once()
                finally:
                    path.write_bytes(original)

if __name__=="__main__": unittest.main()
