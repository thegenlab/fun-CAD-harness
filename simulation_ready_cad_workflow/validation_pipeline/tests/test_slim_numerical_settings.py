"""Harness step limits and cache schedules, using stub traces only."""
from copy import deepcopy
from contextlib import redirect_stdout
import io
import json
import math
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch

from ..slim_harness.numerical import run_fixed_attempts, numerical_policy_matches, NUMERICAL_POLICY


class NumericalSettingsTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)

    def run_attempts(self,test,max_step_s=.001,solver_iterations=160):
        calls=[]
        def once(path,step):
            calls.append(step)
            value={"samples":[],"signals":{},"physical_pass":False,
                   "failure":{"kind":"contact_penetration"},"step_s":step}
            path.write_text(json.dumps(value))
            return value
        result=run_fixed_attempts(once,test,self.root/"trace.json",max_step_s=max_step_s,solver_iterations=solver_iterations)
        return calls,result

    def test_default_schedule_remains_one_ms_and_two_halvings(self):
        calls,result=self.run_attempts({"duration_s":.01})
        self.assertEqual([.001,.0005,.00025],calls)
        self.assertEqual({"requested_max_step_s":.001,"effective_initial_step_s":.001,"solver_iterations":160},result["numerical_settings"])

    def test_finer_start_keeps_inputs_and_exactly_two_halvings(self):
        test={"duration_s":2.15,"step_s":.001,"loads":[{"force_n":[-5,0,0]}],"assertions":[{"expression":"unchanged"}]}
        before=deepcopy(test)
        calls,result=self.run_attempts(test,.0005)
        self.assertEqual([.0005,.00025,.000125],calls)
        self.assertEqual(before,test)
        self.assertTrue(result["numerical_failure_unresolved"])
        self.assertFalse(result["physical_pass"])
        self.assertTrue(all(Path(row["trace_file"]).is_file() for row in result["numerical_attempts"]))

    def test_smaller_author_bound_is_not_increased(self):
        calls,result=self.run_attempts({"duration_s":.01,"step_s":.0002},.0005)
        self.assertEqual([.0002,.0001,.00005],calls)
        self.assertEqual(.0002,result["numerical_settings"]["effective_initial_step_s"])
        self.assertTrue(numerical_policy_matches(result,{"step_s":.0002},.001))

    def test_invalid_caller_bound_creates_no_attempt(self):
        for value in (0,-1,.0011,float("nan"),float("inf"),-float("inf")):
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,"positive, finite and at most"):
                run_fixed_attempts(lambda *_:self.fail("must not execute"),{"duration_s":.01},self.root/"never.json",max_step_s=value)
        self.assertEqual([],list(self.root.iterdir()))

    def test_matching_only_the_last_retry_is_not_sufficient(self):
        _,result=self.run_attempts({"duration_s":.01})
        self.assertEqual(.00025,result["step_s"])
        self.assertFalse(numerical_policy_matches(result,{},.0005))
        self.assertTrue(numerical_policy_matches(result,{},.001))

    def test_missing_or_inconsistent_policy_metadata_is_not_guessed(self):
        self.assertFalse(numerical_policy_matches({"step_s":.0005},{},.0005))
        trace={"numerical_policy":NUMERICAL_POLICY,
               "numerical_settings":{"requested_max_step_s":.001,"effective_initial_step_s":.0005},
               "numerical_attempts":[{"max_step_s":.0005}]}
        self.assertFalse(numerical_policy_matches(trace,{},.0005))

    def test_runtime_main_forwards_outer_numerical_settings(self):
        from ..slim_harness import runtime
        path=self.root/"request.json"
        request={"scene":{"bodies":[]},"test":{"duration_s":.01},"output":str(self.root/"trace.json"),
                 "numerical_settings":{"max_step_s":.0005}}
        path.write_text(json.dumps(request))
        with patch.object(runtime,"simulate",return_value={}) as simulate,patch.object(runtime.sys,"argv",["runtime",str(path)]),redirect_stdout(io.StringIO()):
            runtime.main()
        simulate.assert_called_once_with(request["scene"],request["test"],Path(request["output"]),max_step_s=.0005,solver_iterations=160)

    def test_runtime_simulate_forwards_bound_without_loading_native(self):
        from ..slim_harness import runtime
        with patch.object(runtime,"run_fixed_attempts",return_value={"sentinel":True}) as attempts, \
             patch.object(runtime,"load_pychrono_core",side_effect=AssertionError("No native allowed")):
            result=runtime.simulate({}, {"duration_s":.01}, self.root/"trace.json",max_step_s=.0005)
        self.assertEqual({"sentinel":True},result)
        self.assertEqual(.0005,attempts.call_args.kwargs["max_step_s"])

    def test_runtime_records_full_duration_using_effective_finer_bound(self):
        from ..slim_harness import runtime
        from ..slim_harness.trace_storage import open_trace
        test={"duration_s":.00125,"step_s":.001}
        calls=[]
        def solved(_prepared,actual_test,_path,writer,*,max_step_s,solver_iterations,progress=None):
            calls.append(max_step_s)
            count=math.ceil(actual_test["duration_s"]/max_step_s)
            dt=actual_test["duration_s"]/count
            for index in range(count+1):
                writer.append({"time_s":index*dt,"body_poses":{}},{"time_s":index*dt})
            return writer.commit({"physical_pass":True,"failure":None,"step_s":dt,"elapsed_seconds":0.})
        output=self.root/"recorded.json"
        with patch.object(runtime,"_simulate",side_effect=solved), \
             patch.object(runtime,"load_pychrono_core",side_effect=AssertionError("No native allowed")):
            result=runtime.simulate({},test,output,max_step_s=.0005)
        self.assertEqual([.0005],calls)
        trace=open_trace(output)
        self.assertEqual(4,len(trace["samples"]))
        self.assertEqual(.00125,trace["samples"][-1]["time_s"])
        self.assertLessEqual(result["step_s"],.0005)
        self.assertEqual({"duration_s":.00125,"step_s":.001},test)

    def test_workflow_cli_passes_step_and_existing_timeout(self):
        from ..slim_harness import workflow
        with patch.object(workflow,"SlimWorkflow") as constructor, \
             patch.object(sys,"argv",
                          ['workflow','project','--physics-max-step-seconds','.0005','--physics-timeout-seconds','1800','--one-revision']), \
             redirect_stdout(io.StringIO()):
            constructor.return_value.run.return_value={}
            workflow.main()
        self.assertEqual(.0005,constructor.call_args.kwargs["physics_max_step_seconds"])
        self.assertEqual(1800,constructor.call_args.kwargs["physics_timeout_seconds"])
        constructor.return_value.run.assert_called_once_with(one_revision=True)


    def test_iterations_validation_prevents_execution(self):
        for value in (0,-1,1.5,True,"320",None):
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,"positive integer"):
                run_fixed_attempts(lambda *_:self.fail("must not execute"),{},self.root/"never.json",solver_iterations=value)
        self.assertEqual([],list(self.root.iterdir()))

    def test_custom_iterations_are_retained_across_step_retries_and_cache_match(self):
        calls,result=self.run_attempts({"duration_s":.01},solver_iterations=320)
        self.assertEqual([.001,.0005,.00025],calls)
        self.assertEqual(320,result["numerical_settings"]["solver_iterations"])
        self.assertTrue(numerical_policy_matches(result,{},solver_iterations=320))
        self.assertFalse(numerical_policy_matches(result,{},solver_iterations=160))
        self.assertFalse(numerical_policy_matches(result,{},solver_iterations=640))

    def test_runtime_main_forwards_explicit_iterations(self):
        from ..slim_harness import runtime
        path=self.root/"iterations-request.json"
        request={"scene":{},"test":{},"output":str(self.root/"out.json"),"numerical_settings":{"solver_iterations":320}}
        path.write_text(json.dumps(request))
        with patch.object(runtime,"simulate",return_value={}) as simulate,patch.object(runtime.sys,"argv",["runtime",str(path)]),redirect_stdout(io.StringIO()):runtime.main()
        self.assertEqual(320,simulate.call_args.kwargs["solver_iterations"])

    def test_workflow_cache_and_cli_keep_iterations(self):
        from ..slim_harness import workflow
        obj=workflow.SlimWorkflow.__new__(workflow.SlimWorkflow);obj.physics_max_step_seconds=.001;obj.physics_solver_iterations=320
        _,trace=self.run_attempts({"duration_s":.01},solver_iterations=320);trace["test_id"]="one"
        self.assertTrue(obj._numerical_cache_matches([trace],[{"id":"one"}]))
        obj.physics_solver_iterations=160
        self.assertFalse(obj._numerical_cache_matches([trace],[{"id":"one"}]))
        with patch.object(workflow,"SlimWorkflow") as constructor,patch.object(sys,"argv",['workflow','project','--physics-solver-iterations','320']),redirect_stdout(io.StringIO()):
            constructor.return_value.run.return_value={};workflow.main()
        self.assertEqual(320,constructor.call_args.kwargs["physics_solver_iterations"])

    def test_trial_cache_does_not_reuse_another_iteration_budget(self):
        from ..slim_harness import trial_cache
        from ..slim_harness.implementation import native_implementation_binding
        folder=self.root/".runtime_runs/author_tools/simulate-candidate/tests/one";folder.mkdir(parents=True)
        source=self.root/"original.obj";source.write_text("original geometry")
        test={"id":"one","duration_s":.01};prepared={"bodies":[{"id":"part","mesh_file":str(source)}],"tests":[test]}
        path=folder/"physical.trace.json"
        trace={"samples":[],"signals":{},"test_id":"one","physical_pass":True,"failure":None,"numerical_failure_unresolved":False,
               "execution_implementation":native_implementation_binding(),"numerical_policy":NUMERICAL_POLICY,
               "numerical_settings":{"requested_max_step_s":.001,"effective_initial_step_s":.001,"solver_iterations":320},
               "numerical_attempts":[{"max_step_s":.001}]}
        path.write_text(json.dumps(trace));request={"scene":prepared,"test":test,"output":str(path),"numerical_settings":{"max_step_s":.001,"solver_iterations":320}}
        (folder/"request.json").write_text(json.dumps(request))
        with patch.object(trial_cache,"validate_trace_measurements"):
            receipt=trial_cache.store_trial(prepared,test,path,sys.executable);self.assertIsNotNone(receipt)
            self.assertIsNone(trial_cache.reuse_trial(self.root,prepared,test,self.root/"wrong.json",sys.executable,solver_iterations=160))
            self.assertIsNotNone(trial_cache.reuse_trial(self.root,prepared,test,self.root/"right.json",sys.executable,solver_iterations=320))


if __name__=='__main__':unittest.main()
