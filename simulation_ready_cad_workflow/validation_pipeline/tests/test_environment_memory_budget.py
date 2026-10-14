"""Host/container budgets and small owned-process tests; no memory stress load."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from ..runtime import memory_budget as memory
from ..runtime.bounded_subprocess import run_bounded_subprocess,BoundedSubprocessError
from ..slim_harness.physics_invocation import _rss_limit

GIB=1024**3
PREFIX="simulation_ready_cad_workflow.validation_pipeline.runtime.memory_budget."

class EnvironmentMemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)

    def pair(self,name,limit,used):
        folder=self.root/name;folder.mkdir(parents=True,exist_ok=True)
        a=folder/'memory.max';b=folder/'memory.current'
        a.write_text(str(limit));b.write_text(str(used));return a,b

    def test_large_host_gets_more_than_old_three_gib(self):
        snapshot=memory.available_memory(memory=SimpleNamespace(total=64*GIB,available=48*GIB),cgroup_files=[])
        with patch(PREFIX+'available_memory',return_value=snapshot):guard=memory.AutoMemoryLimit()
        self.assertAlmostEqual(38.4,guard.initial/GIB,places=6)
        self.assertEqual(48*GIB,guard.initial+guard.reserve)
        self.assertEqual('auto',_rss_limit(None,None))

    def test_host_available_and_container_ancestors_both_limit_capacity(self):
        pairs=[self.pair('parent',8*GIB,7*GIB),self.pair('parent/child',4*GIB,GIB)]
        result=memory.available_memory(memory=SimpleNamespace(total=128*GIB,available=100*GIB),cgroup_files=pairs)
        self.assertEqual(4*GIB,result['effective_total_bytes']);self.assertEqual(GIB,result['effective_available_bytes'])
        result=memory.available_memory(memory=SimpleNamespace(total=128*GIB,available=GIB//2),cgroup_files=pairs)
        self.assertEqual(GIB//2,result['effective_available_bytes'])

    def test_unlimited_v1_and_v2_do_not_replace_host_availability(self):
        pairs=[self.pair('v1',2**63-4096,123),self.pair('v2','max',123)]
        result=memory.available_memory(memory=SimpleNamespace(total=8*GIB,available=3*GIB),cgroup_files=pairs)
        self.assertEqual(3*GIB,result['effective_available_bytes'])
        self.assertEqual([],result['cgroup_constraints'])

    def test_cgroup_mount_root_and_namespace_paths(self):
        membership=self.root/'cgroup';mountinfo=self.root/'mountinfo';mount=self.root/'controller'
        for kind,controllers,member,root,tail in [('cgroup2','','/tenant/child','/tenant','rw'),('cgroup','memory','/child','/tenant','rw,memory')]:
            with self.subTest(kind=kind):
                membership.write_text(f'0:{controllers}:{member}\n')
                mountinfo.write_text(f'1 0 0:1 {root} {mount.as_posix()} rw - {kind} cgroup {tail}\n')
                pairs=memory.cgroup_memory_files(membership,mountinfo)
                names=('memory.max','memory.current') if kind=='cgroup2' else ('memory.limit_in_bytes','memory.usage_in_bytes')
                self.assertEqual([(mount/'child'/names[0],mount/'child'/names[1]),(mount/names[0],mount/names[1])],pairs)

    def test_own_allocations_are_not_subtracted_twice_but_competition_tightens(self):
        samples=[{'effective_available_bytes':1000},{'effective_available_bytes':400},{'effective_available_bytes':150}]
        with patch(PREFIX+'available_memory',side_effect=samples):
            guard=memory.AutoMemoryLimit()
            self.assertEqual(800,guard.limit_for_rss(600))
            self.assertEqual(550,guard.limit_for_rss(600))

    def test_explicit_project_limit_and_auto_override(self):
        (self.root/'resource_policy.json').write_text(json.dumps({'max_process_tree_memory_gib':5}))
        self.assertEqual(5*GIB,_rss_limit(self.root,None))
        (self.root/'resource_policy.json').write_text(json.dumps({'max_rss_bytes':'auto'}))
        self.assertEqual('auto',_rss_limit(self.root,None))

    def test_small_real_child_returns_measured_auto_budget(self):
        result=run_bounded_subprocess([sys.executable,'-c','print("memory-smoke")'],timeout_seconds=10,max_rss_bytes='auto')
        self.assertEqual(0,result.returncode);self.assertIn(b'memory-smoke',result.stdout)
        self.assertEqual('auto',result.memory_budget['mode'])
        self.assertLessEqual(result.memory_budget['initial_max_rss_bytes'],result.memory_budget['initial']['effective_available_bytes'])

    def test_auto_guard_stops_only_its_small_owned_tree(self):
        peer=subprocess.Popen([sys.executable,'-c','import time;time.sleep(15)'])
        try:
            snapshot={'effective_available_bytes':16*1024**2}
            with patch(PREFIX+'available_memory',return_value=snapshot):
                with self.assertRaises(BoundedSubprocessError) as caught:
                    run_bounded_subprocess([sys.executable,'-c','import time; data=bytearray(32*1024**2); time.sleep(10)'],timeout_seconds=5,max_rss_bytes='auto')
            self.assertEqual('BOUNDED_SUBPROCESS_RSS',caught.exception.code)
            self.assertEqual('auto',caught.exception.memory_budget['mode'])
            self.assertIsNone(peer.poll())
        finally:
            peer.terminate();peer.wait(timeout=5)

    def test_real_timeout_keeps_auto_budget_evidence(self):
        with self.assertRaises(BoundedSubprocessError) as caught:
            run_bounded_subprocess([sys.executable,'-c','import time;time.sleep(10)'],timeout_seconds=.3,max_rss_bytes='auto')
        self.assertEqual('BOUNDED_SUBPROCESS_TIMEOUT',caught.exception.code)
        self.assertGreater(caught.exception.memory_budget['initial_max_rss_bytes'],0)

    def test_construction_respects_project_override_and_records_actual_limit(self):
        from ..slim_harness.construction_limits import run_simulation_bounded
        prefix='simulation_ready_cad_workflow.validation_pipeline.slim_harness.construction_limits.'
        (self.root/'resource_policy.json').write_text(json.dumps({'max_process_tree_memory_gib':5}))
        result=SimpleNamespace(returncode=0,memory_budget={'mode':'explicit','max_rss_bytes':5*GIB})
        with patch(prefix+'simulation_preflight',return_value={'effective_timeout_seconds':30,'configured_timeout_seconds':30}),patch(prefix+'run_bounded_subprocess',return_value=result) as run:
            run_simulation_bounded(['fixture'],scene={},geometry={},test={'id':'one'},policy={},history_dir=self.root/'construction-simulations',report_path=self.root/'cost.json')
        self.assertEqual(5*GIB,run.call_args.kwargs['max_rss_bytes'])
        self.assertEqual(result.memory_budget,json.loads((self.root/'cost.json').read_text())['memory_budget'])

    def test_no_available_memory_is_a_resource_outcome_before_execution(self):
        from contextlib import nullcontext
        from ..slim_harness.physics_invocation import run_physics_attempts
        budget=SimpleNamespace(limit=lambda seconds:seconds,active=lambda phase:nullcontext())
        def unavailable(*args,**kwargs):raise BoundedSubprocessError('BOUNDED_SUBPROCESS_MEMORY_UNAVAILABLE','No available memory')
        result=run_physics_attempts(workspace=self.root,project=None,folder=self.root/'attempt',scene={},test={'id':'one'},chrono_python=sys.executable,budget=budget,timeout_seconds=1,runner=unavailable)
        self.assertEqual('RESOURCE_POLICY_REQUIRED',result['status'])
        self.assertFalse(result['execution_started'])


if __name__=='__main__':unittest.main()
