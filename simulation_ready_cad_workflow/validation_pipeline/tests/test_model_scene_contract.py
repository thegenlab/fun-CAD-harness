"""模型层与正式源自测分层的实际回归；不放松正式Harness或修改源文件。"""
from pathlib import Path
import json,tempfile,unittest
from unittest.mock import patch
from ..slim_harness.contracts import DeliveryError,load_model_scene,load_scene
from benchmark_templates.cad_native import prepare,_scene_physics,SourcePhysicsError

class ModelSceneContractTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
    def model(self):
        return {"schema_version":"slim_scene_v1","units":"mm","bodies":[{"id":"frame","source":"frame.step","fixed":True},{"id":"arm","source":"arm.step","fixed":False}],"joints":[{"id":"hinge","type":"revolute","body_a":"arm","body_b":"frame","anchor":[0,0,0],"axis":[0,0,1],"damping":0.1,"region_a":{"points":[[0,0,-1],[0,0,1]],"radius":2},"region_b":{"points":[[0,0,-1],[0,0,1]],"radius":2}}],"gravity_m_s2":[0,0,-9.81]}
    def complete(self):
        value=self.model()
        value["tests"]=[{"id":"move","duration_s":1,"purpose":"normal operation","inputs":[{"joint":"hinge","mode":"effort","value":1,"max_effort":1}],"assertions":[{"id":"position","purpose":"measured movement","expression":"last(signal('x')) > 0"}]}]
        return value
    def write(self,value):
        path=self.root/"scene.json";path.write_text(json.dumps(value),encoding="utf-8");return path
    def test_model_accepts_missing_and_empty_source_tests(self):
        for value in (self.model(),{**self.model(),"tests":[]}):
            with self.subTest(tests=value.get("tests")):self.assertEqual(value,load_model_scene(self.write(value)))
    def test_formal_scene_still_requires_tests(self):
        for value in (self.model(),{**self.model(),"tests":[]}):
            with self.subTest(tests=value.get("tests")),self.assertRaisesRegex(DeliveryError,"Propose at least one"):
                load_scene(self.write(value))
    def test_model_keeps_operation_prose_without_validating_it_as_a_test(self):
        value=self.model();value["tests"]=[{"id":"normal_use","operation":"Turn the handle and release it"}]
        path=self.write(value);self.assertEqual(value,load_model_scene(path))
        with self.assertRaisesRegex(DeliveryError,"test duration"):load_scene(path)
    def test_invalid_axis_is_rejected_in_both_layers(self):
        value=self.complete();value["joints"][0]["axis"]=[0,0,0]
        for loader in (load_model_scene,load_scene):
            with self.subTest(loader=loader.__name__),self.assertRaisesRegex(DeliveryError,"axis must be nonzero"):loader(self.write(value))
    def test_negative_passive_damping_is_rejected_in_both_layers(self):
        value=self.complete();value["joints"][0]["damping"]=-1
        for loader in (load_model_scene,load_scene):
            with self.subTest(loader=loader.__name__),self.assertRaisesRegex(DeliveryError,"Passive damping"):loader(self.write(value))
    def test_model_gravity_is_still_checked_without_source_tests(self):
        value=self.model();value["gravity_m_s2"]=[0,0,float("nan")]
        with self.assertRaisesRegex(DeliveryError,"gravity must be finite"):load_model_scene(self.write(value))
    def test_model_spur_and_equivalent_headers_are_still_checked(self):
        for key in ("spur_gears","equivalent_elasticity"):
            value=self.model();value[key]={"schema_version":"unknown_physics"}
            with self.subTest(header=key),self.assertRaises(DeliveryError):load_model_scene(self.write(value))
    def test_valid_full_scene_is_returned_unchanged(self):
        value=self.complete();value["spur_gears"]=[]
        path=self.write(value);before=path.read_bytes()
        self.assertEqual(value,load_scene(path));self.assertEqual(value,load_model_scene(path));self.assertEqual(before,path.read_bytes())
    def test_full_scene_expression_validation_is_not_weakened(self):
        value=self.complete();value["tests"][0]["assertions"][0]["expression"]="1 < 2 and 3 < 4"
        path=self.write(value)
        self.assertEqual(value,load_model_scene(path))
        with self.assertRaisesRegex(DeliveryError,"Invalid measurement expression"):load_scene(path)
    def test_full_scene_error_order_remains_test_before_environment(self):
        value=self.model();value["gravity_m_s2"]=[0,0,float("nan")]
        with self.assertRaisesRegex(DeliveryError,"Propose at least one"):load_scene(self.write(value))
    def test_observed_joint_note_is_preserved_metadata_only(self):
        value=self.model()
        value["joints"][0]["note"]="理想转动副：输入轴轴承。连续转动此轴为源操作。"
        path=self.write(value)
        loaded=load_model_scene(path)
        self.assertEqual(value,loaded)
        _scene_physics(loaded)
        self.assertEqual(value["joints"][0]["note"],loaded["joints"][0]["note"])
        loaded["joints"][0]["hidden_drive_torque_nm"]=1
        with self.assertRaisesRegex(SourcePhysicsError,"UNSUPPORTED_SOURCE_FIELDS"):
            _scene_physics(loaded)
    def test_observed_design_notes_are_metadata_and_preserved(self):
        value=self.model()
        value["design_notes"]={"height_range_table_top_z_mm":[180,300],"internal_forces":["finite_stiffness_leadscrew_coupling"],"print_certified":False}
        loaded=load_model_scene(self.write(value))
        _scene_physics(loaded)
        self.assertEqual(value,loaded)
        self.assertNotIn("equivalent_elasticity",loaded)

    def test_observed_assembly_pose_only_is_metadata(self):
        value=self.model();value["assembly_pose"]={"crank_deg":101,"branch":"upper"}
        _scene_physics(value)
        value["hidden_motor_torque_nm"]=1
        with self.assertRaisesRegex(SourcePhysicsError,"UNSUPPORTED_SOURCE_FIELDS"):_scene_physics(value)
    def test_prepare_uses_test_free_view_preserves_source_and_does_not_move_geometry(self):
        from build123d import Box,Pos,export_step
        export_step(Pos(10,0,0)*Box(2,2,2),self.root/"body.step")
        value={"schema_version":"slim_scene_v1","units":"mm","bodies":[{"id":"body","source":"body.step","fixed":False,"density_kg_m3":1000}],"joints":[],"gravity_m_s2":[0,0,-9.81],"assembly_pose":{"position":[999,999,999],"note":"metadata only"},"authoring_sources":["../source_laws.py"],"tests":[{"id":"operation_text","operation":"Move the handle; no embedded test program"}]}
        path=self.write(value);before=path.read_bytes()
        from ..slim_harness.contracts import compile_scene
        with patch("simulation_ready_cad_workflow.validation_pipeline.slim_harness.contracts.compile_scene",wraps=compile_scene) as compiled:
            target=prepare(path,self.root/"prepared")
        self.assertEqual([],compiled.call_args.args[0]["tests"])
        result=json.loads(target.read_text(encoding="utf-8"))
        self.assertEqual([],result["tests"]);self.assertEqual(value,result["cad_loader"]["origin"]["source_scene"])
        self.assertEqual("not_executed_benchmark_owns_trials",result["cad_loader"]["source_tests_execution"])
        self.assertEqual(value["assembly_pose"],result["assembly_pose"]);self.assertEqual(value["authoring_sources"],result["authoring_sources"])
        self.assertAlmostEqual(.010,result["bodies"][0]["position_m"][0],places=8)
        self.assertEqual(before,path.read_bytes())

if __name__=="__main__":unittest.main()
