"""Containment uses original closed materials, never a welded body union."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import trimesh

from ..slim_harness.audit import audit_motion
from ..slim_harness.geometry import prepare_geometry


def pose(position, quaternion=(1,0,0,0)):
    return {"position_abs_m":list(position),"rotation_abs_quat_wxyz":list(quaternion)}


class SlimOriginalPartAuditTests(unittest.TestCase):
    def setUp(self):
        root=Path(__file__).resolve().parents[3]/".runtime_runs"/"slim_harness_20260907"/"audit_part_tests"
        root.mkdir(parents=True,exist_ok=True)
        self.temporary=tempfile.TemporaryDirectory(dir=root)
        self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)
        environment=mock.patch.dict(os.environ,{"SIMREADY_STEP_SHAPE_CACHE_DIR":str(self.root/"step_cache")})
        environment.start()
        self.addCleanup(environment.stop)

    def prepare(self,bodies,units="m"):
        source=self.root/"scene.json"
        source.write_text(json.dumps({"schema_version":"slim_scene_v1","units":units,"bodies":bodies}),encoding="utf-8")
        result=prepare_geometry(source,self.root/"geometry")
        self.assertEqual([],result["errors"])
        return result

    def test_touching_step_parts_do_not_become_nonmanifold_containment_fragments(self):
        import xml.parsers.expat
        from build123d import Box,Compound,Pos,export_step

        export_step(Compound(children=[Pos(-5,0,0)*Box(10,10,10),Pos(5,0,0)*Box(10,10,10)]),self.root/"frame.step")
        export_step(Pos(-5,0,20)*Box(1,1,1),self.root/"probe.step")
        geometry=self.prepare([
            {"id":"frame","source":"frame.step","fixed":True},
            {"id":"probe","source":"probe.step","fixed":False},
        ],"mm")
        frame=geometry["bodies"][0]
        self.assertFalse(trimesh.load_mesh(frame["mesh_file"],process=True).is_watertight,
                         "The fixture must expose cross-part vertex welding")
        separated={"time_s":0,"body_poses":{body["id"]:pose(body["position_m"]) for body in geometry["bodies"]}}
        report=audit_motion(geometry,[separated],self.root/"separated.json")
        self.assertTrue(report["passed"])
        self.assertEqual(2,report["containment"]["source_part_counts"]["frame"])
        self.assertEqual(2,report["containment"]["witness_counts"]["frame"])
        nested={"time_s":.001,"body_poses":{"frame":pose(frame["position_m"]),"probe":pose([-.005,0,0])}}
        report=audit_motion(geometry,[separated,nested],self.root/"nested.json")
        self.assertFalse(report["passed"])
        self.assertEqual(.001,report["violations"][0]["time_s"])
        self.assertGreater(report["max_overlap_m3"],1e-12)
        self.assertEqual(0,report["containment"]["mesh_part_contains_calls"])

    def test_overlapping_original_mesh_parts_use_union_not_even_odd_cancellation(self):
        frame=trimesh.Scene()
        for name,x in (("left",-.003),("right",.003)):
            mesh=trimesh.creation.box(extents=[.02,.02,.02])
            mesh.apply_translation([x,x/2,x/3])
            frame.add_geometry(mesh,node_name=name,geom_name=name)
        (self.root/"frame.glb").write_bytes(trimesh.exchange.gltf.export_glb(frame))
        probe=trimesh.creation.box(extents=[.001,.001,.001]);probe.apply_translation([0,0,.04]);probe.export(self.root/"probe.obj")
        geometry=self.prepare([
            {"id":"frame","source":"frame.glb","fixed":True},
            {"id":"probe","source":"probe.obj","fixed":False},
        ])
        combined=trimesh.load_mesh(geometry["bodies"][0]["mesh_file"],process=False)
        self.assertFalse(combined.contains([[0,0,0]])[0],"Combined parity incorrectly treats the overlap as empty")
        sample={"time_s":0,"body_poses":{"frame":pose([0,0,0]),"probe":pose([0,0,0])}}
        report=audit_motion(geometry,[sample],self.root/"mesh-union.json")
        self.assertFalse(report["passed"])
        self.assertEqual("mesh_containment",report["violations"][0]["kind"])
        self.assertEqual(2,report["containment"]["witness_counts"]["frame"])

    def test_mesh_cavity_shells_remain_one_material_and_do_not_fill_the_void(self):
        outer=trimesh.creation.box(extents=[.02,.02,.02])
        cavity=trimesh.creation.box(extents=[.01,.01,.01]);cavity.invert()
        trimesh.util.concatenate([outer,cavity]).export(self.root/"hollow.obj")
        probe=trimesh.creation.box(extents=[.001,.001,.001]);probe.apply_translation([0,0,.03]);probe.export(self.root/"probe.obj")
        geometry=self.prepare([
            {"id":"hollow","source":"hollow.obj","fixed":True},
            {"id":"probe","source":"probe.obj","fixed":False},
        ])
        hole={"time_s":0,"body_poses":{"hollow":pose([0,0,0]),"probe":pose([0,0,0])}}
        self.assertTrue(audit_motion(geometry,[hole],self.root/"mesh-void.json")["passed"])
        material={"time_s":.001,"body_poses":{"hollow":pose([0,0,0]),"probe":pose([.008,0,0])}}
        report=audit_motion(geometry,[hole,material],self.root/"mesh-material.json")
        self.assertFalse(report["passed"])
        self.assertEqual(.001,report["violations"][0]["time_s"])
        self.assertEqual(1,report["containment"]["source_part_counts"]["hollow"])
        self.assertEqual(2,report["containment"]["witness_counts"]["hollow"])

    def test_original_step_annular_hole_with_translated_rotated_world_pose(self):
        import xml.parsers.expat
        from build123d import Box,Cylinder,Pos,export_step

        export_step(Pos(100,200,300)*(Cylinder(10,10)-Cylinder(5,12)),self.root/"annulus.step")
        export_step(Pos(100,200,340)*Box(1,1,1),self.root/"probe.step")
        geometry=self.prepare([
            {"id":"annulus","source":"annulus.step","fixed":False},
            {"id":"probe","source":"probe.step","fixed":False},
        ],"mm")
        q=[np.sqrt(.5),np.sqrt(.5),0,0]
        empty_hole={"time_s":0,"body_poses":{"annulus":pose([1,2,3],q),"probe":pose([1,2,3],q)}}
        report=audit_motion(geometry,[empty_hole],self.root/"step-hole.json")
        self.assertTrue(report["passed"])
        self.assertGreater(report["containment"]["step_classification_queries"],0)
        inside_material={"time_s":.001,"body_poses":{"annulus":pose([1,2,3],q),"probe":pose([1.007,2,3],q)}}
        report=audit_motion(geometry,[empty_hole,inside_material],self.root/"step-material.json")
        self.assertFalse(report["passed"])
        self.assertEqual(.001,report["violations"][0]["time_s"])
        self.assertGreater(report["max_overlap_m3"],1e-12)
        self.assertEqual(0,report["containment"]["mesh_part_contains_calls"])


if __name__=="__main__": unittest.main()
