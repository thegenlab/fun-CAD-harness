"""Original BRep bounds for small declared wheel/floor numerical contact only."""
from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from ..slim_harness.construction_planar_contact import (
    PlanarContactProofError, PlanarSupportContactProof, _faces, _plane, _transform,
)
from ..slim_harness.geometry import prepare_geometry


class ConstructionPlanarContactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import json
        from build123d import Box, Cylinder, Pos, Rot, export_step

        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        floor = Pos(0, 0, -10) * Box(1000, 1000, 20)
        wheel = Pos(0, 0, 50) * Rot(90, 0, 0) * (Cylinder(50, 20) - Cylinder(35, 22))
        cls.floor_shape, cls.wheel_shape = floor.wrapped, wheel.wrapped
        export_step(floor, cls.root / 'floor.step')
        export_step(wheel, cls.root / 'wheel.step')
        scene = {'schema_version': 'slim_scene_v1', 'units': 'mm', 'tests': [], 'joints': [], 'bodies': [
            {'id': 'floor', 'source': 'floor.step', 'fixed': True},
            {'id': 'wheel', 'source': 'wheel.step', 'fixed': False}]}
        path = cls.root / 'scene.json'
        path.write_text(json.dumps(scene))
        geometry = prepare_geometry(path, cls.root / 'geometry')
        if geometry['errors']:
            raise AssertionError(geometry['errors'])
        cls.bodies = geometry['bodies']
        face_index = next(i for i, face in enumerate(_faces(cls.floor_shape), 1)
                          if np.dot(_plane(face)[1], [0, 0, 1]) > .999)
        cls.declaration = {'schema_version': 'slim_construction_planar_contact_v1',
            'stage': 'benchmark_construction', 'contact_kind': 'wheel_tread_on_planar_support',
            'body_pair': ['floor', 'wheel'], 'support_body': 'floor', 'tread_body': 'wheel',
            'support_solid_id': 'solid_1', 'support_face_index': face_index,
            'source_bindings': {body['id']: {key: body[key] for key in ['source_sha256', 'source_solid_ids']} for body in cls.bodies},
            'tread_region': {'axis_origin_source_m': [0, 0, .05], 'axis_direction_source': [0, 1, 0],
                             'axial_interval_m': [-.01, .01], 'inner_radius_m': .035, 'outer_radius_m': .0502}}

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def proof(self, declaration=None, **kwargs):
        return PlanarSupportContactProof.from_source(declaration or self.declaration, self.bodies,
            native_penetration_tolerance_m=.00005, gravity_m_s2=[0, 0, -9.81], **kwargs)

    def contact(self, downward_mm):
        from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
        wheel = _transform(self.wheel_shape, np.eye(3), [0, 0, -downward_mm])
        operation = BRepAlgoAPI_Common(self.floor_shape, wheel)
        self.assertTrue(operation.IsDone())
        poses = {body['id']: {'position_abs_m': np.asarray(body['position_m']).copy(),
                             'rotation_abs_quat_wxyz': body['rotation_wxyz']} for body in self.bodies}
        poses['wheel']['position_abs_m'][2] -= downward_mm * .001
        return operation.Shape(), poses

    def test_shallow_original_contact_in_tread_and_actual_floor_face_is_allowed(self):
        shape, poses = self.contact(.02)
        report = self.proof().prove(['wheel', 'floor'], shape, poses, native_max_penetration_m=.00001)
        self.assertTrue(report['allowed'], report)
        self.assertAlmostEqual(.00002, report['maximum_depth_m'], delta=1e-8)
        wheel=next(body for body in self.bodies if body['id']=='wheel')
        expected=wheel['fidelity']['step_linear_deflection_m']+.00005+1e-8
        self.assertAlmostEqual(expected, report['depth_limit_m'])
        self.assertFalse(report['native_contacts_or_friction_modified'])
        self.assertFalse(report['original_geometry_modified'])

    def test_deep_contact_is_rejected_even_if_overlap_volume_is_tiny(self):
        from build123d import Box, Pos
        shape, poses = self.contact(.2)
        report = self.proof().prove(['wheel', 'floor'], shape, poses, native_max_penetration_m=0.)
        self.assertFalse(report['allowed'])
        self.assertEqual('original_depth_exceeds_contact_band', report['reason'])
        tiny = (Pos(50, 0, -.1) * Box(.001, .001, .2)).wrapped
        report = self.proof().prove(['wheel', 'floor'], tiny, poses, native_max_penetration_m=0.)
        self.assertLess(report['original_overlap_m3'], 1e-12)
        self.assertFalse(report['allowed'])

    def test_excludes_non_tread_material_unrelated_pairs_and_native_failure(self):
        shape, poses = self.contact(.02)
        narrow = copy.deepcopy(self.declaration)
        narrow['tread_region']['inner_radius_m'] = .049999
        report = self.proof(narrow).prove(['wheel', 'floor'], shape, poses, native_max_penetration_m=0.)
        self.assertFalse(report['allowed'])
        self.assertGreater(report['outside_tread_region_m3'], 1e-12)
        self.assertFalse(self.proof().prove(['wheel', 'frame'], shape, poses, native_max_penetration_m=0.)['allowed'])
        self.assertFalse(self.proof().prove(['wheel', 'floor'], shape, poses, native_max_penetration_m=.00006)['allowed'])

    def test_rejects_frozen_scope_stale_sources_and_joint_interference(self):
        for field, value in [('stage', 'frozen_evaluation'), ('contact_kind', 'bearing_joint')]:
            declaration = copy.deepcopy(self.declaration)
            declaration[field] = value
            with self.assertRaises(PlanarContactProofError):
                self.proof(declaration)
        declaration = copy.deepcopy(self.declaration)
        declaration['source_bindings']['wheel']['source_sha256'] = '0' * 64
        with self.assertRaises(PlanarContactProofError):
            self.proof(declaration)
        with self.assertRaises(PlanarContactProofError):
            self.proof(joints=[{'body_a': 'floor', 'body_b': 'wheel', 'type': 'revolute'}])

    def test_finite_support_patch_cannot_hide_contact_beyond_its_edges(self):
        from build123d import Box, Pos
        _shape, poses = self.contact(.02)
        outside = (Pos(501, 0, -.01) * Box(1, 1, .02)).wrapped
        report = self.proof().prove(['wheel', 'floor'], outside, poses, native_max_penetration_m=0.)
        self.assertFalse(report['allowed'])
        self.assertGreater(report['outside_support_band_m3'], 1e-12)

    def test_fast_body_bound_proof_matches_shallow_and_deep_contact(self):
        proof = self.proof()
        self.assertIsNotNone(proof.fast_data)
        for depth_mm, accepted in [(.02, True), (.2, False)]:
            shape, poses = self.contact(depth_mm)
            slow = proof.prove(['floor', 'wheel'], shape, poses, native_max_penetration_m=0.)
            fast = proof.prove_body(['floor', 'wheel'], poses, native_max_penetration_m=0.)
            self.assertEqual(accepted, fast['allowed'], fast)
            self.assertEqual(slow['allowed'], fast['allowed'])
            self.assertAlmostEqual(slow['maximum_depth_m'], fast['maximum_depth_m'], delta=1e-8)
        _shape, poses = self.contact(.02)
        poses['wheel']['position_abs_m'][0] += .49
        edge = proof.prove_body(['floor', 'wheel'], poses, native_max_penetration_m=0.)
        self.assertFalse(edge['allowed'])
        self.assertIn('support_edges', edge['reason'])

    def test_fast_proof_cannot_skip_non_tread_material(self):
        declaration = copy.deepcopy(self.declaration)
        declaration['tread_region']['inner_radius_m'] = .049999
        _shape, poses = self.contact(.02)
        result = self.proof(declaration).prove_body(['floor', 'wheel'], poses, native_max_penetration_m=0.)
        self.assertFalse(result['allowed'])
        self.assertIn('non_tread', result['reason'])


if __name__ == '__main__':
    unittest.main()
