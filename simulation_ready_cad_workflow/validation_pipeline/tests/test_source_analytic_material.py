"""Original analytic material must not bridge a hollow or sealed cavity."""
import unittest

from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeBox
from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt

from ..slim_harness.source_analytic_material import contained_axial_capsules


class SourceAnalyticMaterialTests(unittest.TestCase):
    @staticmethod
    def fused_solid(parts):
        from OCP.TopAbs import TopAbs_SOLID
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopoDS import TopoDS
        shape = parts[0]
        for part in parts[1:]:
            fuse = BRepAlgoAPI_Fuse(shape, part)
            if not fuse.IsDone():
                raise AssertionError('Source fixture fuse failed')
            shape = fuse.Shape()
        solids = []
        explorer = TopExp_Explorer(shape, TopAbs_SOLID)
        while explorer.More():
            solids.append(TopoDS.Solid_s(explorer.Current()))
            explorer.Next()
        if len(solids) != 1:
            raise AssertionError('Source fixture must be one connected solid')
        return solids[0]

    def test_two_coaxial_followers_connected_above_do_not_bridge_air_gap(self):
        # W12 同源结构：两根横向柱由立柱和上梁连成一个实体，轴向中间仍为空气。
        parts = [BRepPrimAPI_MakeBox(gp_Pnt(-26,-1,39),52,2,2).Solid()]
        for x in (-25,25):
            parts.append(BRepPrimAPI_MakeBox(gp_Pnt(x-1,-1,30),2,2,10).Solid())
            parts.append(BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(x-3,0,30),gp_Dir(1,0,0)),2,6).Solid())
        capsules = contained_axial_capsules(self.fused_solid(parts))
        self.assertEqual(2, len(capsules))
        for capsule, interval in zip(capsules, ([-28.,-22.], [22.,28.])):
            self.assertEqual(interval, capsule['proof']['source_cylinder_axial_interval_mm'])
            self.assertAlmostEqual(.002, capsule['radius_m'])
            self.assertAlmostEqual(.002, capsule['straight_length_m'])
            self.assertEqual(0., capsule['proof']['missing_volume_mm3'])
            self.assertEqual('connected_exposed_interval_after_whole_group_failed',
                             capsule['proof']['interval_selection'])

    def test_successful_long_candidate_still_spans_hidden_source_material(self):
        # 整轴中段被实体方套遮住，只剩两端圆柱面；完整原料仍应得到原长候选。
        cylinder = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(-28,0,30),gp_Dir(1,0,0)),2,56).Solid()
        collar = BRepPrimAPI_MakeBox(gp_Pnt(-22,-3,27),44,6,6).Solid()
        capsules = contained_axial_capsules(self.fused_solid([cylinder, collar]))
        self.assertEqual(1, len(capsules))
        capsule = capsules[0]
        self.assertEqual([-28.,28.], capsule['proof']['source_cylinder_axial_interval_mm'])
        self.assertEqual(2, len(capsule['proof']['source_face_axial_intervals_mm']))
        self.assertAlmostEqual(.052, capsule['straight_length_m'])
        self.assertEqual(0., capsule['proof']['missing_volume_mm3'])
        self.assertNotIn('interval_selection', capsule['proof'])

    def test_offset_y_axis_solid_cylinder_has_contained_capsule(self):
        cylinder = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(-60,-34,80), gp_Dir(0,1,0)), 3, 68).Solid()
        capsules = contained_axial_capsules(cylinder)
        self.assertEqual(1, len(capsules))
        capsule = capsules[0]
        for actual, expected in zip(capsule['center_source_m'], [-.06,0,.08]):
            self.assertAlmostEqual(expected, actual, places=12)
        self.assertEqual([0.,1.,0.], capsule['axis_source'])
        self.assertAlmostEqual(.003, capsule['radius_m'])
        self.assertAlmostEqual(.062, capsule['straight_length_m'])
        self.assertAlmostEqual(.068, capsule['source_cylinder_length_m'])
        self.assertEqual(0., capsule['proof']['missing_volume_mm3'])
        self.assertTrue(capsule['proof']['cut_is_done'])

    def test_open_hollow_tube_never_becomes_filled_outer_cylinder(self):
        outer = BRepPrimAPI_MakeCylinder(4,30).Solid()
        bore = BRepPrimAPI_MakeCylinder(2,30).Solid()
        cut = BRepAlgoAPI_Cut(outer,bore)
        self.assertTrue(cut.IsDone())
        from OCP.TopAbs import TopAbs_SOLID
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopoDS import TopoDS
        explorer = TopExp_Explorer(cut.Shape(),TopAbs_SOLID)
        self.assertTrue(explorer.More())
        self.assertEqual([], contained_axial_capsules(TopoDS.Solid_s(explorer.Current())))

    def test_enclosed_cavity_is_not_filled_by_an_outer_surface_claim(self):
        outer = BRepPrimAPI_MakeCylinder(4,30).Solid()
        cavity = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(0,0,10),gp_Dir(0,0,1)),2,10).Solid()
        cut = BRepAlgoAPI_Cut(outer,cavity)
        self.assertTrue(cut.IsDone())
        from OCP.TopAbs import TopAbs_SOLID
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopoDS import TopoDS
        explorer = TopExp_Explorer(cut.Shape(),TopAbs_SOLID)
        self.assertTrue(explorer.More())
        self.assertEqual([], contained_axial_capsules(TopoDS.Solid_s(explorer.Current())))


if __name__ == '__main__':
    unittest.main()
