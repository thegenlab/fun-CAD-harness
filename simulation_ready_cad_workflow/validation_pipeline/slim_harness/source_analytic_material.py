"""Source-derived material supplements; the complete original mesh stays.

An axial capsule is added only when an entire cylinder, reconstructed from
original analytic faces, has no missing material in an OCCT Boolean cut.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ['contained_axial_capsules']


def contained_axial_capsules(solid) -> list[dict]:
    """Return capsules contained in one original OCP solid in millimetres.

Coaxial exposed face intervals may span hidden material between faces. Try
that complete interval first; if it fails, try connected exposed intervals
separately. Every candidate must pass the same exact zero-missing-volume test.
"""
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRepGProp import BRepGProp
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
    from OCP.GeomAbs import GeomAbs_Cylinder
    from OCP.GProp import GProp_GProps
    from OCP.TopAbs import TopAbs_FACE, TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopTools import TopTools_ListOfShape
    from OCP.TopoDS import TopoDS
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt

    if solid is None or solid.IsNull() or solid.ShapeType() != TopAbs_SOLID:
        return []
    if not BRepCheck_Analyzer(solid).IsValid():
        return []
    groups = []
    explorer = TopExp_Explorer(solid, TopAbs_FACE)
    face_index = 0
    while explorer.More():
        face = TopoDS.Face_s(explorer.Current())
        try:
            surface = BRepAdaptor_Surface(face, True)
            if surface.GetType() == GeomAbs_Cylinder:
                cylinder = surface.Cylinder()
                direction = cylinder.Axis().Direction()
                axis = np.array([direction.X(), direction.Y(), direction.Z()])
                axis /= np.linalg.norm(axis)
                if axis[int(np.argmax(np.abs(axis)))] < 0:
                    axis = -axis
                location = cylinder.Location()
                location = np.array([location.X(), location.Y(), location.Z()])
                anchor = location-axis*float(location@axis)
                radius = float(cylinder.Radius())
                limits = [surface.FirstUParameter(), surface.LastUParameter(),
                          surface.FirstVParameter(), surface.LastVParameter()]
                if radius > 0 and np.isfinite([radius, *axis, *anchor, *limits]).all():
                    u = (limits[0]+limits[1])/2
                    points = [surface.Value(u, v) for v in limits[2:]]
                    points = [np.array([point.X(), point.Y(), point.Z()]) for point in points]
                    # These tolerances only group nominally identical source
                    # surfaces. They never relax the Boolean containment test.
                    group = next((item for item in groups if abs(item['radius']-radius) <= 1e-9
                                  and np.linalg.norm(item['axis']-axis) <= 1e-10
                                  and np.linalg.norm(item['anchor']-anchor) <= 1e-9), None)
                    if group is None:
                        group = {'radius': radius, 'axis': axis, 'anchor': anchor, 'intervals': [], 'faces': []}
                        groups.append(group)
                    interval = sorted(float((point-group['anchor'])@group['axis']) for point in points)
                    group['intervals'].append(interval)
                    group['faces'].append(face_index)
        except (RuntimeError, ValueError, ArithmeticError):
            # An unrecognized face cannot authorize additional material.
            pass
        explorer.Next()
        face_index += 1

    def contained_candidate(group):
        lower = min(interval[0] for interval in group['intervals'])
        upper = max(interval[1] for interval in group['intervals'])
        length, radius = upper-lower, group['radius']
        if not math.isfinite(length) or length <= 2*radius:
            return None
        axis = group['axis']
        start = group['anchor']+lower*axis
        center = group['anchor']+((lower+upper)/2)*axis
        try:
            builder = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(*map(float, start)), gp_Dir(*map(float, axis))), radius, length)
            candidate = builder.Solid()
            if not builder.IsDone():
                return None
            arguments, tools = TopTools_ListOfShape(), TopTools_ListOfShape()
            arguments.Append(candidate)
            tools.Append(solid)
            cut = BRepAlgoAPI_Cut()
            cut.SetArguments(arguments)
            cut.SetTools(tools)
            cut.SetNonDestructive(True)
            cut.Build()
            if not cut.IsDone():
                return None
            missing = GProp_GProps()
            BRepGProp.VolumeProperties_s(cut.Shape(), missing)
            missing_volume = abs(float(missing.Mass()))
            if not math.isfinite(missing_volume) or missing_volume != 0.:
                return None
            candidate_properties = GProp_GProps()
            BRepGProp.VolumeProperties_s(candidate, candidate_properties)
            candidate_volume = float(candidate_properties.Mass())
            if not math.isfinite(candidate_volume) or candidate_volume <= 0:
                return None
        except (RuntimeError, ValueError, ArithmeticError):
            return None
        return {
            'type': 'capsule', 'center_source_m': (center*.001).tolist(), 'axis_source': axis.tolist(),
            'radius_m': radius*.001, 'straight_length_m': (length-2*radius)*.001,
            'source_cylinder_length_m': length*.001,
            'proof': {
                'method': 'OCCT non-destructive Cut(complete_source_derived_cylinder, original_solid)',
                'cut_is_done': True, 'missing_volume_mm3': missing_volume,
                'candidate_cylinder_volume_mm3': candidate_volume,
                'source_cylinder_face_indices': group['faces'],
                'source_face_axial_intervals_mm': group['intervals'],
                'source_axis_anchor_mm': group['anchor'].tolist(),
                'source_cylinder_axial_interval_mm': [lower, upper],
                'capsule_containment': 'Same radius; straight segment length L-2r; hemispherical ends stay between the containing cylinder end planes',
                'original_solid_modified': False,
            },
        }

    capsules = []
    for group in groups:
        # 原整组候选成功时原样保留，包括被其它材料遮住的连续长轴。
        candidate = contained_candidate(group)
        if candidate is not None:
            capsules.append(candidate)
            continue
        # 整组失败后只按暴露区间实际相交/相接分组，不用容差跨越空隙。
        # 每一组仍需通过同一严格 Cut，不能把空腔或缺料当成支承材料。
        components = []
        for interval, face in sorted(zip(group['intervals'], group['faces'])):
            if not components or interval[0] > components[-1]['upper']:
                components.append({'upper': interval[1], 'intervals': [], 'faces': []})
            component = components[-1]
            component['upper'] = max(component['upper'], interval[1])
            component['intervals'].append(interval)
            component['faces'].append(face)
        if len(components) == 1:
            continue  # 同一区间已经失败，不重复求解同一个布尔运算。
        for component in components:
            candidate = contained_candidate({**group, 'intervals': component['intervals'],
                                             'faces': component['faces']})
            if candidate is not None:
                candidate['proof']['interval_selection'] = 'connected_exposed_interval_after_whole_group_failed'
                capsules.append(candidate)
    return capsules
