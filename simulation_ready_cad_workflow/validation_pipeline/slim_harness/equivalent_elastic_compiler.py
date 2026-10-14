"""Compile explicit, source-bound nominal elasticity into the small runtime ABI.

No FE calibration, manufacturing-rest field, author callback or time script is
required or executed. Original bodies/masses/joints stay in the prepared scene.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .equivalent_elasticity import validate_equivalent_elasticity
from .validation_scope import make_validation_scope


SCHEMA = "slim_equivalent_elasticity_v1"
KINDS = {"torsional_spring","point_axis_spring","linear_spring","friction_clutch","profile_detent","shaft_support"}


class EquivalentCompileError(ValueError):
    pass


def _hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def _value_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),
                                    ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def _number(value,label,positive=False):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or (positive and value<=0):
        raise EquivalentCompileError(label+" must be a finite"+(" positive" if positive else "")+" number")
    return float(value)


def _vector(value,label,unit=False):
    if not isinstance(value,list) or len(value)!=3:
        raise EquivalentCompileError(label+" needs three metre-based numbers")
    vector=np.asarray([_number(x,label) for x in value])
    if unit and abs(np.linalg.norm(vector)-1)>1e-8:
        raise EquivalentCompileError(label+" must be a unit direction")
    return vector


def validate_equivalent_header(declaration):
    """Light scene-load validation; numerical/runtime/source checks follow compilation."""
    if not isinstance(declaration,dict) or declaration.get("schema_version")!=SCHEMA or declaration.get("units")!="m":
        raise EquivalentCompileError("Expected an explicit metre-based slim_equivalent_elasticity_v1 declaration")
    for key in ("elastic_reference_body_ids","flexible_contact_regions","elements"):
        if not isinstance(declaration.get(key),list):
            raise EquivalentCompileError(key+" must be explicitly supplied as a list")
    if not declaration["elements"]:
        raise EquivalentCompileError("Declare at least one actual equivalent element")
    ids=[row.get("id") for row in declaration["elements"]]
    if any(not isinstance(key,str) or not key for key in ids) or len(ids)!=len(set(ids)):
        raise EquivalentCompileError("Equivalent elements need unique nonempty IDs")
    for row in declaration["elements"]:
        if row.get("kind") not in KINDS:
            raise EquivalentCompileError("CAPABILITY_GAP: unsupported equivalent element "+str(row.get("kind")))
        if not isinstance(row.get("parameter_basis"),dict) or not row.get("source_body_ids"):
            raise EquivalentCompileError("Each equivalent element needs explicit parameter_basis and source_body_ids")


def _bind_refs(refs,design,bindings):
    if not isinstance(refs,list) or not refs:
        raise EquivalentCompileError("Nominal parameters/regions need actual source_refs")
    bound=[]
    for source in refs:
        path=Path(source["path"])
        if path.is_absolute():
            raise EquivalentCompileError("Author source_refs paths must be relative to the design")
        target=(design/path).resolve()
        if not target.is_file() or not source.get("locator"):
            raise EquivalentCompileError("Source evidence file or locator is missing: "+str(path))
        actual=_hash(target)
        if source.get("sha256") is not None and source["sha256"]!=actual:
            raise EquivalentCompileError("Declared source evidence is stale: "+str(path))
        bindings[str(target)]=actual
        bound.append({"path":str(target),"sha256":actual,"locator":source["locator"]})
    return bound


def derive_nominal_stiffness(element):
    """Use explicit author nominal inputs; do not claim measured or calibrated k."""
    basis=element["parameter_basis"]
    if basis.get("method") not in {"geometry_material_nominal","declared_nominal"}:
        raise EquivalentCompileError("Specify geometry_material_nominal or declared_nominal parameter basis")
    if basis.get("calibration")!="not_calibrated" or not basis.get("rationale") or not basis.get("operating_scope"):
        raise EquivalentCompileError("State the nominal rationale, operating_scope and calibration:not_calibrated")
    if element["kind"] == "shaft_support":
        from .shaft_supports import validate_support
        if basis["method"] != "declared_nominal" or basis.get("derivation") is not None:
            raise EquivalentCompileError("Shaft supports require explicit nominal bearing and stop parameters")
        validate_support(element)
        return
    key="stiffness_nm_rad" if element["kind"]=="torsional_spring" else "normal_stiffness_n_m" if element["kind"]=="friction_clutch" else "stiffness_n_m"
    derivation=basis.get("derivation")
    if basis["method"]=="declared_nominal":
        if derivation is not None:
            raise EquivalentCompileError("Use geometry_material_nominal when asking the compiler to derive stiffness")
        _number(element.get(key),key,True)
        return
    if not isinstance(derivation,dict):
        raise EquivalentCompileError("Geometry/material nominal stiffness requires its explicit derivation inputs")
    E=_number(derivation["young_modulus_pa"],"Young modulus",True)
    width=_number(derivation["width_m"],"nominal width",True)
    thickness=_number(derivation["thickness_m"],"nominal bending thickness",True)
    length=_number(derivation["length_m"],"nominal effective free length",True)
    parallel=derivation.get("parallel_count",1)
    if isinstance(parallel,bool) or not isinstance(parallel,int) or parallel<1:
        raise EquivalentCompileError("parallel_count must be a positive integer")
    inertia=width*thickness**3/12
    formula=derivation["formula"]
    if formula=="rectangular_strip_torsion" and key=="stiffness_nm_rad":
        stiffness=parallel*E*inertia/length
    elif formula=="cantilever_rectangular_leaf" and key!="stiffness_nm_rad":
        stiffness=parallel*3*E*inertia/length**3
    else:
        raise EquivalentCompileError("CAPABILITY_GAP: unsupported nominal stiffness formula for "+element["kind"])
    if key in element and not math.isclose(float(element[key]),stiffness,rel_tol=1e-8,abs_tol=0.):
        raise EquivalentCompileError("Supplied stiffness disagrees with its declared nominal derivation")
    element[key]=stiffness
    basis["compiler_derivation"]={"formula":formula,"second_moment_m4":inertia,
        "result_field":key,"nominal_value":stiffness,
        "input_authority":"Explicit author-declared nominal geometry/material parameters, source files bound; not dimensional manufacturing or FE calibration evidence"}


def _rotation(body):
    import trimesh
    return trimesh.transformations.quaternion_matrix(body["rotation_wxyz"])[:3,:3]


def _world_point(body,local):
    return np.asarray(body["position_m"])+_rotation(body)@_vector(local,"body-relative point")


def _profile_from_source(element,records,bindings,design):
    """Sample a radius follower against the actual static STEP contour.

    Only a translation along the actual declared prismatic pair is supported.
    The resulting piecewise-linear h(q) is geometric input, not a timed latch.
    """
    from ..intake.step_shape_cache import load_step_shape
    from ..mechanism.occt_collision_fidelity import distance_points_to_shape_m
    source=element.get("profile_source")
    if not isinstance(source,dict) or source.get("kind")!="step_radius_follower":
        raise EquivalentCompileError("CAPABILITY_GAP: profile_detent needs a program-collected step_radius_follower profile")
    owner=source["source_body_id"]
    if owner not in (element["body_a"],element["body_b"]):
        raise EquivalentCompileError("Profile source must be one of the actual translating pair")
    body=records[owner]
    path=Path(body["source_file"])
    if path.suffix.lower() not in {".step",".stp"}:
        raise EquivalentCompileError("CAPABILITY_GAP: geometric detent extraction currently requires original STEP")
    source["source_refs"]=_bind_refs(source["source_refs"],design,bindings)
    loaded=load_step_shape(path,unit="MM")
    if loaded.source_sha256!=body["source_sha256"]:
        raise EquivalentCompileError("Profile source STEP changed")
    requested=source["source_solid_ids"]
    if not requested or len(requested)!=len(set(requested)) or not set(requested)<=set(body["source_solid_ids"]):
        raise EquivalentCompileError("Profile must identify actual source solid IDs")
    shapes=[loaded.solids[int(key.split("_")[-1])-1] for key in requested]
    origin=_vector(source["follower_center_at_source_pose_m"],"source follower center")
    normal=_vector(source["normal_direction_source"],"source profile normal",True)
    radius=_number(source["follower_radius_m"],"source follower radius",True)
    search=source["normal_offset_search_m"]
    if len(search)!=2 or _number(search[0],"normal search lower")>=_number(search[1],"normal search upper"):
        raise EquivalentCompileError("Source profile search must be a finite increasing normal interval")
    a,b=records[element["body_a"]],records[element["body_b"]]
    axis_world=_rotation(a)@_vector(element["axis_a_body"],"root travel axis",True)
    q0=float(axis_world@(_world_point(b,element["point_b_body_m"])-_world_point(a,element["point_a_body_m"])))
    direction=_rotation(body).T@axis_world*(1 if owner==element["body_a"] else -1)
    if abs(float(direction@normal))>1e-8:
        raise EquivalentCompileError("Profile travel and follower normal must be perpendicular")
    divisions=source["divisions"]
    if isinstance(divisions,bool) or not isinstance(divisions,int) or not 2<=divisions<=256:
        raise EquivalentCompileError("Declare 2..256 source profile intervals for this bounded collector")
    lower,upper=map(float,element["working_range_m"])
    if not math.isfinite(lower+upper) or lower>=upper:
        raise EquivalentCompileError("Profile working range must increase")
    knots=[]
    for coordinate in np.linspace(lower,upper,divisions+1):
        center=origin+(coordinate-q0)*direction
        def gap(offset):
            point=center+float(offset)*normal
            return min(float(distance_points_to_shape_m(shape,[point])[0]) for shape in shapes)-radius
        lo,hi=map(float,search)
        if gap(lo)>0 or gap(hi)<0:
            raise EquivalentCompileError("Profile follower contact is not bracketed in the declared source interval")
        for _ in range(32):
            middle=(lo+hi)/2
            if gap(middle)>=0:hi=middle
            else:lo=middle
        knots.append({"coordinate_m":float(coordinate),"height_m":(lo+hi)/2})
    element["profile_knots"]=knots
    source["source_sha256"]=loaded.source_sha256
    source["compiler_profile"]={"coordinate_at_source_pose_m":q0,"travel_direction_source":direction.tolist(),
        "normal_solve_width_m":(search[1]-search[0])/2**32,
        "maximum_knot_spacing_m":(upper-lower)/divisions,
        "representation":"Continuous piecewise-linear original-STEP radius-follower contour; no time/state lookup; interpolation error is an uncalibrated model approximation"}


def compile_equivalent_elasticity(scene,geometry):
    declaration=scene.get("equivalent_elasticity")
    if declaration is None:
        return None
    validate_equivalent_header(declaration)
    if scene.get("fea") is not None:
        raise EquivalentCompileError("Choose one explicit elastic representation; do not combine FE and equivalent forces")
    scene_path=Path(geometry["scene_file"]).resolve();design=scene_path.parent
    if not scene_path.is_file() or _hash(scene_path)!=geometry["scene_sha256"]:
        raise EquivalentCompileError("Current source scene does not match prepared geometry")
    records={body["id"]:body for body in geometry["bodies"]}
    specs={body["id"]:body for body in scene["bodies"]}
    if set(records)!=set(specs) or geometry.get("errors"):
        raise EquivalentCompileError("Equivalent compiler requires complete current original geometry records")
    bindings={str(scene_path):geometry["scene_sha256"]}
    checked=set()
    for key,body in records.items():
        path=Path(body["source_file"]).resolve()
        if path!=(design/specs[key]["source"]).resolve():
            raise EquivalentCompileError("Source geometry path does not match scene body "+key)
        if path not in checked:
            if not path.is_file() or _hash(path)!=body["source_sha256"]:
                raise EquivalentCompileError("Source geometry changed: "+key)
            checked.add(path)
        bindings[str(path)]=body["source_sha256"]
    value=copy.deepcopy(declaration)
    references=value["elastic_reference_body_ids"]
    if len(references)!=len(set(references)) or not set(references)<=set(records):
        raise EquivalentCompileError("Elastic references must be unique actual source body IDs")
    model_sources=set()
    for element in value["elements"]:
        sources=element["source_body_ids"]
        if len(sources)!=len(set(sources)) or not set(sources)<=set(records):
            raise EquivalentCompileError("Element source_body_ids must identify distinct actual bodies")
        model_sources.update(sources)
        basis=element["parameter_basis"]
        basis["source_refs"]=_bind_refs(basis.get("source_refs"),design,bindings)
        derive_nominal_stiffness(element)
        if element["kind"] == "shaft_support":
            from .shaft_supports import compile_support
            compile_support(element, records)
        if element["kind"]=="profile_detent":
            pairs=[joint for joint in scene.get("joints",[]) if joint.get("type")=="prismatic"
                   and {joint["body_a"],joint["body_b"]}=={element["body_a"],element["body_b"]}]
            if len(pairs)!=1:
                raise EquivalentCompileError("CAPABILITY_GAP: geometric detent profile needs its actual single prismatic pair")
            joint_axis=np.asarray(pairs[0].get("axis",[0.,0.,1.]),dtype=float)
            joint_axis/=np.linalg.norm(joint_axis)
            model_axis=_rotation(records[element["body_a"]])@_vector(element["axis_a_body"],"profile axis",True)
            if abs(float(model_axis@joint_axis))<1-1e-8:
                raise EquivalentCompileError("Geometric detent coordinate must follow the actual prismatic axis")
            _profile_from_source(element,records,bindings,design)
    if not set(references)<=model_sources:
        raise EquivalentCompileError("Every excluded elastic reference needs an explicit equivalent element")
    region_owners=set()
    support_pairs = {}
    for element in value["elements"]:
        if element["kind"] == "shaft_support":
            pair = frozenset((element["body_a"],element["body_b"]))
            if pair in support_pairs:
                raise EquivalentCompileError("A body pair may have only one complete shaft-support provider")
            support_pairs[pair] = element["id"]
    for region in value["flexible_contact_regions"]:
        region["source_refs"]=_bind_refs(region.get("source_refs"),design,bindings)
        owner=region["region_owner_body"];region_owners.add(owner)
        pair=set(region["body_pair"])
        support_id = support_pairs.get(frozenset(pair))
        if region.get("active_element_id") is not None and region["active_element_id"] != support_id:
            raise EquivalentCompileError("A conditional flexible region must bind its actual shaft-support pair")
        if support_id is not None:
            region["active_element_id"] = support_id
        if not any(owner in element["source_body_ids"] and pair=={element["body_a"],element["body_b"]} for element in value["elements"]):
            raise EquivalentCompileError("A local flexible contact region must belong to its explicit equivalent pair/source")
    if not model_sources<=set(references)|region_owners:
        raise EquivalentCompileError("Declare each elastic source as a reference body or as a retained local flexible region")
    validate_equivalent_elasticity(value,records,scene.get("joints",[]))
    if any(_hash(path)!=expected for path,expected in bindings.items()):
        raise EquivalentCompileError("Source changed during equivalent-model compilation")
    model_binding={"schema_version":"slim_equivalent_model_binding_v1",
        "source_declaration_sha256":_value_hash(declaration),"compiled_declaration_sha256":_value_hash(value),
        "scene_sha256":geometry["scene_sha256"],"compiler_implementation_sha256":_hash(__file__),
        "parameter_scope":"Nominal, explicitly uncalibrated source-bound equivalent parameters; not elastic continuum or fabrication acceptance"}
    scope=make_validation_scope(geometry["bodies"],elastic_reference_body_ids=references,
        flexible_contact_regions=value["flexible_contact_regions"],source_bindings=bindings,model_binding=model_binding)
    return {"declaration":value,"validation_scope":scope,"source_bindings":bindings,"model_binding":model_binding}
