"""Validate delivered physical data without a mechanism or task catalog."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any
import numpy as np

from .measurements import validate_expression


class DeliveryError(ValueError):
    pass


def _finite(value, label, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise DeliveryError(f"{label} must be finite")
    if positive and value <= 0:
        raise DeliveryError(f"{label} must be positive")
    return float(value)


def _vector(value, label):
    if not isinstance(value, list) or len(value) != 3:
        raise DeliveryError(f"{label} must contain three numbers")
    return [_finite(x, label) for x in value]


def segment_distance(a0, a1, b0, b1):
    """Distance of two closed 3-D segments, including point regions."""
    p, q, u, v = map(np.asarray, (a0, b0, np.subtract(a1, a0), np.subtract(b1, b0)))
    w = p-q
    aa, bb, cc, dd, ee = np.dot(u,u), np.dot(u,v), np.dot(v,v), np.dot(u,w), np.dot(v,w)
    candidates = []
    if aa > 1e-20 and cc > 1e-20 and aa*cc-bb*bb > 1e-20:
        s, t = (bb*ee-cc*dd)/(aa*cc-bb*bb), (aa*ee-bb*dd)/(aa*cc-bb*bb)
        if 0 <= s <= 1 and 0 <= t <= 1:
            candidates.append(np.linalg.norm(w+s*u-t*v))
    for s in (0., 1.):
        t = np.clip((ee+s*bb)/cc,0,1) if cc > 1e-20 else 0
        candidates.append(np.linalg.norm(w+s*u-t*v))
    for t in (0., 1.):
        s = np.clip((t*bb-dd)/aa,0,1) if aa > 1e-20 else 0
        candidates.append(np.linalg.norm(w+s*u-t*v))
    return float(min(candidates))


def _read_scene(path: Path, *, experiment=None) -> dict[str, Any]:
    try:
        scene = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        if experiment is not None:
            from .frozen_experiment import bind_scene
            try: scene = bind_scene(scene, experiment)
            except ValueError as exc: raise DeliveryError(str(exc)) from exc
    except FileNotFoundError as exc:
        raise DeliveryError("scene.json has not been delivered") from exc
    except json.JSONDecodeError as exc:
        raise DeliveryError(f"scene.json is incomplete or malformed: {exc.msg}") from exc
    return scene


def _validate_model_structure(scene, *, missing_region_joint_ids=()):
    if not isinstance(scene,dict): raise DeliveryError("scene.json must contain a scene object")
    if scene.get("schema_version") != "slim_scene_v1" or scene.get("units") not in {"m","mm"}:
        raise DeliveryError("Expected slim_scene_v1 and units m or mm")
    bodies = scene.get("bodies")
    if not isinstance(bodies, list) or not bodies:
        raise DeliveryError("Deliver at least one physical body")
    ids = [x.get("id") for x in bodies]
    if any(not isinstance(x,str) or not x for x in ids) or len(set(ids)) != len(ids):
        raise DeliveryError("Body IDs must be unique nonempty strings")
    if not any(not b.get("fixed",False) for b in bodies):
        raise DeliveryError("A mechanical test needs at least one dynamic body")
    for body in bodies:
        if not isinstance(body.get("source"),str):
            raise DeliveryError(f"Body {body['id']} needs source geometry")
        if "fixed" in body and not isinstance(body["fixed"],bool):
            raise DeliveryError("fixed must be a boolean")
    joints = scene.get("joints", [])
    joint_ids = [j.get("id") for j in joints]
    if any(not isinstance(x,str) or not x for x in joint_ids) or len(set(joint_ids)) != len(joint_ids):
        raise DeliveryError("Joint IDs must be unique nonempty strings")
    for joint in joints:
        if joint.get("type") not in {"revolute","prismatic","fixed","spherical"}:
            raise DeliveryError(f"CAPABILITY_GAP: native behavior {joint.get('type')} needs a Harness implementation")
        if joint.get("body_a") not in ids or joint.get("body_b") not in ids or joint["body_a"] == joint["body_b"]:
            raise DeliveryError(f"Joint {joint['id']} needs two distinct declared bodies")
        _vector(joint.get("anchor"), "joint anchor")
        axis = _vector(joint.get("axis",[0,0,1]), "joint axis")
        if np.linalg.norm(axis) < 1e-9:
            raise DeliveryError("Joint axis must be nonzero")
        if _finite(joint.get("damping",0),"passive joint damping") < 0:
            raise DeliveryError("Passive damping cannot supply energy")
        for side in ("a","b"):
            if f"region_{side}" not in joint and joint["id"] in missing_region_joint_ids:
                continue  # 仅显式声明模型诊断：未知支承，不编造材料区域。
            region = joint.get(f"region_{side}",{})
            if not isinstance(region, dict):
                raise DeliveryError(f"Joint {joint['id']} region_{side} must be an object")
            points = region.get("points")
            if not isinstance(points,list) or len(points) != 2:
                raise DeliveryError(f"Joint {joint['id']} needs two endpoints for region_{side}")
            for point in points:
                _vector(point,"connection region point")
            _finite(region.get("radius"),"connection region radius",True)


def _validate_source_tests(scene):
    # 正式Harness的源自测要求保持不变，不用于benchmark模型装载。
    bodies = scene["bodies"]
    ids = [b["id"] for b in bodies]
    joints = scene.get("joints", [])
    joint_ids = [j["id"] for j in joints]
    tests = scene.get("tests")
    if not isinstance(tests,list) or not tests:
        raise DeliveryError("Propose at least one task-specific physical test")
    test_ids = [t.get("id") for t in tests]
    if any(not isinstance(x,str) or not x for x in test_ids) or len(set(test_ids)) != len(test_ids):
        raise DeliveryError("Test IDs must be unique nonempty strings")
    for test in tests:
        duration = _finite(test.get("duration_s"),"test duration",True)
        step = _finite(test.get("step_s",.001),"solver step",True)
        if step > duration:
            raise DeliveryError("Solver step exceeds test duration")
        if not test.get("purpose") or not test.get("assertions"):
            raise DeliveryError("Each test needs a purpose and measurable assertions")
        for drive in test.get("inputs",[]):
            if drive.get("joint") not in joint_ids or drive.get("mode") not in {"speed","position","effort"}:
                raise DeliveryError("Input needs a declared joint and physical drive mode")
            joint = joints[joint_ids.index(drive["joint"])]
            if joint["type"] not in {"revolute","prismatic"}:
                raise DeliveryError("This native input helper requires a revolute or prismatic joint")
            _finite(drive.get("max_effort"),"input effort bound",True)
            _finite(drive.get("value",0),"input value")
        for load in test.get("loads",[]):
            if load.get("body") not in ids:
                raise DeliveryError("External load names an unknown body")
            if "id" in load and (not isinstance(load["id"], str) or not load["id"]):
                raise DeliveryError("External load id must be a nonempty string")
            if "rotational_damping_nm_s_rad" in load:
                if _finite(load["rotational_damping_nm_s_rad"], "external rotational damping") < 0:
                    raise DeliveryError("External rotational damping must be passive")
                axis = _vector(load.get("axis_abs", [0,0,1]), "external damping axis")
                if np.linalg.norm(axis) < 1e-9: raise DeliveryError("External damping axis must be nonzero")
            for key in ("force_n","torque_nm","point"):
                if key in load: _vector(load[key],key)
        for assertion in test["assertions"]:
            if not assertion.get("id") or not assertion.get("purpose") or not isinstance(assertion.get("expression"),str):
                raise DeliveryError("Assertions need id, purpose, and expression")
            try:
                validate_expression(assertion["expression"])
            except (ValueError, SyntaxError) as exc:
                raise DeliveryError(
                    f"Invalid measurement expression in test {test['id']!r}, "
                    f"assertion {assertion['id']!r}: {exc}"
                ) from exc


def _validate_model_headers(scene):
    _vector(scene.get("gravity_m_s2",[0,-9.81,0]),"gravity")
    if "spur_gears" in scene:
        from .spur_gears import validate_spur_gears
        try: validate_spur_gears(scene["spur_gears"])
        except (KeyError, TypeError, ValueError) as exc: raise DeliveryError(str(exc)) from exc
    if "equivalent_elasticity" in scene:
        from .equivalent_elastic_compiler import validate_equivalent_header
        try: validate_equivalent_header(scene["equivalent_elasticity"])
        except (KeyError,TypeError,ValueError) as exc: raise DeliveryError(str(exc)) from exc


def load_declared_model_scene(path: Path, *, declared_ideal_joint_ids) -> dict[str, Any]:
    """独立诊断读取：只允许明确已有理想副缺region，物理字段校验完全共用。

    已提供的坏region仍拒绝；正式load_model_scene/load_scene不使用此例外。
    """
    scene = _read_scene(path)
    _validate_model_structure(scene, missing_region_joint_ids=set(declared_ideal_joint_ids))
    _validate_model_headers(scene)
    return scene


def load_model_scene(path: Path) -> dict[str, Any]:
    """校验完整物理模型，原样保留但不验证/执行源自测。"""
    scene = _read_scene(path)
    _validate_model_structure(scene)
    _validate_model_headers(scene)
    return scene


def load_scene(path: Path, *, experiment=None) -> dict[str, Any]:
    """正式Harness完整入口：保留模型、源测试、扩展头的原校验顺序。"""
    scene = _read_scene(path, experiment=experiment)
    _validate_model_structure(scene)
    _validate_source_tests(scene)
    _validate_model_headers(scene)
    return scene


def compile_scene(scene: dict, geometry: dict) -> dict:
    from .geometry import geometry_distance_to_point
    from .source_support import check_supports
    value = copy.deepcopy(scene)
    try: value["source_support_checks"]=check_supports(scene,geometry)
    except (ValueError,KeyError,TypeError) as exc: raise DeliveryError(f"Source support: {exc}") from exc
    value.pop("validation_scope",None)  # This is a program-owned compiled result, not author input.
    scale = .001 if scene["units"] == "mm" else 1.
    records = {b["id"]: b for b in geometry["bodies"]}
    value["bodies"] = geometry["bodies"]
    value["units"] = "m"
    for joint in value.get("joints",[]):
        joint["anchor"] = (np.asarray(joint["anchor"])*scale).tolist()
        axis = np.asarray(joint.get("axis",[0,0,1]),dtype=float)
        axis = axis/np.linalg.norm(axis)
        joint["axis"] = axis.tolist()
        for side in ("a","b"):
            region = joint[f"region_{side}"]
            region["radius"] *= scale
            world = np.asarray(region["points"],dtype=float)*scale
            body = records[joint[f"body_{side}"]]
            anchor = np.asarray(joint["anchor"])
            if joint["type"] in {"revolute","prismatic"}:
                # Position along the native axis is arbitrary, but the axis
                # itself must pass through the real local connection region.
                extent = max(np.linalg.norm(world-anchor,axis=1))+region["radius"]
                axis_ends = [anchor-extent*axis,anchor+extent*axis]
                frame_gap = segment_distance(*axis_ends,*world)
            else:
                frame_gap = segment_distance(anchor,anchor,*world)
            if frame_gap > region["radius"] + .00005:
                raise DeliveryError(f"Joint {joint['id']} native constraint frame misses region_{side}")
            distances = [geometry_distance_to_point(body,p.tolist()) for p in (*world,world.mean(axis=0))]
            if max(distances) > region["radius"] + .00005:
                raise DeliveryError(f"Joint {joint['id']} region_{side} is not near real source material: {distances} m")
            region["points"] = (world-np.asarray(body["position_m"])).tolist()
            region["source_distance_m"] = distances
        a,b = (joint[f"region_{s}"] for s in ("a","b"))
        pa,pb = (np.asarray(records[joint[f"body_{s}"]]["position_m"]) for s in ("a","b"))
        gap = segment_distance(*(np.asarray(a["points"])+pa),*(np.asarray(b["points"])+pb))-a["radius"]-b["radius"]
        if gap > .00005:
            raise DeliveryError(f"Joint {joint['id']} starts outside its local connection range: gap {gap} m")
    for test in value["tests"]:
        for load in test.get("loads",[]):
            if "point" in load: load["point"] = (np.asarray(load["point"])*scale).tolist()
    if scene.get("spur_gears"):
        from .spur_gears import compile_spur_gears
        try: value["spur_gears"] = compile_spur_gears(scene, geometry)
        except (KeyError, TypeError, ValueError) as exc: raise DeliveryError(str(exc)) from exc
    if scene.get("equivalent_elasticity") is not None:
        from .equivalent_elastic_compiler import compile_equivalent_elasticity
        try: equivalent=compile_equivalent_elasticity(scene,geometry)
        except (KeyError,TypeError,ValueError) as exc: raise DeliveryError(str(exc)) from exc
        value["equivalent_elasticity"]=equivalent["declaration"]
        value["validation_scope"]=equivalent["validation_scope"]
    return value
