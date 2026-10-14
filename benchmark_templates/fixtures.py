"""Benchmark 自己提供的试验载荷；不补修被测机械的内部结构。"""

from dataclasses import dataclass
import math
from .native import Signal


@dataclass
class HangingCableLoad:
    body: object
    drum_angle: Signal
    drum_speed: Signal
    radius_m: float
    initial_height_m: float
    tension_n: float = 0.0


def add_hanging_cable_load(
    model,
    *,
    drum_id,
    frame_id,
    axis,
    radius_m,
    mass_kg,
    initial_center,
    stiffness_n_m=30000.0,
    damping_ratio=0.2,
):
    """原卷筒接入真实负载刚体和有张力、可松弛的等效绳；目前用于Z向上的台架。

    绳的伸长由实际卷筒角和负载位置共同决定。重量会通过绳张力反作用到原卷筒；
    没有把负载位置直接设成 r*angle，也没有给卷筒加隐藏电机。绳弯曲、逐圈接触、
    打结和强度不在这个试验夹具模型内。预拉伸对应挂载后静止的初始张力。
    """
    from simulation_ready_cad_workflow.validation_pipeline.mechanism.pychrono_core import (
        load_pychrono_core,
    )

    ch = load_pychrono_core()
    g = model.system.GetGravitationalAcceleration()
    if abs(g.x) + abs(g.y) > 1e-9 or g.z >= 0:
        raise ValueError("本挂载夹具要求原场景Z向上、非零向下重力")
    if min(radius_m, mass_kg, stiffness_n_m) <= 0:
        raise ValueError("卷绕半径、负载质量和绳刚度必须为正")
    if not math.isfinite(damping_ratio) or damping_ratio < 0:
        raise ValueError("绳阻尼必须耗能，damping_ratio不能为负")
    b = model.objects["bodies"]
    drum = b[drum_id]
    frame = b[frame_id]
    if not frame.IsFixed():
        raise ValueError("当前挂载夹具只支持明确固定的参考台架")
    # 立方配重的质量是协议规定值；尺寸按名义钢密度确定，不用于材料强度结论。
    edge = (mass_kg / 7850.0) ** (1 / 3)
    load = ch.ChBody()
    load.SetName("benchmark_payload")
    load.SetMass(mass_kg)
    load.SetInertiaXX(ch.ChVector3d(*(mass_kg * edge * edge / 6,) * 3))
    load.SetPos(ch.ChVector3d(*initial_center))
    material = ch.ChContactMaterialNSC()
    material.SetFriction(0.3)
    box = ch.ChCollisionShapeBox(material, edge, edge, edge)
    load.AddCollisionShape(box)
    load.EnableCollision(True)
    model.system.Add(load)
    guide = ch.ChLinkLockPrismatic()
    guide.Initialize(
        load, frame, ch.ChFramed(load.GetPos(), ch.ChQuaterniond(1, 0, 0, 0))
    )
    model.system.Add(guide)
    angle = Signal(drum, "body_angle", reference=frame, axis=axis)
    speed = Signal(drum, "body_angular_speed", reference=frame, axis=axis)
    initial_angle = angle.read()
    height = float(load.GetPos().z)
    fixture = HangingCableLoad(load, angle, speed, radius_m, height)
    load_acc = load.AddAccumulator()
    drum_acc = drum.AddAccumulator()
    frame_acc = frame.AddAccumulator()
    preload = mass_kg * (-g.z) / stiffness_n_m
    damping = 2 * damping_ratio * math.sqrt(stiffness_n_m * mass_kg)
    old = model.before_step

    def apply_tension(tension):
        load.EmptyAccumulator(load_acc)
        drum.EmptyAccumulator(drum_acc)
        frame.EmptyAccumulator(frame_acc)
        force = ch.ChVector3d(0, 0, tension)
        direction = frame.GetRot().Rotate(ch.ChVector3d(*axis))
        direction = direction / direction.Length()
        load.AccumulateForce(load_acc, force, load.GetPos(), False)
        drum.AccumulateTorque(drum_acc, direction * (-radius_m * tension), False)
        # 固定试架承担导向传动的反力/反扭，不假装算出了实际绳绕卷筒的局部轴承载荷。
        frame.AccumulateForce(frame_acc, -force, load.GetPos(), False)
        frame.AccumulateTorque(frame_acc, direction * (radius_m * tension), False)
        fixture.tension_n = tension

    def before(t, dt):
        if old:
            old(t, dt)
        extension = (
            preload
            + radius_m * (angle.read() - initial_angle)
            + height
            - load.GetPos().z
        )
        rate = radius_m * speed.read() - load.GetPosDt().z
        tension = (
            max(0.0, stiffness_n_m * extension + damping * rate)
            if extension > 0
            else 0.0
        )
        apply_tension(tension)

    model.before_step = before
    apply_tension(stiffness_n_m * preload)
    # 复制暴露给记录器的字典，不修改原装载器闭包中的机械刚体/累加器列表。
    model.objects["bodies"] = {**b, "benchmark_payload": load}
    model.objects["fixture_declarations"] = [
        dict(
            kind="hanging_mass_with_finite_tension_only_cable",
            mass_kg=mass_kg,
            cube_edge_m=edge,
            radius_m=radius_m,
            stiffness_n_m=stiffness_n_m,
            damping_ns_m=damping,
            preload_extension_m=preload,
            source_drum_id=drum_id,
            source_unchanged=True,
            guide="ideal vertical prismatic guide on fixed benchmark frame",
            reaction_model="generalized cable reaction carried by fixed frame; not local drum-bearing rope-force fidelity",
            scope="external test fixture; no rope-strand/contact/strength certification",
        )
    ]
    model.keepalive = (model.keepalive, load, guide, box, material, fixture)
    model.scope += "; external guided benchmark mass with finite tension-only generalized cable; fixed frame takes reaction; original winch source unchanged; no free-swing/rope-strand/drum-bearing-force or load-rating certification"
    return fixture


def prepared_with_source_fixture_joint(
    prepared_path, original_scene_path, joint_id, *, purpose
):
    """恢复原源明确声明的环境移动副；不把它冒充实体CAD导轨。

    调用者须先独立确认该副属于测试环境防摆/导向，并在purpose写清来源。
    几何准备视图只移除这个环境副；正式scene/compile的实体连接校验不变。
    这里只还原原轴、初态和阻尼，后续仍由现有build_native创建和推进。
    """
    import copy
    import json
    from pathlib import Path
    from simulation_ready_cad_workflow.validation_pipeline.slim_harness.contracts import (
        load_model_scene,
    )

    if not isinstance(purpose, str) or not purpose.strip():
        raise ValueError("必须说明原源环境约束的用途与证据；不能自动把失败机械副当环境")
    prepared_file = Path(prepared_path).resolve(strict=True)
    original_file = Path(original_scene_path).resolve(strict=True)
    prepared = json.loads(prepared_file.read_text(encoding="utf-8"))
    original = load_model_scene(original_file)
    candidates = [x for x in original.get("joints", []) if x["id"] == joint_id]
    if len(candidates) != 1:
        raise ValueError("原scene必须唯一声明所指定环境副")
    source_spec = candidates[0]
    if source_spec["type"] != "prismatic":
        raise ValueError("本接口只恢复已明确指定的源环境prismatic，不接管机械轴承")
    if any(x["id"] == joint_id for x in prepared.get("joints", [])):
        raise ValueError("准备视图仍含此副，不能重复添加环境约束")
    records = {x["id"]: x for x in prepared["bodies"]}
    if any(source_spec["body_" + side] not in records for side in ("a", "b")):
        raise ValueError("原环境副所引用的真实刚体不在prepared中")
    scale = 0.001 if original["units"] == "mm" else 1.0
    spec = copy.deepcopy(source_spec)
    spec["anchor"] = [float(x) * scale for x in source_spec["anchor"]]
    length = math.sqrt(sum(float(x) ** 2 for x in source_spec["axis"]))
    spec["axis"] = [float(x) / length for x in source_spec["axis"]]
    for side in ("a", "b"):
        region = spec["region_" + side]
        position = records[spec["body_" + side]]["position_m"]
        region["points"] = [
            [float(x) * scale - float(position[k]) for k, x in enumerate(point)]
            for point in source_spec["region_" + side]["points"]
        ]
        region["radius"] = float(source_spec["region_" + side]["radius"]) * scale
    # 原理想环境副没有可脱开的CAD连接区，不能让运行器按材料间距自动删副。
    spec["source_declared_exact"] = True
    prepared.setdefault("joints", []).append(spec)
    record = {
        "original_scene_path": str(original_file),
        "prepared_geometry_view": str(prepared_file),
        "joint_id": joint_id,
        "purpose": purpose,
        "original_spec": copy.deepcopy(source_spec),
        "restored_spec": copy.deepcopy(spec),
        "added_marker": {"source_declared_exact": True},
        "scope": "original declared ideal environment guide restored; no CAD guide or physical retention evidence; original source file unchanged",
    }
    prepared.setdefault("cad_loader", {}).setdefault(
        "source_fixture_joints", []
    ).append(record)
    return prepared
