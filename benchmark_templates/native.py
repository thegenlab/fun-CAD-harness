# 公共执行器：所有设计共用这一份，judge 不必重写。
# 阅读顺序：先看 NativeModel（适配器交什么），再看 Operation（怎么操作），
# 最后看 run（每一步怎么跑）。main 是命令行入口。
# 它负责实际推进和记录；当前只做试跑，不给完整 Benchmark 判 PASS。
"""Experimental common PyChrono runner. Adapters bind source objects and laws.

No benchmark PASS is issued. Fixed protocol, stepping and native observations
belong here; model-specific force updates remain in reviewed source adapters.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
import argparse
import csv
import importlib.util
import json
import math
import sys
import time


def canonical_time(value):
    """统一协议时刻到皮秒精度，仅消除0.3累加的浮点差，不是物理通过容差。"""
    if isinstance(value, (int, float)) and math.isfinite(value):
        return round(value, 12)
    return value


# 按文件路径导入原工程/适配器，返回模块，随后可调用其中的类和函数。
# 注意：导入会执行文件顶层代码；源文件应将自行运行测试的代码放在 main 保护内。
def load_source_module(path):
    path = Path(path).resolve(strict=True)  # 文件必须存在，转为绝对路径
    sys.path.insert(0, str(path.parent))  # 让原工程能找到旁边的 mechanism.py 等依赖
    name = f"source_{path.stem}_{len(sys.modules)}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)  # 真正执行导入；不会自动调用其 main()
    return module


@dataclass
class Signal:
    # 一个“观测通道”：指定从哪个原生物理对象读哪种量。
    # dataclass 自动生成初始化方法，所以可以写 Signal(某根轴, 'shaft_angle')。
    obj: Any  # PyChrono 对象，不是零件名字符串
    kind: str  # 角度/位置/速度；由下面 read() 选择读取方法
    sign: float = 1.0  # 只允许翻转方向，不能放大运动来让机械显得合格
    reference: Any = None  # 可选的原生参考刚体；不默认把它固定
    axis: tuple = (0.0, 0.0, 1.0)
    _last_angle: float | None = field(default=None, init=False)
    _unwrapped: float = field(default=0.0, init=False)

    def read(self):
        # 从求解器里的真实对象读取；judge 不提供自己算出的“通过结果”。
        if self.kind in ("shaft_angle", "shaft_position"):
            # 原生 ChShaft 的 GetPos() 是广义坐标：通常是 rad；
            # 历史重物组件把它用于位移坐标，此时量纲由原模型定义为 m。
            value = self.obj.GetPos()
        elif self.kind == "shaft_speed":
            value = self.obj.GetPosDt()  # 对应坐标对时间的导数
        elif self.kind == "source_scalar":
            # 少量特殊观测钩子；必须读取已审源状态，不能返回期望运动或通过标签。
            value = self.obj()
        elif self.kind in ("body_position", "body_speed"):
            vector = (
                self.obj.GetPos()
                if self.kind == "body_position"
                else self.obj.GetPosDt()
            )
            if self.reference is not None:
                if self.kind == "body_position":
                    vector = vector - self.reference.GetPos()
                else:
                    offset = self.obj.GetPos() - self.reference.GetPos()
                    vector = (
                        vector
                        - self.reference.GetPosDt()
                        - self.reference.GetAngVelParent().Cross(offset)
                    )
                vector = self.reference.GetRot().RotateBack(vector)
            n = math.sqrt(sum(x * x for x in self.axis))
            value = (
                sum(x * y for x, y in zip((vector.x, vector.y, vector.z), self.axis))
                / n
            )
        elif self.kind in ("body_angle", "body_angular_speed"):
            n = math.sqrt(sum(x * x for x in self.axis))
            axis = tuple(x / n for x in self.axis)
            if self.kind == "body_angular_speed":
                vector = self.obj.GetAngVelParent()
                if self.reference is not None:
                    vector = self.reference.GetRot().RotateBack(
                        vector - self.reference.GetAngVelParent()
                    )
                value = sum(x * y for x, y in zip((vector.x, vector.y, vector.z), axis))
            else:
                q = self.obj.GetRot()
                if self.reference is not None:
                    q = self.reference.GetRot().GetConjugate() * q
                # 固定轴运动的连续扭转角；不是通用3D姿态误差。
                raw = 2 * math.atan2(
                    q.e1 * axis[0] + q.e2 * axis[1] + q.e3 * axis[2], q.e0
                )
                if self._last_angle is None:
                    self._unwrapped = raw
                else:
                    self._unwrapped += (raw - self._last_angle + math.pi) % (
                        2 * math.pi
                    ) - math.pi
                self._last_angle = raw
                value = self._unwrapped
        else:
            raise ValueError(f"Unsupported signal kind: {self.kind}")
        if self.sign not in (-1, 1):
            raise ValueError(
                "Signal sign must be +1 or -1; arbitrary rescaling is not allowed"
            )
        value = float(value) * self.sign
        if not math.isfinite(value):  # NaN/无穷大意味着数值结果不能正常使用
            raise RuntimeError("Nonfinite native signal")
        return value


@dataclass
class Actuator:
    # 一个“施力入口”：源轴载荷、刚体施力或刚体力矩；不强设输出运动。
    obj: Any  # 被正常操作的原生轴或刚体
    kind: str  # shaft / body_force / body_torque
    reference: Any = None
    axis: tuple = (0.0, 0.0, 1.0)
    point_local: tuple = (0.0, 0.0, 0.0)
    coordinate_offset: float = 0.0  # 源关节以装配态为零时，显式扣除该已记录初值
    react_on_reference: bool = (
        True  # 内部成对操作为True；外部人力只用参考系测量时为False
    )
    # 记住公共执行器上一周期加了多少力；init=False 表示 judge 不填这个运行状态。
    _last_input: float = field(default=0.0, init=False)
    _accumulator: Any = field(default=None, init=False)
    _reaction_accumulator: Any = field(default=None, init=False)
    _position_signal: Any = field(default=None, init=False)

    def coordinate(self):
        # 控制器需要当前实际位置，判断离目标还差多少、是否到达行程终点。
        if self.kind == "body_force":
            from simulation_ready_cad_workflow.validation_pipeline.mechanism.pychrono_core import (
                load_pychrono_core,
            )

            c = load_pychrono_core()
            point = self.obj.TransformPointLocalToParent(
                c.ChVector3d(*self.point_local)
            )
            if self.reference is not None:
                point = self.reference.TransformPointParentToLocal(point)
            norm = math.sqrt(sum(x * x for x in self.axis))
            return (
                sum(x * y for x, y in zip((point.x, point.y, point.z), self.axis))
                / norm
                - self.coordinate_offset
            )
        if self.kind == "body_torque":
            if self._position_signal is None:
                self._position_signal = Signal(
                    self.obj,
                    "body_position" if self.kind == "body_force" else "body_angle",
                    reference=self.reference,
                    axis=self.axis,
                )
            return self._position_signal.read() - self.coordinate_offset
        if self.kind != "shaft":
            raise ValueError(f"Unsupported actuator: {self.kind}")
        return float(self.obj.GetPos())

    def speed(self):
        if self.kind == "body_force":
            from simulation_ready_cad_workflow.validation_pipeline.mechanism.pychrono_core import (
                load_pychrono_core,
            )

            c = load_pychrono_core()
            point = self.obj.TransformPointLocalToParent(
                c.ChVector3d(*self.point_local)
            )
            velocity = self.obj.GetPosDt() + self.obj.GetAngVelParent().Cross(
                point - self.obj.GetPos()
            )
            if self.reference is not None:
                velocity -= (
                    self.reference.GetPosDt()
                    + self.reference.GetAngVelParent().Cross(
                        point - self.reference.GetPos()
                    )
                )
                velocity = self.reference.GetRot().RotateBack(velocity)
            norm = math.sqrt(sum(x * x for x in self.axis))
            return (
                sum(
                    x * y
                    for x, y in zip((velocity.x, velocity.y, velocity.z), self.axis)
                )
                / norm
            )
        if self.kind == "body_torque":
            return Signal(
                self.obj,
                "body_speed" if self.kind == "body_force" else "body_angular_speed",
                reference=self.reference,
                axis=self.axis,
            ).read()
        return float(self.obj.GetPosDt())

    def clear_input(self):
        # 只撤销“我们上一周期加的人工输入”，保留原模型自身的载荷。
        # 之后原工程会重算内部力；不能把发条力和重力也一并清掉。
        if self.kind in ("body_force", "body_torque"):
            if self._accumulator is not None:
                self.obj.EmptyAccumulator(self._accumulator)
            if self._reaction_accumulator is not None:
                self.reference.EmptyAccumulator(self._reaction_accumulator)
        else:
            self.obj.SetAppliedLoad(float(self.obj.GetAppliedLoad()) - self._last_input)
        self._last_input = 0.0

    def add_input(self, effort):
        if not math.isfinite(effort):
            raise ValueError("Nonfinite input effort")
        # 在已有内部载荷上叠加本周期人工输入，而不是覆盖原模型载荷。
        if self.kind in ("body_force", "body_torque"):
            from simulation_ready_cad_workflow.validation_pipeline.mechanism.pychrono_core import (
                load_pychrono_core,
            )

            ch = load_pychrono_core()
            if self._accumulator is None:
                self._accumulator = self.obj.AddAccumulator()
            axis = ch.ChVector3d(*self.axis)
            axis = axis / axis.Length()
            if self.reference is not None:
                axis = self.reference.GetRot().Rotate(axis)
            vector = axis * effort
            if self.kind == "body_torque":
                self.obj.AccumulateTorque(self._accumulator, vector, False)
            else:
                point = self.obj.TransformPointLocalToParent(
                    ch.ChVector3d(*self.point_local)
                )
                self.obj.AccumulateForce(self._accumulator, vector, point, False)
            if self.reference is not None and self.react_on_reference:
                if self._reaction_accumulator is None:
                    self._reaction_accumulator = self.reference.AddAccumulator()
                if self.kind == "body_torque":
                    self.reference.AccumulateTorque(
                        self._reaction_accumulator, -vector, False
                    )
                else:
                    self.reference.AccumulateForce(
                        self._reaction_accumulator, -vector, point, False
                    )
        else:
            self.obj.SetAppliedLoad(float(self.obj.GetAppliedLoad()) + effort)
        self._last_input += effort


@dataclass
class Operation:
    # 一次有限操作：对谁施力、何时开始/结束、怎么控制、力度多大。
    # 对转轴而言：位置 rad、速度 rad/s、力矩 N·m；时间均为秒。
    actuator: str  # 对应 NativeModel.actuators 中的键，如 going_wind
    start_s: float  # 开始时刻，包含该时刻
    end_s: float  # 结束时刻，从该时刻起不再施加输入
    mode: str  # effort=给力矩，speed=追踪速度，position=追踪位置
    value: float  # 对应模式的力矩/目标速度/目标位置
    max_effort: float  # 力度上限：即使目标没达到，也不无限加力
    gain: float = 0.0  # 偏差乘这个系数，得到纠偏力度
    damping: float = 0.0  # 速度阻尼：速度越大，扣除的力度越大
    min_effort: float | None = None  # None 允许正负对称；0 表示只往一个方向施力
    stop_at: float | None = None  # 可选行程终点；到达后本次操作不再启动
    stop_direction: int = 1  # +1 表示达到上界停止，-1 表示达到下界停止
    bias_effort: float = 0.0  # 源正常操作的前馈力；与反馈一起受同一总上限限制

    def __post_init__(self):
        self.start_s = canonical_time(self.start_s)
        self.end_s = canonical_time(self.end_s)

    def validate(self):
        # 开始仿真前检查操作参数：不能有空值、无穷大、反向时间窗等。
        values = [
            self.start_s,
            self.end_s,
            self.value,
            self.max_effort,
            self.gain,
            self.damping,
            self.bias_effort,
        ]
        values += [v for v in (self.min_effort, self.stop_at) if v is not None]
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
            raise ValueError("Operation values must be finite numbers")
        if not 0 <= self.start_s < self.end_s or self.max_effort <= 0:
            raise ValueError("Invalid operation interval or effort limit")
        if self.mode not in ("effort", "speed", "position", "resistance"):
            raise ValueError("Unsupported operation mode")
        if self.bias_effort and self.mode not in ("speed", "position"):
            raise ValueError(
                "Feedforward belongs to a bounded speed/position operation"
            )
        if self.gain < 0 or self.damping < 0 or self.stop_direction not in (-1, 1):
            raise ValueError("Invalid controller or stop direction")
        if (
            self.min_effort is not None
            and not -self.max_effort <= self.min_effort <= self.max_effort
        ):
            raise ValueError("min_effort outside effort limit")

    def effort(self, q, speed):
        # q、speed 都来自本步实际状态。这里只算应施加的力，不直接修改位置/速度。
        if self.mode == "resistance":
            raw = -abs(self.value) * (1 if speed > 0 else -1 if speed < 0 else 0)
        elif self.mode == "effort":
            raw = self.value - self.damping * speed
        elif self.mode == "speed":
            raw = self.gain * (self.value - speed) - self.damping * speed
        else:
            raw = self.gain * (self.value - q) - self.damping * speed
        raw += self.bias_effort
        # 将计算结果夹在允许范围内；这是“有限力度”的实现位置。
        low = -self.max_effort if self.min_effort is None else self.min_effort
        return min(self.max_effort, max(low, raw))


@dataclass
class NativeModel:
    # 适配器只需交回这个对象，公共执行器就知道该操作、测量什么。
    system: Any  # 原工程完整建立的原生 ChSystem
    signals: dict[str, Signal]  # 名称 → 观测通道；不是全部机械零件清单
    actuators: dict[str, Actuator]  # 名称 → 人工操作入口
    before_step: Callable | None = None  # 每步求解前，更新原工程内部力；不推进时间
    after_step: Callable | None = None  # 每步求解后，同步原工程缓存状态；不推进时间
    keepalive: Any = None  # 持有原模型，避免其 Python 对象过早被释放
    scope: str = ""  # 说明仿真范围，例如“仅走时组件”
    initial_state: str = ""  # 说明初始储能/准备状态，避免误称为从零上弦
    commit_inputs: Callable | None = None  # 所有人工输入加入后刷新源力缓存，不推进时间
    task_evidence_gaps: dict[str, str] = field(
        default_factory=dict
    )  # 原功能缺少的物理证据；有数值也不得升级整项通过
    # 每步可记录多条真实接触/拨动事件；标量曲线不足以保存同时发声的部件身份。
    observation_events: list[dict] = field(default_factory=list)
    _default_update_api: str | None = field(default=None, init=False)


def run(
    model,
    operations,
    *,
    duration_s,
    step_s,
    output,
    preparation_end_s=None,
    required_signals=(),
    sample_times=(),
):
    duration_s = canonical_time(duration_s)
    sample_times = tuple(canonical_time(t) for t in sample_times)
    # 1. 核对本次任务实际声明的必需输入与观测，不写死某种机械。
    if not all(math.isfinite(x) and x > 0 for x in (duration_s, step_s)):
        raise ValueError("Positive finite duration/step required")
    if step_s > duration_s:
        raise ValueError("Step exceeds duration")
    for required in required_signals:
        if required not in model.signals:
            raise ValueError(f"Missing required observable: {required}")
    # 未绑定秒针只是不记录它，不会从原机械模型中删除任何部件。
    for op in operations:
        op.validate()
        if op.actuator not in model.actuators:
            raise ValueError(f"Missing actuator: {op.actuator}")
    for i, a in enumerate(operations):
        # 同一入口不能同时收到两条互相冲突的操作；不同入口可以并行操作。
        for b in operations[i + 1 :]:
            if a.actuator == b.actuator and max(a.start_s, b.start_s) < min(
                a.end_s, b.end_s
            ):
                raise ValueError(
                    "Overlapping commands on one actuator require a reviewed combined operation"
                )
    if abs(float(model.system.GetChTime())) > 1e-12:
        # 仿真时钟从 0 开始；这不要求模型储能为 0，预加载应记在 initial_state。
        raise ValueError("Expected a freshly constructed system at t=0")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)  # 使用新目录，避免覆盖旧试验
    from .passive_resistance import ResistanceLoads
    resistance = ResistanceLoads(model, operations)
    # 默认将最后一条准备操作的结束时刻视为“全部松手”。
    prep_end = (
        max((op.end_s for op in operations), default=0.0)
        if preparation_end_s is None
        else preparation_end_s
    )
    # 2. 先保存本次配置。protocol 是输出记录，不是要求用户填写的新输入格式。
    protocol = {
        "duration_s": duration_s,
        "step_s": step_s,
        "preparation_end_s": prep_end,
        "operations": [vars(op) for op in operations],
        "scope": model.scope,
        "initial_state": model.initial_state,
        "functional_verdict": None,
        "input_columns": {
            "input.<port>": "requested generalized effort after operation limit",
            "input_actual.<port>": "Port effort after source mapping; implicit resistance uses the qualified accepted-state evaluation described in passive_resistance",
            "input_average_power.<port>": "For implicit resistance ports: accepted-state constitutive torque times step-average angular velocity",
        },
        "passive_resistance": resistance.contract,
    }
    (output / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    started = time.perf_counter()  # 记录真实耗时，区别于仿真里的秒数
    initial = {name: signal.read() for name, signal in model.signals.items()}
    last = initial.copy()
    release_state = initial.copy() if prep_end == 0 else None  # 松手瞬间的读数
    stopped, events = set(), []  # 已停止的操作编号，以及停止等事件
    steps, now = 0, 0.0
    # 步进不能跨过操作开始/结束时刻，必要时最后一步缩短。
    boundaries = sorted(
        {
            duration_s,
            *[t for t in sample_times if 0 < t < duration_s],
            *[
                t
                for op in operations
                for t in (op.start_s, op.end_s)
                if 0 < t < duration_s
            ],
        }
    )
    status, error = "running", None
    # 审核用原生刚体位姿：每20ms及试验边界保存，便于从原CAD重放。
    # 这只是采样可视证据，不冒充逐步原BRep穿透审计。
    pose_bodies = getattr(model, "objects", {}).get("bodies", {})
    pose_stream = (
        (output / "body_poses.jsonl").open("w", encoding="utf-8")
        if pose_bodies
        else None
    )
    pose_next = 0.0

    def record_poses(t):
        if pose_stream is None:
            return
        values = {}
        for name, body in pose_bodies.items():
            pos, rot = body.GetPos(), body.GetRot()
            values[name] = {
                "position_m": [pos.x, pos.y, pos.z],
                "rotation_wxyz": [rot.e0, rot.e1, rot.e2, rot.e3],
            }
        sample = {"time_s": t, "bodies": values}
        # Preserve actual conditional support activity for the existing saved-pose
        # source audit. These are source-model observations, not inferred states.
        equivalent = getattr(model, "objects", {}).get("equivalent")
        if equivalent is not None:
            sample["equivalent_elasticity"] = equivalent.sample()
        pose_stream.write(json.dumps(sample) + "\n")
        pose_stream.flush()

    record_poses(0.0)
    pose_next = 0.02
    try:
        # 3. 创建轨迹表：实际时间、阶段、真实观测，以及每个入口的人工输入。
        with (output / "motion.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "time_s",
                    "native_time_s",
                    "phase",
                    *model.signals,
                    *["input." + name for name in model.actuators],
                    *["input_actual." + name for name in model.actuators],
                    *["input_average_power." + name for name in resistance.loads],
                ]
            )
            writer.writerow(
                [
                    0.0,
                    0.0,
                    "initial",
                    *initial.values(),
                    *[0.0 for _ in model.actuators],
                    *[0.0 for _ in model.actuators],
                    *[0.0 for _ in resistance.loads],
                ]
            )
            while now < duration_s - 1e-12:
                # 这些很小的时间容差只处理浮点舍入，不是机械功能的通过容差。
                boundary = next(t for t in boundaries if t > now + 1e-12)
                dt = min(step_s, boundary - now)
                for actuator in model.actuators.values():
                    actuator.clear_input()
                resistance.clear()
                if model.before_step:
                    # 顺序很重要：撤销旧人工输入 → 更新内部力 → 加新人工输入。
                    model.before_step(now, dt)
                if abs(float(model.system.GetChTime()) - now) > 1e-10:
                    raise RuntimeError(
                        "Source before_step advanced physics; common runner owns stepping"
                    )
                applied = {name: 0.0 for name in model.actuators}
                for index, op in enumerate(operations):
                    # 未到时间、已经结束、已达到行程终点的操作，都不再施力。
                    if not op.start_s <= now < op.end_s - 1e-12 or index in stopped:
                        continue
                    actuator = model.actuators[op.actuator]
                    q, speed = actuator.coordinate(), actuator.speed()
                    if (
                        op.stop_at is not None
                        and op.stop_direction * (q - op.stop_at) >= 0
                    ):
                        # 记住“本次已停止”；之后即使轴稍微回弹，也不重新启动操作。
                        stopped.add(index)
                        events.append(
                            {
                                "time_s": now,
                                "event": "operation_travel_limit",
                                "operation": index,
                            }
                        )
                        continue
                    applied[op.actuator] = op.effort(q, speed)
                    if op.mode == "resistance" and op.actuator in resistance.loads:
                        resistance.prepare(op.actuator, op, dt)
                    else:
                        actuator.add_input(applied[op.actuator])
                # 原生刚体累加器需要在“当前步全部内力+外部操作”齐备后刷新。
                # 源模型可提供专门提交钩子；不能在加人工输入前就捕获最终力缓存。
                if model.commit_inputs:
                    model.commit_inputs(now, dt)
                elif any(
                    getattr(a, "kind", None) in ("body_force", "body_torque")
                    for a in model.actuators.values()
                ):
                    if model._default_update_api == "bool":
                        model.system.Update(True)
                    else:
                        try:
                            model.system.Update()
                            model._default_update_api = "default"
                        except TypeError as exc:
                            if "ChSystem_Update" not in str(exc):
                                raise
                            model.system.Update(True)
                            model._default_update_api = "bool"
                if abs(float(model.system.GetChTime()) - now) > 1e-10:
                    raise RuntimeError("Input commit advanced physics")
                result = model.system.DoStepDynamics(dt)
                # 这是整个公共执行器唯一推进物理时间的地方。
                # 核对求解器确实推进了 dt；程序运行过不等于物理计算成功。
                advanced = float(model.system.GetChTime())
                if (
                    result is False
                    or not math.isfinite(advanced)
                    or abs(advanced - now - dt) > max(1e-10, dt * 1e-6)
                ):
                    raise RuntimeError(
                        "Native solver failed to advance the requested step"
                    )
                resistance.accepted(model.actuators)
                actual_applied = {name: actuator._last_input for name, actuator in model.actuators.items()}
                if model.after_step:
                    model.after_step(advanced, dt)
                if abs(float(model.system.GetChTime()) - advanced) > 1e-10:
                    raise RuntimeError("Source after_step advanced physics")
                # 在真实求解到达边界后，给记录使用协议的精确边界标签；
                # native_time_s 另存原生浮点时刻，不改物理状态或求解时钟。
                now = boundary if abs(advanced - boundary) <= 1e-10 else advanced
                steps += 1
                last = {name: signal.read() for name, signal in model.signals.items()}
                if release_state is None and now >= prep_end - 1e-12:
                    release_state = last.copy()
                # 每行输入是“刚完成的这一小步”施加的量；观测是该步结束后的状态。
                writer.writerow(
                    [
                        now,
                        advanced,
                        "prepare" if now <= prep_end + 1e-12 else "released",
                        *last.values(),
                        *applied.values(),
                        *actual_applied.values(),
                        *resistance.power.values(),
                    ]
                )
                if now >= pose_next - 1e-12 or now in boundaries:
                    record_poses(now)
                    while pose_next <= now + 1e-12:
                        pose_next += 0.02
                if steps % 100 == 0:
                    stream.flush()  # 周期性落盘，长运行中也能查看已完成的部分
            status = "completed"
    except Exception as exc:
        # 保留失败原因并继续向调用者抛错，不能把异常吞掉当作成功。
        status, error = "execution_error", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if pose_stream is not None:
            pose_stream.close()
        # 4. 成功或失败都撤去本执行器的人工输入，并保存已完成的观测。
        for actuator in model.actuators.values():
            actuator.clear_input()
        resistance.clear()
        report = {
            **protocol,
            "status": status,
            "error": error,
            "actual_time_s": now,
            "steps": steps,
            "wall_s": time.perf_counter() - started,
            "initial": initial,
            "final": last,
            "release_state": release_state,
            # 每个通道分别做“末值减松手值”；速度通道得到速度差，不是位移。
            "delta_after_release": (
                None
                if release_state is None
                else {name: last[name] - release_state[name] for name in last}
            ),
            "events": events,
            "observation_events": list(getattr(model, "observation_events", ())),
        }
        (output / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return report


def main():
    # 命令行入口：指定适配器、原工程目录、步长和新输出目录。
    # --observe 默认观察 5 秒；--duration 仅用于显式指定更短的启动前缀。
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step", type=float, required=True)
    parser.add_argument(
        "--duration",
        type=float,
        help="Short startup prefix; does not replace the complete F01 trial",
    )
    parser.add_argument("--observe", type=float, default=5.0)
    args = parser.parse_args()
    # 允许适配器用 from common_runtime import ... 获取同一份公共定义。
    sys.modules.setdefault("common_runtime", sys.modules[__name__])
    adapter = load_source_module(args.adapter)
    model = adapter.build(
        args.source, args.step
    )  # judge 的接入部分：加载原模型、绑定对象
    operations = adapter.preparation()  # judge 填的有限准备操作列表
    prep_end = max((op.end_s for op in operations), default=0.0)
    duration = args.duration if args.duration is not None else prep_end + args.observe
    # 从这里开始，统一走公共流程；judge 不再自己写时间循环或记录逻辑。
    result = run(
        model,
        operations,
        duration_s=duration,
        step_s=args.step,
        output=args.output,
        preparation_end_s=prep_end,
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("status", "actual_time_s", "steps", "wall_s", "scope")
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    # 直接执行本文件才启动；被适配器 import 时不启动。
    main()
