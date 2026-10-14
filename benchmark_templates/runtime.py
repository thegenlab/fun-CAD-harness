"""公共任务接口：模板定动作和判据，独立绑定文件接入真实 PyChrono 模型。

这里不创建某一款机械，不让 judge 修改测试判据。所有家族调用同一个执行器。
结果只表示本轮声明任务的观测与判定，不是正式 Harness 认证。
"""

from dataclasses import dataclass
from pathlib import Path
import argparse
import ast
import csv
import json
import math
import sys
import time
from collections.abc import Mapping
from . import native
from simulation_ready_cad_workflow.validation_pipeline.slim_harness.measurements import (
    validate_expression,
    evaluate,
)

# 原有闹钟原型的物理对象接口继续用于真实模型；家族模板只需要下面六个类型。
NativeModel = native.NativeModel
Signal = native.Signal
Actuator = native.Actuator
load_source_module = native.load_source_module


@dataclass(frozen=True)
class Parameter:
    name: str
    description: str
    unit: str
    minimum: float | None = None
    maximum: float | None = None
    default: object = None
    choices: tuple | None = None
    owner: str = "protocol"

    def resolve(self, values):
        value = values.get(self.name, self.default)
        if value is None:
            raise ValueError(
                f"协议参数未填写：{self.name}（{self.description}，{self.unit}）"
            )
        if self.choices is not None:
            if value not in self.choices:
                raise ValueError(f"{self.name} 必须属于 {self.choices}")
            return value
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
        ):
            raise ValueError(f"{self.name} 必须为有限数值")
        if self.minimum is not None and value < self.minimum:
            raise ValueError(f"{self.name} 小于允许下限")
        if self.maximum is not None and value > self.maximum:
            raise ValueError(f"{self.name} 大于允许上限")
        return value


class ProtocolValues(Mapping):
    """只校验选定变体实际读取的参数，未选可选功能不索要额外字段。"""

    def __init__(self, parameters, values):
        self.parameters = {p.name: p for p in parameters}
        self.values = dict(values)
        self.used = {}

    def __getitem__(self, name):
        if name not in self.parameters:
            raise KeyError(name)
        if name not in self.used:
            self.used[name] = self.parameters[name].resolve(self.values)
        return self.used[name]

    def __iter__(self):
        return iter(self.parameters)

    def __len__(self):
        return len(self.parameters)


@dataclass(frozen=True)
class Action:
    # 一条正常操作，不是强制运动轨迹；实际力度始终有上限。
    port: str
    start_s: float
    end_s: float
    mode: str
    value: float
    max_effort: float
    gain: float = 0.0
    damping: float = 0.0
    min_effort: float | None = None
    stop_at: float | None = None
    stop_direction: int = 1
    bias_effort: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "start_s", native.canonical_time(self.start_s))
        object.__setattr__(self, "end_s", native.canonical_time(self.end_s))

    def operation(self):
        return native.Operation(
            self.port,
            self.start_s,
            self.end_s,
            self.mode,
            self.value,
            self.max_effort,
            self.gain,
            self.damping,
            self.min_effort,
            self.stop_at,
            self.stop_direction,
            self.bias_effort,
        )


@dataclass(frozen=True)
class Check:
    id: str
    task_ids: tuple[str, ...]
    expression: str
    description: str
    evidence: str = "physics"  # manual 为题面明确要求的实物检查；不伪造原生信号

    def __post_init__(self):
        if self.evidence != "physics":
            return
        # 只归一window/after/before的时间常量，力、角度、长度阈值一律不动。
        tree = ast.parse(self.expression, mode="eval")
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                positions = (
                    (1, 2)
                    if node.func.id == "window"
                    else (1,) if node.func.id in ("after", "before") else ()
                )
                for i in positions:
                    if i < len(node.args) and isinstance(node.args[i], ast.Constant):
                        node.args[i].value = native.canonical_time(node.args[i].value)
        object.__setattr__(self, "expression", ast.unparse(tree))

    def signals(self):
        if self.evidence == "manual":
            return set()
        if self.evidence != "physics":
            raise ValueError("Unknown evidence kind")
        tree = validate_expression(self.expression)
        return {
            node.args[0].value
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "signal"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        }

    def boundaries(self):
        # 自动识别 window/after/before 中的固定时刻，保证这些边界有真实求解样本。
        if self.evidence == "manual":
            return set()
        result = set()
        for node in ast.walk(validate_expression(self.expression)):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            args = (
                node.args[1:] if node.func.id in ("window", "after", "before") else ()
            )
            for arg in args:
                if isinstance(arg, ast.Constant) and isinstance(
                    arg.value, (int, float)
                ):
                    result.add(float(arg.value))
        return result


@dataclass(frozen=True)
class Trial:
    id: str
    duration_s: float
    actions: tuple[Action, ...]
    checks: tuple[Check, ...]
    description: str = ""
    gravity: tuple[float, float, float] | float | None = None
    required_signals: tuple[str, ...] = ()
    sample_times: tuple[float, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "duration_s", native.canonical_time(self.duration_s))
        object.__setattr__(
            self,
            "sample_times",
            tuple(native.canonical_time(t) for t in self.sample_times),
        )


@dataclass(frozen=True)
class Template:
    family_id: str
    source_task_ids: tuple[str, ...]
    required_signals: tuple[str, ...]
    trials: tuple[Trial, ...]
    description: str = ""
    optional_signals: tuple[str, ...] = ()
    variant: str = "default"

    def validate(self):
        if not self.trials or not self.source_task_ids:
            raise ValueError("模板必须有任务和可执行试验")
        ids = [trial.id for trial in self.trials]
        if len(ids) != len(set(ids)):
            raise ValueError("试验 ID 重复")
        covered = set()
        for trial in self.trials:
            if (
                not math.isfinite(trial.duration_s)
                or trial.duration_s <= 0
                or not trial.checks
            ):
                raise ValueError(f"{trial.id} 缺少正时长或判据")
            check_ids = [check.id for check in trial.checks]
            if len(check_ids) != len(set(check_ids)):
                raise ValueError(f"{trial.id} 判据 ID 重复")
            for action in trial.actions:
                action.operation().validate()
                if action.end_s > trial.duration_s + 1e-12:
                    raise ValueError(f"{trial.id} 的操作超过试验时长")
            for check in trial.checks:
                check.signals()
                if not check.task_ids or not set(check.task_ids) <= set(
                    self.source_task_ids
                ):
                    raise ValueError(f"{check.id} 没有对应原功能")
                covered.update(check.task_ids)
                for t in check.boundaries():
                    if not 0 <= t <= trial.duration_s:
                        raise ValueError(f"{check.id} 的观察窗口超过仿真时长")
            for t in trial.sample_times:
                if not math.isfinite(t) or not 0 <= t <= trial.duration_s:
                    raise ValueError("采样边界必须位于实际试验中")
        if covered != set(self.source_task_ids):
            raise ValueError(f"漏测原功能：{set(self.source_task_ids)-covered}")
        return self


def build_trial_model(binding, source_dir, step_s, protocol, trial):
    """原正常试验准备可依trial身份选择夹具；不向binding开放改写动作/判据。"""
    factory = getattr(binding, "build_model_for_trial", None)
    if factory is not None:
        return factory(Path(source_dir), step_s, protocol, trial.id)
    return binding.build_model(Path(source_dir), step_s, protocol)


def execute(
    template,
    binding,
    protocol,
    *,
    source_dir,
    output,
    step_s,
    trial_id=None,
    startup_seconds=None,
):
    """调用同一个物理执行器；启动前缀不计算完整功能通过。"""
    template.validate()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    selected = [t for t in template.trials if trial_id is None or t.id == trial_id]
    if not selected:
        raise ValueError("没有选中的试验")
    summary = {
        "family_id": template.family_id,
        "variant": template.variant,
        "source_task_ids": list(template.source_task_ids),
        "trials": [],
        "harness_certified": False,
        "started_at_unix": time.time(),
    }
    for trial in selected:
        destination = output / trial.id
        record = {
            "id": trial.id,
            "status": "loading",
            "checks": [],
            "functional_pass": None,
        }
        summary["trials"].append(record)
        try:
            # 每次试验重新加载完整原模型，不能复用上一试验已改变的状态。
            model = build_trial_model(binding, source_dir, step_s, protocol, trial)
            trial_tasks = {task for check in trial.checks for task in check.task_ids}
            gaps = {
                task: reason
                for task, reason in getattr(model, "task_evidence_gaps", {}).items()
                if task in trial_tasks
            }
            if gaps:
                record["task_evidence_gaps"] = gaps
            required = set(template.required_signals) | set(trial.required_signals)
            required |= {name for c in trial.checks for name in c.signals()}
            required -= {"time_s", "native_time_s"}
            missing = required - set(model.signals)
            if missing:
                raise ValueError(f"缺少原生观测绑定：{sorted(missing)}")
            if trial.gravity is not None:
                from simulation_ready_cad_workflow.validation_pipeline.mechanism.pychrono_core import (
                    load_pychrono_core,
                )

                ch = load_pychrono_core()
                if isinstance(trial.gravity, (int, float)):
                    magnitude = float(trial.gravity)
                    if not math.isfinite(magnitude) or magnitude < 0:
                        raise ValueError("Invalid gravity magnitude")
                    old = model.system.GetGravitationalAcceleration()
                    norm = math.sqrt(old.x**2 + old.y**2 + old.z**2)
                    if norm == 0 and magnitude > 0:
                        raise ValueError(
                            "Source gravity direction unavailable; explicit source gravity binding required"
                        )
                    vector = (
                        (0.0, 0.0, 0.0)
                        if magnitude == 0
                        else tuple(v * magnitude / norm for v in (old.x, old.y, old.z))
                    )
                else:
                    vector = trial.gravity
                model.system.SetGravitationalAcceleration(ch.ChVector3d(*vector))
            duration = (
                trial.duration_s
                if startup_seconds is None
                else min(trial.duration_s, startup_seconds)
            )
            boundaries = set(trial.sample_times) | {
                x for c in trial.checks for x in c.boundaries()
            }
            result = native.run(
                model,
                [a.operation() for a in trial.actions],
                duration_s=duration,
                step_s=step_s,
                output=destination,
                required_signals=tuple(required),
                sample_times=tuple(x for x in boundaries if x <= duration),
            )
            record.update(
                status=result["status"],
                actual_time_s=result["actual_time_s"],
                steps=result["steps"],
                model_scope=model.scope,
                initial_state=model.initial_state,
            )
            if startup_seconds is None:
                with (destination / "motion.csv").open(encoding="utf-8") as f:
                    rows = list(csv.DictReader(f))
                signals = {
                    key: [float(row[key]) for row in rows]
                    for key in rows[0]
                    if key != "phase"
                }
                for check in trial.checks:
                    if check.evidence == "manual":
                        record["checks"].append(
                            {
                                "id": check.id,
                                "task_ids": list(check.task_ids),
                                "description": check.description,
                                "passed": None,
                                "status": "awaiting_manual_evidence",
                            }
                        )
                        continue
                    try:
                        assessed = evaluate(check.expression, signals)
                    except (
                        ValueError,
                        TypeError,
                        IndexError,
                        ZeroDivisionError,
                    ) as exc:
                        # 判定程序出错只能说明这次没有结论，不能冒充机械功能失败。
                        assessed = {
                            "passed": None,
                            "status": "assessment_error",
                            "error": str(exc),
                        }
                    unavailable = {
                        task: gaps[task] for task in check.task_ids if task in gaps
                    }
                    if unavailable and assessed.get("status") != "assessment_error":
                        # 组件的真实数值仍保存；缺销/缺碰撞等事实不能只藏在scope文字里。
                        assessed["numerical_passed"] = assessed.get("passed")
                        assessed.update(
                            passed=None,
                            status="unavailable_physical_evidence",
                            evidence_gaps=unavailable,
                        )
                    record["checks"].append(
                        {
                            "id": check.id,
                            "task_ids": list(check.task_ids),
                            "description": check.description,
                            **assessed,
                        }
                    )
                record["functional_pass"] = (
                    None
                    if any(c["passed"] is None for c in record["checks"])
                    else all(c["passed"] for c in record["checks"])
                )
                if any(c.get("status") == "assessment_error" for c in record["checks"]):
                    record["status"] = "assessment_error"
            else:
                record["scope"] = "startup_prefix_only"
        except Exception as exc:
            record.update(
                status="binding_or_execution_error",
                error=f"{type(exc).__name__}: {exc}",
            )
        (output / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return summary


def run_cli(build, parameters):
    parser = argparse.ArgumentParser(
        description="机械家族模板：接入独立 Python 绑定文件运行"
    )
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step", type=float, default=0.001)
    parser.add_argument("--trial")
    parser.add_argument("--startup-seconds", type=float)
    args = parser.parse_args()
    binding = load_source_module(args.binding)
    values = ProtocolValues(parameters, binding.PROTOCOL)
    template = build(values).validate()
    protocol = dict(values.used)
    result = execute(
        template,
        binding,
        protocol,
        source_dir=args.source,
        output=args.output,
        step_s=args.step,
        trial_id=args.trial,
        startup_seconds=args.startup_seconds,
    )
    print(json.dumps(result, ensure_ascii=False))
    # 物理失败仍是成功完成测试；缺绑定、执行错误和判定错误则向批量调度器报错。
    if any(t["status"] != "completed" for t in result["trials"]):
        raise SystemExit(2)
    return result
