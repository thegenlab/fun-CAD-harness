"""把统一操作意图映射到真实手动输入；不新增机械关节或输出驱动。"""

import math
from .native import Actuator
from simulation_ready_cad_workflow.validation_pipeline.mechanism.pychrono_core import (
    load_pychrono_core,
)


class PointForceTorquePort(Actuator):
    """在实际手柄点施力，向模板提供角坐标/力矩接口。

    point_local是移动体COM局部材料点；axis_point_reference是参考体COM局部轴心。
    axis位于参考体系；direction_mode=tangent自动取当前切线，world/reference则用
    给定force_direction。每步用真实力臂换算，不能用固定设计半径代替当前位置。
    这是外部人力，不自动向参考体施加反力。max_force_n是正常手力上限；奇异力臂报错。
    """

    def __init__(
        self,
        obj,
        reference,
        *,
        point_local,
        axis_point_reference,
        axis=(0, 0, 1),
        direction_mode="tangent",
        force_direction=None,
        max_force_n,
    ):
        super().__init__(
            obj, "body_torque", reference=reference, axis=tuple(axis), react_on_reference=False
        )
        if direction_mode not in ("tangent", "world", "reference"):
            raise ValueError("未知手力方向模式")
        if not math.isfinite(max_force_n) or max_force_n <= 0:
            raise ValueError("手力上限必须正且有限")
        self.material_point = tuple(point_local)
        self.axis_point = tuple(axis_point_reference)
        self.direction_mode = direction_mode
        self.force_direction = force_direction
        self.max_force_n = max_force_n
        self._force_accumulator = None
        self.actual_force_n = 0.0
        self.actual_moment_arm_m = 0.0

    def _geometry(self):
        c = load_pychrono_core()
        point = self.obj.TransformPointLocalToParent(c.ChVector3d(*self.material_point))
        center = self.reference.TransformPointLocalToParent(c.ChVector3d(*self.axis_point))
        axis = self.reference.GetRot().Rotate(c.ChVector3d(*self.axis))
        axis = axis / axis.Length()
        r = point - center
        if self.direction_mode == "tangent":
            direction = axis.Cross(r)
        else:
            if self.force_direction is None:
                raise ValueError("固定方向模式缺少force_direction")
            direction = c.ChVector3d(*self.force_direction)
            if self.direction_mode == "reference":
                direction = self.reference.GetRot().Rotate(direction)
        if direction.Length() <= 1e-12:
            raise ValueError("正常输入点或方向没有可用旋转力臂")
        direction = direction / direction.Length()
        lever = r.Cross(direction).Dot(axis)
        if abs(lever) <= 1e-12:
            raise ValueError("手力方向与当前轴线产生零力矩，不能完成此输入映射")
        return point, direction, lever

    def add_input(self, effort):
        if not math.isfinite(effort):
            raise ValueError("人工力矩非有限")
        point, direction, lever = self._geometry()
        force = max(-self.max_force_n, min(self.max_force_n, effort / lever))
        if self._force_accumulator is None:
            self._force_accumulator = self.obj.AddAccumulator()
        self.obj.AccumulateForce(self._force_accumulator, direction * force, point, False)
        self.actual_force_n = force
        self.actual_moment_arm_m = lever
        self._last_input += force * lever

    def clear_input(self):
        if self._force_accumulator is not None:
            self.obj.EmptyAccumulator(self._force_accumulator)
        self._last_input = 0.0
        self.actual_force_n = 0.0
