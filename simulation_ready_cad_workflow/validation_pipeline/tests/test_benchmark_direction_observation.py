"""方向观测的倾斜转轴/归一化/无定义状态；不模拟车辆。"""

import math
import unittest
from benchmark_templates.observations import signed_direction_angle


class DirectionObservationTests(unittest.TestCase):
    def rotate(self, vector, axis, angle):
        norm = math.sqrt(sum(x * x for x in axis))
        u = tuple(x / norm for x in axis)
        cross = (
            u[1] * vector[2] - u[2] * vector[1],
            u[2] * vector[0] - u[0] * vector[2],
            u[0] * vector[1] - u[1] * vector[0],
        )
        dot = sum(x * y for x, y in zip(u, vector))
        return tuple(
            vector[i] * math.cos(angle)
            + cross[i] * math.sin(angle)
            + u[i] * dot * (1 - math.cos(angle))
            for i in range(3)
        )

    def test_tilted_and_vertical_steering_axes(self):
        for axis in ((0, 0, 1), (0.3, 0, 0.9539392014169457)):
            for angle in (-0.4, 0, 0.25, 0.8):
                with self.subTest(axis=axis, angle=angle):
                    v = self.rotate((0, 1, 0), axis, angle)
                    self.assertAlmostEqual(
                        angle, signed_direction_angle((0, 1, 0), v, axis), places=12
                    )

    def test_direction_scale_does_not_change_angle(self):
        v = self.rotate((0, 2, 0), (0, 0, 1), 0.25)
        self.assertAlmostEqual(
            0.25, signed_direction_angle((0, 7, 0), v, (0, 0, 3)), places=12
        )

    def test_undefined_parallel_measurement_is_rejected(self):
        with self.assertRaises(ValueError):
            signed_direction_angle((0, 0, 1), (0, 0, 1), (0, 0, 1))
        with self.assertRaises(ValueError):
            signed_direction_angle((0, 1, 0), (1, 0, 0), (0, 0, 0))


if __name__ == "__main__":
    unittest.main()
