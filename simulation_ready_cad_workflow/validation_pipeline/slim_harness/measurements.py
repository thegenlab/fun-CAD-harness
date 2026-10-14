"""Small expression evaluator over Harness-recorded numeric signals.

Optional expressions, with no task/template identifiers::

    cycles(window(signal("piston.position.x_m"), 0.8, 6.0), 0.030, 0.001) >= 2
    max(abs(window(signal("beam.angular_speed.z_rad_s"), 2.0, 3.0))) <= 0.02

``window(values, start_s, end_s)`` includes recorded samples at both boundaries;
it does not interpolate or invent samples. Its time coordinates survive nested
windows, ``after``/``before``, and ordinary elementwise arithmetic, so integral
uses the selected time interval rather than the full trace's clock.

``cycles(values, minimum_stroke, return_tolerance)`` counts non-overlapping,
ordered low-high-low or high-low-high triples of consecutive turning points.
Both legs must reach the minimum stroke, and the returned endpoint must be
within the stated tolerance of that cycle's start. The trace endpoints may
close a cycle; incomplete leading/trailing fragments do not get extrapolated.
It measures excursions, not accumulated travel, velocity settling, or return
to a separately declared absolute home. Stroke and tolerance use the signal's
own units. Only roundoff-sized reversals are suppressed: 64 * float64 epsilon
* max(abs(values), minimum_stroke). No physical-amplitude smoothing is applied.
"""
from __future__ import annotations

import ast
from typing import Any, Collection
import numpy as np


MEASUREMENT_FUNCTION_NAMES = frozenset({
    "signal", "min", "max", "mean", "span", "rms", "first", "last", "abs",
    "integral", "window", "cycles", "after", "before", "relative_position",
})


def validate_expression(expression: str, functions: Collection[str] = MEASUREMENT_FUNCTION_NAMES) -> ast.Expression:
    """Check the existing expression language without reading physical signals."""
    tree = ast.parse(expression, mode="eval")
    allowed = (ast.Expression, ast.Constant, ast.Call, ast.Name, ast.Load,
               ast.BinOp, ast.UnaryOp, ast.Compare, ast.Add, ast.Sub, ast.Mult,
               ast.Div, ast.Pow, ast.USub, ast.UAdd, ast.Lt, ast.LtE, ast.Gt,
               ast.GtE, ast.Eq, ast.NotEq)
    for node in ast.walk(tree):
        if not isinstance(node, allowed):
            raise ValueError(f"Unsupported measurement syntax: {type(node).__name__}")
        if isinstance(node, ast.Call) and (
            not isinstance(node.func, ast.Name) or node.func.id not in functions
            or node.keywords
        ):
            raise ValueError("Only documented measurement functions are callable")
        if isinstance(node, ast.Name) and node.id not in functions:
            raise ValueError(f"Unknown measurement function: {node.id}")
    return tree


class _TimedValues(np.ndarray):
    """An expression array carrying its recorded time coordinates internally."""

    def __new__(cls, values, times):
        result = np.asarray(values, dtype=float).view(cls)
        result.times = np.asarray(times, dtype=float)
        return result

    def __array_finalize__(self, original):
        self.times = getattr(original, "times", None)


def _finite_scalar(value, label):
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind not in "fiu" or not np.isfinite(array):
        raise ValueError(f"{label} must be a finite number")
    return float(array)


def cycles(values, minimum_stroke, return_tolerance) -> int:
    """Count complete consecutive-extrema excursions; see the module contract."""
    data = np.asarray(values, dtype=float)
    if data.ndim != 1 or not data.size or not np.isfinite(data).all():
        raise ValueError("Cycle counting requires a nonempty finite one-dimensional signal")
    stroke = _finite_scalar(minimum_stroke, "minimum_stroke")
    tolerance = _finite_scalar(return_tolerance, "return_tolerance")
    if stroke <= 0 or tolerance < 0:
        raise ValueError("minimum_stroke must be positive and return_tolerance nonnegative")
    if data.size < 3:
        return 0
    noise = 64 * np.finfo(float).eps * max(stroke, float(np.max(np.abs(data))))
    turning = [(0, float(data[0]))]
    direction = 0
    extreme_index, extreme = 0, float(data[0])
    for index in range(1, len(data)):
        value = float(data[index])
        if direction == 0:
            change = value - turning[0][1]
            if abs(change) > noise:
                direction = 1 if change > 0 else -1
                extreme_index, extreme = index, value
        elif direction * (value - extreme) >= 0:
            extreme_index, extreme = index, value
        elif direction * (value - extreme) < -noise:
            turning.append((extreme_index, extreme))
            direction = -direction
            extreme_index, extreme = index, value
    if turning[-1][0] != len(data) - 1:
        turning.append((len(data) - 1, float(data[-1])))
    count, index = 0, 0
    while index + 2 < len(turning):
        a, b, c = (point[1] for point in turning[index:index + 3])
        opposite_legs = (b > a and c < b) or (b < a and c > b)
        if (opposite_legs and abs(b-a) + noise >= stroke and abs(c-b) + noise >= stroke
                and abs(c-a) <= tolerance + noise):
            count += 1
            index += 2  # Adjacent cycles may share their returned endpoint only.
        else:
            index += 1
    return count


class _RelativePositions:
    """Read-only coordinates from recorded COM poses; no inferred motion."""

    def __init__(self, samples, times):
        self.samples, self.times = samples, np.asarray(times, dtype=float)
        self.poses = None
        self.cache = {}

    def _load(self):
        if self.samples is None:
            raise ValueError("relative_position requires recorded body poses")
        if len(self.samples) != len(self.times) or not len(self.times):
            raise ValueError("Body poses must cover every signal time")
        poses = {}
        for index, sample in enumerate(self.samples):
            if sample.get("time_s") != self.times[index]:
                raise ValueError("Body pose times must match signal times")
            for name, pose in sample.get("body_poses", {}).items():
                rows = poses.setdefault(name, [])
                if len(rows) != index:
                    raise ValueError("Body poses must remain available throughout the trace")
                rows.append([*pose["position_abs_m"], *pose["rotation_abs_quat_wxyz"]])
        self.poses = {}
        for name, rows in poses.items():
            values = np.asarray(rows, dtype=float)
            if values.shape != (len(self.times), 7) or not np.isfinite(values).all():
                raise ValueError("Body poses must contain finite positions and quaternions")
            norms = np.linalg.norm(values[:, 3:], axis=1)
            if np.any(norms == 0):
                raise ValueError("Body orientation quaternion must be nonzero")
            values[:, 3:] /= norms[:, None]
            self.poses[name] = values

    def __call__(self, body, reference, axis):
        if not isinstance(body, str) or not isinstance(reference, str) or axis not in ("x", "y", "z"):
            raise ValueError("relative_position requires body, reference body and x/y/z axis")
        if self.poses is None:
            self._load()
        if body not in self.poses or reference not in self.poses:
            raise ValueError("relative_position names an unknown recorded body")
        key = body, reference
        if key not in self.cache:
            target, frame = self.poses[body], self.poses[reference]
            delta = target[:, :3]-frame[:, :3]
            w, vector = frame[:, 3:4], frame[:, 4:]
            cross = 2*np.cross(vector, delta)
            # Inverse quaternion rotation: R_reference.T @ (p_body-p_reference).
            self.cache[key] = delta-w*cross+np.cross(vector, cross)
        return _TimedValues(self.cache[key][:, ("x", "y", "z").index(axis)], self.times)


def _evaluate(expression: str, signals: dict[str, list[float]], relative_positions) -> dict[str, Any]:
    times = np.asarray(signals["time_s"], dtype=float)

    def signal(name):
        if name not in signals:
            raise ValueError(f"Unknown physical signal: {name}")
        return _TimedValues(signals[name], times)

    def timed(values):
        data = np.asarray(values)
        clock = getattr(values, "times", None)
        clock = times if clock is None else np.asarray(clock, dtype=float)
        if data.ndim != 1 or data.shape != clock.shape:
            raise ValueError("Time window requires one value per recorded time")
        if not np.isfinite(clock).all() or np.any(np.diff(clock) < 0):
            raise ValueError("Recorded time must be finite and monotonic")
        return data, clock

    def subset(values, threshold, *, later):
        data, clock = timed(values)
        threshold = _finite_scalar(threshold, "time threshold")
        mask = clock >= threshold if later else clock <= threshold
        selected = data[mask]
        if not selected.size:
            raise ValueError("The requested time window has no samples")
        return _TimedValues(selected, clock[mask])

    def window(values, start_s, end_s):
        data, clock = timed(values)
        start = _finite_scalar(start_s, "window start")
        end = _finite_scalar(end_s, "window end")
        if start > end:
            raise ValueError("Window start must not exceed its end")
        mask = (clock >= start) & (clock <= end)
        if not mask.any():
            raise ValueError("The requested time window has no samples")
        return _TimedValues(data[mask], clock[mask])

    def integral(values):
        data, clock = timed(values)
        return np.trapezoid(data, clock)

    functions = {
        "signal": signal, "relative_position": relative_positions,
        "min": lambda x: np.min(np.asarray(x)),
        "max": lambda x: np.max(np.asarray(x)), "mean": lambda x: np.mean(np.asarray(x)),
        "span": lambda x: np.ptp(np.asarray(x)), "rms": lambda x: np.sqrt(np.mean(np.square(np.asarray(x)))),
        "first": lambda x: x[0], "last": lambda x: x[-1], "abs": np.abs,
        "integral": integral, "window": window, "cycles": cycles,
        "after": lambda x, t: subset(x, t, later=True),
        "before": lambda x, t: subset(x, t, later=False),
    }
    tree = validate_expression(expression, functions)
    try:
        with np.errstate(divide="raise", invalid="raise", over="raise"):
            value = eval(compile(tree, "<measurement>", "eval"), {"__builtins__": {}}, functions)
    except (FloatingPointError, OverflowError) as exc:
        raise ValueError("Measurement produced a nonfinite value") from exc
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError("An assertion must evaluate to a single comparison")
    operands = []
    if isinstance(tree.body, ast.Compare):
        for operand in (tree.body.left, *tree.body.comparators):
            result = eval(compile(ast.Expression(operand), "<measurement>", "eval"), {"__builtins__": {}}, functions)
            array = np.asarray(result)
            if not np.all(np.isfinite(array)):
                raise ValueError("Measurement operand is not finite")
            operands.append(float(array) if array.ndim == 0 else {
                "min": float(array.min()), "max": float(array.max()), "count": array.size,
            })
    return {"expression": expression, "passed": bool(value), "observed_operands": operands}


def evaluate(expression: str, signals: dict[str, list[float]], *, pose_samples=None) -> dict[str, Any]:
    """Evaluate signals plus optional recorded body-local COM coordinates.

    relative_position(body, reference, axis) returns the target body's COM in
    the moving reference body's COM-centered local frame, in metres. Both
    translation and orientation are taken from each actual recorded sample.
    """
    return _evaluate(expression, signals, _RelativePositions(pose_samples, signals["time_s"]))


def evaluate_assertions(test: dict, signals: dict, *, pose_samples=None) -> list[dict]:
    relative_positions = _RelativePositions(pose_samples, signals["time_s"])
    rows = []
    for assertion in test["assertions"]:
        try:
            result = _evaluate(assertion["expression"], signals, relative_positions)
        except (ValueError, TypeError, SyntaxError, IndexError, ZeroDivisionError) as exc:
            result = {"passed": False, "error": str(exc)}
        rows.append({**assertion, **result})
    return rows
