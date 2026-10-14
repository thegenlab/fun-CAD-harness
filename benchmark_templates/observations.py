"""共享真实运动观察；judge只绑定输入/输出，不自行编造周期判据。"""

import math
from .native import Signal


class CycleReturnObserver:
    """在实际输入每前进一整周期时观察输出回位；不按墙钟猜输入相位。

    交叉点只在相邻真实样本之间线性插值，不外推；未完成一周时count为0。
    输入逆转不会重复计算同一阈值，只有超过下一个整周相位才增加计数。
    """

    def __init__(self, input_read, output_read, period):
        if not math.isfinite(period) or period <= 0:
            raise ValueError("周期必须有限且为正")
        self.input_read = input_read
        self.output_read = output_read
        self.period = period
        self.start = float(input_read())
        self.home = float(output_read())
        if not math.isfinite(self.start) or not math.isfinite(self.home):
            raise ValueError("初始观测非有限")
        self.previous_input = self.start
        self.previous_output = self.home
        self.completed = 0
        self.max_return_error = 0.0

    def update(self):
        current = float(self.input_read())
        output = float(self.output_read())
        if not math.isfinite(current) or not math.isfinite(output):
            raise ValueError("实际观测非有限")
        threshold = self.start + (self.completed + 1) * self.period
        while current >= threshold and current > self.previous_input:
            fraction = (threshold - self.previous_input) / (
                current - self.previous_input
            )
            if not 0 <= fraction <= 1:
                raise ValueError("整周期采样没有真实交叉区间")
            at_return = self.previous_output + fraction * (
                output - self.previous_output
            )
            self.max_return_error = max(
                self.max_return_error, abs(at_return - self.home)
            )
            self.completed += 1
            threshold = self.start + (self.completed + 1) * self.period
        self.previous_input = current
        self.previous_output = output


def observe_cycle_returns(
    model, input_signal, output_signal, period, *, count_name, error_name
):
    """接原模型after_step后读取真实相位；不施力、不改变机械或推进时间。"""
    observer = CycleReturnObserver(input_signal.read, output_signal.read, period)
    previous = model.after_step

    def after(now, dt):
        if previous:
            previous(now, dt)
        observer.update()

    model.after_step = after
    model.signals[count_name] = Signal(
        lambda: float(observer.completed), "source_scalar"
    )
    model.signals[error_name] = Signal(
        lambda: observer.max_return_error, "source_scalar"
    )
    from .source_loading import retain_objects

    retain_objects(model, observer, after)
    return observer


class PairedPeakPhaseObserver:
    """双侧突出度确认实际极大值；不把运动中的微小反转当完整峰值。

    必须先从实际谷部上升半行程，再从候选最高点下降半行程，才确认该事件。
    候选随真实更高位置更新；确认延迟不改变其实际时间/轴角。
    """

    def __init__(self, crank, positions, stroke):
        self.crank = crank
        self.positions = positions
        self.stroke = stroke
        self.minimum = {k: positions[k].read() for k in positions}
        self.candidates = {k: None for k in positions}
        self.peaks = {k: [] for k in positions}

    def observe(self, time_s):
        theta = self.crank.read()
        for key, signal in self.positions.items():
            x = signal.read()
            candidate = self.candidates[key]
            prominence = 0.5 * self.stroke[key]
            event = {"time_s": time_s, "position_m": x, "crank_angle_rad": theta}
            if candidate is None:
                self.minimum[key] = min(self.minimum[key], x)
                if x - self.minimum[key] >= prominence:
                    self.candidates[key] = event
            else:
                if x > candidate["position_m"]:
                    self.candidates[key] = event
                    candidate = event
                if candidate["position_m"] - x >= prominence:
                    self.peaks[key].append(candidate)
                    self.candidates[key] = None
                    self.minimum[key] = x

    def count(self):
        return min(len(self.peaks["a"]), len(self.peaks["b"]))

    def max_error(self, target):
        """逐对真实峰事件的最坏圆距离；不让正负误差平均抵消，也不误罚2pi跨界。"""
        n = self.count()
        if n == 0:
            return 0.0  # 独立phase_pairs判据要求有足够真实事件，此值不能单独判通过。
        errors = []
        for i in range(n):
            delta = (
                self.peaks["b"][i]["crank_angle_rad"]
                - self.peaks["a"][i]["crank_angle_rad"]
                - target
            )
            errors.append(abs(math.atan2(math.sin(delta), math.cos(delta))))
        return max(errors)

    def phase(self):
        n = self.count()
        if n == 0:
            return 0.0
        differences = [
            (
                self.peaks["b"][i]["crank_angle_rad"]
                - self.peaks["a"][i]["crank_angle_rad"]
            )
            % (2 * math.pi)
            for i in range(n)
        ]
        return math.atan2(
            sum(math.sin(x) for x in differences), sum(math.cos(x) for x in differences)
        ) % (2 * math.pi)


class CyclicPluckObserver:
    """从真实接触事件读曲谱次序；同一相位可以有多根针同时拨动。

    programs为源CAD针表：pin_id、note_id、phase_rad。phase是该针经过拨奏线的
    实际曲筒角，不是按仿真时钟编的期望轨迹。events由实例绑定识别实际针和簧片。
    """

    def __init__(self, programs):
        self.programs = programs
        self.program = None
        self.expected = 0
        self.pending = set()
        self.seen = set()
        self.notes_seen = set()
        self.count = 0
        self.repluck = 0
        self.pin_note = self.tine_note = -1.0
        self.phase_gap = self.restore_error = 0.0
        self.groups = []
        self.order_resolution_gap = False

    def select(self, program, angle):
        if program == self.program:
            return
        self.program = program
        self.notes_seen.clear()
        self.groups = []
        if program not in self.programs:
            return
        grouped = {}
        for pin in self.programs[program]:
            grouped.setdefault(float(pin["phase_rad"]) % math.tau, []).append(pin)
        self.groups = sorted(grouped.items())
        phase = angle % math.tau
        self.expected = next(
            (i for i, (p, _) in enumerate(self.groups) if p >= phase - 1e-12), 0
        )
        self.pending = {p["pin_id"] for p in self.groups[self.expected][1]}

    def observe(self, program, angle, events):
        self.select(program, angle)
        self.pin_note = self.tine_note = -1.0
        self.phase_gap = self.restore_error = 0.0
        accepted = []
        self.order_resolution_gap = False
        if not self.groups:
            return accepted
        pins = {p["pin_id"]: p for p in self.programs[program]}
        # 同一求解步的和弦不会被塞成一个事件；每条身份和接触来源均交回运行记录。
        new_events = [
            e
            for e in events
            if (
                program,
                e["pin_id"],
                e["tine_id"],
                int(round((angle - float(pins[e["pin_id"]]["phase_rad"])) / math.tau)),
            )
            not in self.seen
        ]
        phases = {float(pins[e["pin_id"]]["phase_rad"]) % math.tau for e in new_events}
        # 两个不同针位在同一步首次被看见，只有部分次序证据；和弦同相位不受此限制。
        self.order_resolution_gap = len(phases) > 1

        def unfolded_phase(event):
            phase = float(pins[event["pin_id"]]["phase_rad"])
            return phase + round((angle - phase) / math.tau) * math.tau

        for event in sorted(new_events, key=unfolded_phase):
            pin = pins[event["pin_id"]]
            turn = int(round((angle - float(pin["phase_rad"])) / math.tau))
            key = (program, event["pin_id"], event["tine_id"], turn)
            if key in self.seen:
                continue
            self.seen.add(key)
            self.count += 1
            if event["tine_id"] in self.notes_seen:
                self.repluck += 1
            self.notes_seen.add(event["tine_id"])
            actual_note, target_note = float(event["tine_note_id"]), float(
                pin["note_id"]
            )
            if abs(actual_note - target_note) >= abs(self.tine_note - self.pin_note):
                self.pin_note, self.tine_note = target_note, actual_note
            expected_phase = self.groups[self.expected][0]
            gap = (float(pin["phase_rad"]) - expected_phase) % math.tau
            self.phase_gap = max(self.phase_gap, gap)
            self.restore_error = max(
                self.restore_error, float(event["restore_before_m"])
            )
            if event["pin_id"] in self.pending:
                self.pending.remove(event["pin_id"])
                if not self.pending:
                    self.expected = (self.expected + 1) % len(self.groups)
                    self.pending = {p["pin_id"] for p in self.groups[self.expected][1]}
            accepted.append(
                {
                    **event,
                    "program": program,
                    "turn": turn,
                    "pin_note_id": target_note,
                    "phase_gap_rad": gap,
                }
            )
        return accepted


class InterPluckRecoveryObserver:
    """两次真实接触拨动之间最接近自由位的实测残余；不要求簧片停振。"""

    def __init__(self, initial_displacements):
        self.current = dict(initial_displacements)
        self.previous = dict(initial_displacements)
        self.plucked = set()
        self.minimum = {}

    def sample(self, displacements, contacting):
        # 接触中的瞬时过零不算释放后的恢复；只积累没有针接触的实际采样。
        self.previous = self.current
        self.current = dict(displacements)
        contacting = set(contacting)
        for tine in self.plucked:
            if tine not in contacting:
                value = float(self.current[tine])
                prior = self.minimum.get(tine)
                self.minimum[tine] = value if prior is None else min(prior, value)

    def pluck(self, tine):
        applicable = tine in self.plucked
        minimum = self.minimum.get(tine)
        observed = not applicable or minimum is not None
        previous = float(self.previous[tine])
        # 首次拨动无“再次恢复”前提，0明确是中性项；原始位移仍完整保留。
        # 后续未观测到离开接触的间隔时，保留实际前样本且另报缺证，绝不补0。
        error = (
            0.0 if not applicable else previous if minimum is None else float(minimum)
        )
        result = dict(
            restore_applicable=applicable,
            restore_interval_observed=observed,
            restore_before_m=error,
            pre_contact_displacement_m=previous,
        )
        self.plucked.add(tine)
        self.minimum[tine] = None
        return result


class ContactMotionObserver:
    """真实针接触且簧片在该次接触中确实移动，才确认为拨动；没有动力学副作用。"""

    def __init__(self, initial_positions, minimum_motion_m):
        if not math.isfinite(minimum_motion_m) or minimum_motion_m <= 0:
            raise ValueError("pluck_motion_min_m must be finite and positive")
        self.minimum = float(minimum_motion_m)
        self.previous = {k: tuple(v) for k, v in initial_positions.items()}
        self.pending = {}
        self.confirmed = set()

    def sample(self, time_s, contacts, positions):
        positions = {k: tuple(v) for k, v in positions.items()}
        current = {tuple(event["contact_key"]): event for event in contacts}
        accepted, expired = [], []
        self.started = []
        for key, event in current.items():
            if key in self.confirmed:
                continue
            tine = event["tine_id"]
            if key not in self.pending:
                self.pending[key] = dict(
                    event,
                    contact_start_s=float(time_s),
                    pre_contact_position_m=self.previous[tine],
                    max_contact_motion_m=0.0,
                )
                self.started.append(dict(self.pending[key]))
            pending = self.pending[key]
            motion = math.dist(positions[tine], pending["pre_contact_position_m"])
            pending["max_contact_motion_m"] = max(
                pending["max_contact_motion_m"], motion
            )
            if pending["max_contact_motion_m"] >= self.minimum:
                accepted.append(
                    dict(
                        pending,
                        motion_confirmed_s=float(time_s),
                        confirmation_position_m=positions[tine],
                        confirmation_contacts=event.get("contacts", []),
                    )
                )
                self.confirmed.add(key)
                del self.pending[key]
        for key in list(self.pending):
            if key not in current:
                expired.append(
                    dict(
                        self.pending.pop(key),
                        contact_end_s=float(time_s),
                        reason="contact ended without measured minimum tine motion",
                    )
                )
        self.previous = positions
        return accepted, expired


def signed_direction_angle(reference, current, axis):
    """两根实际方向在指定轴法平面内的有符号角；轮轴方向不受车轮自转影响。"""

    def dot(a, b):
        return sum(x * y for x, y in zip(a, b))

    length = math.sqrt(dot(axis, axis))
    if length <= 1e-15:
        raise ValueError("Measurement axis must be nonzero")
    axis = tuple(x / length for x in axis)

    def projected(v):
        parallel = dot(v, axis)
        v = tuple(v[i] - parallel * axis[i] for i in range(3))
        length = math.sqrt(dot(v, v))
        if length <= 1e-12:
            raise ValueError("Measured direction must not be parallel to angle axis")
        return tuple(x / length for x in v)

    a, b = projected(reference), projected(current)
    cross = (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )
    return math.atan2(dot(axis, cross), dot(a, b))
