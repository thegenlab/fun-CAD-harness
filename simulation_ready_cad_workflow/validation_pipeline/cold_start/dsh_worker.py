"""Original DeepSeek Harness SDK worker, available only through formal isolation."""
from .worker import CodexCliWorker
from ..trajectory.capture import record_turn


class DshWorker(CodexCliWorker):
    def __init__(self, model="deepseek-flash", reasoning_effort="high", executable="dsh", session_id=None):
        if reasoning_effort not in {"low", "high", "max"}:
            raise ValueError("DSH supports low, high, or max reasoning")
        super().__init__(model=model, reasoning_effort=reasoning_effort,
                         executable=executable, session_id=session_id)

    @record_turn
    def run_turn(self, *args, **kwargs):
        if getattr(self, "benchmark_sandbox_context", {}).get("client") != "dsh":
            raise ValueError("DSH requires the installed formal sandbox binding")
        from ..benchmark_sandbox.formal_worker import run_turn
        return run_turn(self, *args, **kwargs)
