"""Select the formal sandbox or local execution for agent turns and authored programs.

The formal sandbox is used whenever it is installed. A distribution without it
runs locally only when ``execution_profile.LOCAL_EXECUTION_ALLOWED`` says so;
otherwise every launch fails closed exactly as a missing sandbox configuration.
"""
from __future__ import annotations

from importlib.util import find_spec
import os
from pathlib import Path

from . import execution_profile

_FORMAL_COMPUTE = __package__.rsplit('.', 1)[0] + '.benchmark_sandbox.formal_compute'
CONTAINER_ENV = 'SIMREADY_CONTAINER_ISOLATION'


def formal_sandbox_required():
    if find_spec(_FORMAL_COMPUTE) is not None:
        return True
    if not execution_profile.LOCAL_EXECUTION_ALLOWED:
        from ..runtime.bounded_subprocess import BoundedSubprocessError
        raise BoundedSubprocessError('FORMAL_SANDBOX_CONFIGURATION',
                                     'The formal sandbox is not installed and local execution is not allowed')
    return False


def mode():
    """Recorded execution mode: formal, container or local."""
    if formal_sandbox_required():
        return 'formal'
    return 'container' if os.environ.get(CONTAINER_ENV) == '1' else 'local'


def run_compute(command, *, workspace, project, inputs, outputs, timeout_seconds,
                max_rss_bytes='auto', max_output_bytes=16 * 1024 * 1024, env=None,
                input_bytes=b'', cwd=None, experimental_no_timeout=False, runner=None):
    if formal_sandbox_required():
        from ..benchmark_sandbox.formal_compute import run_compute as run_formal
        return run_formal(command, workspace=workspace, project=project, inputs=inputs, outputs=outputs,
                          timeout_seconds=timeout_seconds, max_rss_bytes=max_rss_bytes,
                          max_output_bytes=max_output_bytes, env=env, input_bytes=input_bytes, cwd=cwd,
                          experimental_no_timeout=experimental_no_timeout, runner=runner)
    from ..runtime.bounded_subprocess import run_bounded_subprocess
    return (runner or run_bounded_subprocess)(
        command, timeout_seconds=timeout_seconds, max_rss_bytes=max_rss_bytes,
        max_output_bytes=max_output_bytes, env=env, cwd=cwd or Path(workspace).resolve(),
        input_bytes=input_bytes, experimental_no_timeout=experimental_no_timeout)


def geometry_inputs(scene):
    if formal_sandbox_required():
        from ..benchmark_sandbox.formal_compute import geometry_inputs as formal_inputs
        return formal_inputs(scene)
    # Local launches have no data grants to declare.
    return []
