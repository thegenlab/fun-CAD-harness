"""Settings that differ between this distribution and the open-source release.

LOCAL_EXECUTION_ALLOWED: whether authored programs may run without the formal sandbox.
DEFAULT_BUDGET_HOURS / DEFAULT_SIMULATION_TIMEOUT_SECONDS: recommended limits for a
new high-tier run when none are given; None keeps the tier table's values.
FLOATING_PART_ADVICE: tell authors about floating parts (floating_parts.py).
"""
LOCAL_EXECUTION_ALLOWED = True
DEFAULT_BUDGET_HOURS = 5
DEFAULT_SIMULATION_TIMEOUT_SECONDS = 600
FLOATING_PART_ADVICE = True
