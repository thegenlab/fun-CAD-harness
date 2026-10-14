"""Deprecated design tombstone; do not integrate this module.

The abandoned implementation tried to infer physical support from joint axes
and revolute/prismatic/cylindrical templates.  That is the wrong abstraction
for detecting visibly floating assembly parts: the check must be derived from
the authoritative assembly geometry and must not depend on joint semantics.

This path intentionally remains as a tombstone so a later change cannot
silently revive or accidentally import the abandoned design.  Use
``assembly_connectivity`` instead.
"""

from __future__ import annotations


DEPRECATED = True
REPLACEMENT_MODULE = (
    "simulation_ready_cad_workflow.validation_pipeline.mechanism."
    "assembly_connectivity"
)
DEPRECATION_REASON = (
    "Axis- and joint-template-based support inference was rejected. "
    "Floating assembly islands are validated from authoritative geometry."
)


def __getattr__(name: str) -> object:
    raise RuntimeError(
        f"{__name__}.{name} is unavailable because this module is deprecated; "
        f"use {REPLACEMENT_MODULE}"
    )
