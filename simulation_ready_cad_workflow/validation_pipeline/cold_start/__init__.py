"""Canonical unknown-CAD cold-start pipeline.

Import concrete modules directly. Keeping this package root light prevents
the current plan-only runner and cold-start orchestrator from importing each
other during package initialization.
"""

__all__: list[str] = []
