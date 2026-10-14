"""Reusable live and snapshot dashboard for mechanism-validation runs."""

from .model import normalize_run, scan_runs
from .render import render_snapshot

__all__ = ["normalize_run", "render_snapshot", "scan_runs"]
