"""Small source bindings for the three expensive slim Harness caches.

Producers capture these at module startup. This deliberately does not track
the repository, dependencies, environment, or historical publication status.
"""
from __future__ import annotations

import hashlib
from pathlib import Path


def source_sha256(filename: str) -> str:
    return hashlib.sha256((Path(__file__).parent / filename).read_bytes()).hexdigest()


def native_implementation_binding() -> dict[str, str]:
    return {name: source_sha256(name) for name in (
        "runtime.py", "runtime_progress.py", "measurements.py", "contracts.py", "trace_storage.py",
        "fixed_component_filter.py", "source_contact_filter.py", "source_contact_normals.py", "source_capsule_contacts.py",
        "source_queries.py", "source_support.py", "numerical.py",
        "equivalent_elasticity.py", "equivalent_elastic_compiler.py",
        "equivalent_collision_scope.py", "validation_scope.py", "spur_gears.py", "shaft_supports.py",
        "measurement_registry.py", "alarm_measurements.py",
        "clock_contacts/contact_collector.py", "clock_contacts/calibration_binding.py",
        "clock_contacts/runtime_identity.py",
    )}
