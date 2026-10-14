"""Trusted post-intake mechanism discovery and validation dispatch."""

from .discovery import discover_mechanisms
from .dispatch import DispatchBundle, build_dispatch_bundle
from .finalize import finalize_dispatch
from .family_plugins import (
    CompoundMechanismAdapter,
    MechanismAdapter,
    PlainPrismaticAdapter,
    PlainRevoluteAdapter,
    RackPinionAdapter,
    mechanism_adapter_for_plan,
    mechanism_adapter_inventory,
    public_simulation_interface_catalog,
    registered_mechanism_adapters,
)
from .loads import (
    default_compound_load_cases,
    default_load_cases,
    default_prismatic_load_cases,
    default_revolute_load_cases,
)
from .launcher import (
    PyChronoLauncher,
    PyChronoLauncherKind,
    resolve_pychrono_launcher,
)
from .conditional_interface import (
    APPROACHING,
    ENGAGED,
    INACTIVE,
    RELEASING,
    AxialGearPose,
    AxialSlidingSpurGearGeometry,
    ConditionalCompliantInterface,
    InterfaceGeometryObservation,
    assess_initial_functional_geometry,
    audit_conditional_interface_runtime_evidence,
    build_functional_interface_geometry_binding,
    conditional_transmission_gate,
    validate_functional_interface_geometry_binding,
)
from .models import LoadCase, LoadPhase, MechanismPlan, MechanismValidationError
from .whole_machine import build_whole_machine_dispatch

__all__ = [
    "DispatchBundle",
    "LoadCase",
    "LoadPhase",
    "MechanismPlan",
    "MechanismValidationError",
    "PyChronoLauncher",
    "PyChronoLauncherKind",
    "build_dispatch_bundle",
    "build_whole_machine_dispatch",
    "MechanismAdapter",
    "CompoundMechanismAdapter",
    "PlainPrismaticAdapter",
    "PlainRevoluteAdapter",
    "RackPinionAdapter",
    "registered_mechanism_adapters",
    "mechanism_adapter_for_plan",
    "mechanism_adapter_inventory",
    "public_simulation_interface_catalog",
    "default_load_cases",
    "default_compound_load_cases",
    "default_prismatic_load_cases",
    "default_revolute_load_cases",
    "discover_mechanisms",
    "finalize_dispatch",
    "resolve_pychrono_launcher",
    "APPROACHING",
    "ENGAGED",
    "INACTIVE",
    "RELEASING",
    "AxialGearPose",
    "AxialSlidingSpurGearGeometry",
    "ConditionalCompliantInterface",
    "InterfaceGeometryObservation",
    "assess_initial_functional_geometry",
    "audit_conditional_interface_runtime_evidence",
    "build_functional_interface_geometry_binding",
    "conditional_transmission_gate",
    "validate_functional_interface_geometry_binding",
]
