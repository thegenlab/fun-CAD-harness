from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

from simulation_ready_cad_workflow.validation_pipeline.runtime.bounded_capture import (
    atomic_write_bytes,
)


def run_contact_probe(
    *,
    step_seconds: float = 0.001,
    duration_seconds: float = 1.5,
) -> dict[str, Any]:
    if step_seconds <= 0.0 or duration_seconds <= step_seconds:
        raise ValueError("probe duration and step must define at least two steps")

    from ...mechanism.pychrono_core import load_pychrono_core

    chrono = load_pychrono_core()

    system = chrono.ChSystemNSC()
    system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
    system.SetGravitationalAcceleration(chrono.ChVector3d(0.0, 0.0, -9.81))

    # PyChrono 10 permits a null default material in the EasyBody helpers.
    # A collision-enabled body must therefore receive an explicit material
    # matching the NSC contact container; otherwise first contact can crash in
    # native code instead of producing a trustworthy failure result.
    material = chrono.ChContactMaterialNSC()
    ground = chrono.ChBodyEasyBox(
        2.0,
        2.0,
        0.2,
        1000.0,
        True,
        True,
        material,
    )
    ground.SetFixed(True)
    ground.SetPos(chrono.ChVector3d(0.0, 0.0, -0.1))
    sphere = chrono.ChBodyEasySphere(
        0.1,
        1000.0,
        True,
        True,
        material,
    )
    sphere.SetPos(chrono.ChVector3d(0.0, 0.0, 0.75))
    system.Add(ground)
    system.Add(sphere)

    step_count = int(math.ceil(duration_seconds / step_seconds))
    maximum_contacts = 0
    first_contact_step: int | None = None
    for step_index in range(step_count):
        system.DoStepDynamics(step_seconds)
        contacts = int(system.GetContactContainer().GetNumContacts())
        maximum_contacts = max(maximum_contacts, contacts)
        if contacts and first_contact_step is None:
            first_contact_step = step_index + 1

    final_height_m = float(sphere.GetPos().z)
    passed = (
        math.isfinite(final_height_m)
        and 0.07 < final_height_m < 0.14
        and maximum_contacts > 0
    )
    return {
        "schema_version": "0.1",
        "status": "PASSED" if passed else "FAILED",
        "physics_model": "PYCHRONO_NSC_BULLET_DROP_CONTACT",
        "uses_ideal_joint": False,
        "collision_backend": type(system.GetCollisionSystem()).__name__,
        "step_seconds": step_seconds,
        "duration_seconds": step_count * step_seconds,
        "step_count": step_count,
        "first_contact_step": first_contact_step,
        "maximum_contacts": maximum_contacts,
        "final_contacts": int(system.GetContactContainer().GetNumContacts()),
        "final_sphere_center_height_m": final_height_m,
        "expected_resting_center_height_m": 0.1,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a bounded PyChrono Bullet drop-contact health probe."
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--step-seconds", type=float, default=0.001)
    parser.add_argument("--duration-seconds", type=float, default=1.5)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_contact_probe(
        step_seconds=args.step_seconds,
        duration_seconds=args.duration_seconds,
    )
    payload = (
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")
    if args.output is not None:
        atomic_write_bytes(args.output.resolve(), payload)
    print(payload.decode("utf-8"), end="")
    return 0 if result["status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
