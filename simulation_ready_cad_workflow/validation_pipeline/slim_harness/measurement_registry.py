"""Explicit project-owned clock measurement bindings and cache adoption.

Registration is a Harness action. Scene test IDs or author scripts never enable
these measurements on their own. This module does not execute author code.
"""
from __future__ import annotations

import hashlib
import json
import copy
from pathlib import Path

from . import alarm_measurements as kernel
from .clock_contacts.calibration_binding import verify_saved_bundle, CalibrationBindingError

REGISTRATION_FILE = "measurement-registration.json"
MeasurementCapabilityError = kernel.MeasurementCapabilityError


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def value_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False).encode("utf-8")).hexdigest()


def compiled_test_spec(test, units):
    """The only test-unit conversion performed by contracts.compile_scene."""
    result = copy.deepcopy(test)
    scale = .001 if units == "mm" else 1.
    for load in result.get("loads", []):
        if "point" in load:
            load["point"] = [coordinate * scale for coordinate in load["point"]]
    return result


def validate_runtime_registration(binding, test):
    if (binding.get("signature") != kernel.registration_signature(binding)
            or binding.get("implementation_sha256") != kernel.implementation_sha256()
            or binding.get("test_id") != test["id"]
            or binding.get("test_spec_sha256") != value_digest(test)):
        raise MeasurementCapabilityError("Measurement binding does not match this implementation and exact test")
    profile = kernel.measurement_profile_for_registration(binding, test)
    for field in ("duration_s", "step_s"):
        if test.get(field) != profile[field] or binding.get(field) != test.get(field):
            raise MeasurementCapabilityError("Registered test changes the reviewed clock " + field)


def load_registered_measurements(project, scene_path):
    """Verify the actual task, plan, test and calibrated collector before use."""
    project, scene_path = Path(project).resolve(), Path(scene_path).resolve()
    path = project / REGISTRATION_FILE
    if not path.exists():
        return {}
    try:
        registry = read(path)
        if (registry["schema_version"] != "slim_project_measurement_registration_v1"
                or registry["criteria_sha256"] != digest(project / "criteria.json")
                or registry["task_sha256"] != digest(project / "task.md")):
            raise ValueError("The frozen task or criteria no longer match the registration")
        scene = read(scene_path)
        tests = {test["id"]: test for test in scene["tests"]}
        result = {}
        for binding in registry["bindings"]:
            test_id = binding["test_id"]
            if test_id in result or test_id not in tests:
                raise ValueError("Duplicate or missing registered test")
            validate_runtime_registration(binding, compiled_test_spec(tests[test_id], scene["units"]))
            if binding["source_test_spec_sha256"] != value_digest(tests[test_id]):
                raise ValueError("The registered source test changed")
            if (binding["criteria_sha256"] != registry["criteria_sha256"]
                    or binding["task_sha256"] != registry["task_sha256"]
                    or not binding["source_plan_bindings"]):
                raise ValueError("Measurement plan has no matching frozen requirements")
            for relative, expected in binding["source_plan_bindings"].items():
                source = (scene_path.parent / relative).resolve()
                if not source.is_relative_to(scene_path.parent) or digest(source) != expected:
                    raise ValueError("Measurement plan or requirements changed: " + relative)
            for relative, expected in binding.get("measurement_policy_bindings", {}).items():
                policy = (project / relative).resolve()
                if not policy.is_relative_to(project) or digest(policy) != expected:
                    raise ValueError("Independently reviewed measurement policy changed: " + relative)
            if test_id == "automatic_alarm_and_stop":
                bundle = verify_saved_bundle(binding["contact_calibration_bundle"], kernel.__file__)
                if bundle["signature"] != binding["contact_calibration_signature"]:
                    raise ValueError("Registered contact calibration changed")
            result[test_id] = binding
        if not result:
            raise ValueError("A registration must contain the reviewed measurements")
        return result
    except (OSError, ValueError, KeyError, TypeError, CalibrationBindingError) as exc:
        raise MeasurementCapabilityError("MEASUREMENT_REGISTRATION_ERROR: " + str(exc)) from exc


def evaluate_native_trace(trace, binding):
    """Evaluate once while retaining the original native rejection/ assertions."""
    if trace.get("registered_measurements") is not None:
        validate_trace_measurements(trace, binding)
        return {}
    original = {"native_assertions": list(trace["assertions"]),
                "native_physical_pass": trace["physical_pass"]}
    result = kernel.evaluate_registered_trace(trace, binding)
    return {**original, **result}


def validate_trace_measurements(trace, binding):
    """Common fresh, raw-cache and validated-bundle acceptance guard."""
    if binding is None:
        return
    if (trace.get("measurement_error")
            or trace.get("measurement_registration_signature") != binding["signature"]
            or binding["implementation_sha256"] != kernel.implementation_sha256()):
        raise MeasurementCapabilityError("Saved trace lacks the current completed measurement binding")
    if binding["test_id"] == "automatic_alarm_and_stop":
        if trace.get("measurement_contact_schema", {}).get("calibration_signature") != binding["contact_calibration_signature"]:
            raise MeasurementCapabilityError("Saved trace lacks the calibrated contact evidence")
    supplemental = trace.get("registered_measurements", {})
    native = trace.get("native_assertions")
    if not isinstance(native, list) or not native or not isinstance(trace.get("native_physical_pass"), bool):
        raise MeasurementCapabilityError("Saved trace did not retain its native physical assertions")
    if trace.get("failure") is not None:
        if (supplemental.get("status") != "not_evaluated_after_native_failure"
                or trace.get("physical_pass") is not False or trace.get("assertions") != native):
            raise MeasurementCapabilityError("Native failure was not preserved")
        return
    if (supplemental.get("status") != "evaluated"
            or supplemental.get("registration_id") != binding["id"]
            or supplemental.get("test_id") != binding["test_id"]
            or not supplemental.get("assertions")):
        raise MeasurementCapabilityError("Registered measurement did not finish on the saved trace")
    assertions = [*native, *supplemental["assertions"]]
    expected_pass = trace["native_physical_pass"] and all(row.get("passed") is True for row in assertions)
    if trace.get("assertions") != assertions or trace.get("physical_pass") is not expected_pass:
        raise MeasurementCapabilityError("Saved result changed or omitted native/supplemental assertions")
