"""Explicit scope of original rigid geometry and declared equivalent elasticity.

The compiler owns model approval and provenance. This small shared contract
prevents reduced-model dynamics, geometry, replay and publication from claiming
that unsolved elastic source surfaces were checked as a deformed continuum.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path, PureWindowsPath


SCHEMA = "slim_validation_scope_v1"
EQUIVALENT_DISPLAY_NAME = "等效弹性模型下的功能验证"
EXCLUDED_CLAIMS = [
    "Deformed original spring surfaces and winding/self-contact clearance are not solved or certified.",
    "Local stress, fatigue life and strength are not established by the equivalent model.",
    "Physical clamps, leaves and slotted sleeves achieving the declared preload are not established.",
    "Uncalibrated geometry/material estimates are not experimental or continuum calibration.",
]


class ValidationScopeError(ValueError):
    pass


def _source_path(value):
    # Scope signatures describe the producing host's absolute path labels.
    # Do not reinterpret Windows drive paths as POSIX-relative paths on import.
    text = str(value)
    if PureWindowsPath(text).is_absolute():
        return str(PureWindowsPath(text))
    if text.startswith('/'):
        return text
    return str(Path(value).resolve())


def _signature(value):
    payload = {key: item for key, item in value.items() if key != "signature"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def _numbers(value, count, label):
    if (not isinstance(value, (list, tuple)) or len(value) != count
            or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in value)):
        raise ValidationScopeError(label + " requires finite numbers")
    return [float(x) for x in value]


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(x in "0123456789abcdef" for x in value)


def _inventory(bodies):
    result = {}
    for body in bodies:
        key = body["id"]
        if not isinstance(key, str) or not key or key in result:
            raise ValidationScopeError("Scope requires a complete unique body inventory")
        source = _source_path(body["source_file"])
        parts = body.get("source_solid_ids")
        if not _digest(body.get("source_sha256")) or not parts or len(set(parts)) != len(parts):
            raise ValidationScopeError("Scope requires source-bound original parts: " + key)
        result[key] = {"source_file": source, "source_sha256": body["source_sha256"],
                       "source_solid_ids": sorted(parts)}
    if not result:
        raise ValidationScopeError("Scope body inventory is empty")
    return dict(sorted(result.items()))


def _regions(rows, ids, references, bindings):
    result, identifiers = [], set()
    for raw in rows:
        row = copy.deepcopy(raw)
        key, pair, owner = row.get("id"), row.get("body_pair"), row.get("region_owner_body")
        if not isinstance(key, str) or not key or key in identifiers:
            raise ValidationScopeError("Flexible contact regions need unique IDs")
        identifiers.add(key)
        if (not isinstance(pair, list) or len(pair) != 2 or len(set(pair)) != 2
                or not set(pair) <= ids or set(pair).intersection(references) or owner not in pair):
            raise ValidationScopeError("A flexible region needs two retained bodies and a pair-local owner: " + key)
        if row.get("shape") != "cylinder":
            raise ValidationScopeError("Unsupported flexible contact region shape: " + key)
        origin = _numbers(row.get("axis_origin_source_m"), 3, "Region origin")
        axis = _numbers(row.get("axis_direction_source"), 3, "Region axis")
        norm = math.sqrt(sum(x*x for x in axis))
        interval = _numbers(row.get("axial_interval_m"), 2, "Region axial interval")
        radius = _numbers([row.get("radius_m")], 1, "Region radius")[0]
        if norm < 1e-12 or radius <= 0 or interval[1] <= interval[0]:
            raise ValidationScopeError("Flexible contact region must be finite and nonempty: " + key)
        refs = row.get("source_refs")
        if not isinstance(refs, list) or not refs:
            raise ValidationScopeError("Flexible region lacks source evidence: " + key)
        for ref in refs:
            path = _source_path(ref['path'])
            if not ref.get("locator") or bindings.get(path) != ref.get("sha256"):
                raise ValidationScopeError("Flexible region evidence is not source-bound: " + key)
            ref["path"] = path
        row.update(body_pair=sorted(pair), axis_origin_source_m=origin,
                   axis_direction_source=[x/norm for x in axis] if abs(norm-1.) > 1e-14 else axis,
                   axial_interval_m=interval, radius_m=radius)
        result.append(row)
    return sorted(result, key=lambda row: row["id"])


def make_validation_scope(bodies, *, elastic_reference_body_ids=(), flexible_contact_regions=(),
                          source_bindings=None, model_binding=None):
    inventory = _inventory(bodies)
    ids, references = set(inventory), list(elastic_reference_body_ids)
    if len(set(references)) != len(references) or not set(references) <= ids:
        raise ValidationScopeError("Elastic references must be unique source body IDs")
    if not ids-set(references):
        raise ValidationScopeError("Equivalent elasticity cannot remove the whole rigid mechanism")
    bindings = {_source_path(path): value for path, value in (source_bindings or {}).items()}
    for body in inventory.values():
        path, expected = body["source_file"], body["source_sha256"]
        if path in bindings and bindings[path] != expected:
            raise ValidationScopeError("Conflicting source bindings for " + path)
        bindings[path] = expected
    if any(not _digest(value) for value in bindings.values()):
        raise ValidationScopeError("Scope requires actual source file digests")
    regions = _regions(flexible_contact_regions, ids, set(references), bindings)
    equivalent = bool(references or regions or model_binding)
    if equivalent and not model_binding:
        raise ValidationScopeError("Equivalent validation requires the executed elastic model binding")
    value = {
        "schema_version": SCHEMA,
        "mode": "equivalent_elastic" if equivalent else "rigid_original",
        "display_name": EQUIVALENT_DISPLAY_NAME if equivalent else "原始刚体几何下的功能验证",
        "elastic_reference_body_ids": sorted(references),
        "rigid_body_ids": sorted(ids-set(references)),
        "flexible_contact_regions": regions,
        "body_inventory": inventory,
        "source_bindings": dict(sorted(bindings.items())),
        "model_binding": copy.deepcopy(model_binding or {}),
        "initial_geometry": "All delivered original bodies at the source initial pose.",
        "dynamic_geometry": ("Original retained rigid bodies; declared elastic references "
                             "and only the listed pair-local flexible contact regions are excluded."
                             if equivalent else "All original rigid bodies. Recorded poses remain complete; audit sampling is disclosed separately."),
        "contact_evidence": "Actual solver contact records remain separate from geometry scope exclusions.",
        "excluded_claims": list(EXCLUDED_CLAIMS) if equivalent else [],
    }
    value["signature"] = _signature(value)
    return value


def validate_validation_scope(scope, bodies=None):
    try:
        if not isinstance(scope, dict) or scope.get("schema_version") != SCHEMA or scope.get("signature") != _signature(scope):
            raise ValidationScopeError("Missing or changed validation scope")
        records = bodies or [dict(id=key, **body) for key, body in scope["body_inventory"].items()]
        expected = make_validation_scope(records,
            elastic_reference_body_ids=scope["elastic_reference_body_ids"],
            flexible_contact_regions=scope["flexible_contact_regions"],
            source_bindings=scope["source_bindings"], model_binding=scope["model_binding"])
        if expected != scope:
            raise ValidationScopeError("Validation scope does not match original inventory and declared limits")
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, ValidationScopeError):
            raise
        raise ValidationScopeError("Invalid validation scope: " + str(exc)) from exc
    return scope


def flexible_regions_for_pair(scope, body_pair):
    pair = sorted(body_pair)
    return [row for row in scope.get("flexible_contact_regions", []) if row["body_pair"] == pair]


def point_in_flexible_contact_region(scope, body_pair, point_abs_m, body_poses, source_positions_m):
    """Classify an actual contact point against its finite, moving source region.

    The caller decides collision treatment. A true result never removes a
    retained contact record or exempts any other pair/point from geometry audit.
    """
    import numpy as np
    point = np.asarray(point_abs_m, dtype=float)
    for row in flexible_regions_for_pair(scope, body_pair):
        owner = row["region_owner_body"]
        pose = body_poses[owner]
        quaternion = np.asarray(pose["rotation_abs_quat_wxyz"], dtype=float)
        quaternion /= np.linalg.norm(quaternion)
        w, x, y, z = quaternion
        rotation = np.asarray([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                               [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                               [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
        source_point = rotation.T@(point-np.asarray(pose["position_abs_m"])) + np.asarray(source_positions_m[owner])
        delta = source_point-np.asarray(row["axis_origin_source_m"])
        axis = np.asarray(row["axis_direction_source"])
        along = float(delta@axis)
        radial = delta-along*axis
        if row["axial_interval_m"][0] <= along <= row["axial_interval_m"][1] and float(radial@radial) <= row["radius_m"]**2:
            return True
    return False


def validate_result_scope(value, scene):
    """Require the same explicit scope in the result, trace summaries and audit."""
    scope = value.get("validation_scope")
    if scope is None:
        if scene.get("equivalent_elasticity"):
            raise ValidationScopeError("Equivalent scene has no published validation scope")
        return None
    validate_validation_scope(scope)
    if set(scope["body_inventory"]) != {body["id"] for body in scene["bodies"]}:
        raise ValidationScopeError("Published scope does not cover the actual scene inventory")
    for test in value.get("tests", []):
        audit = test.get("geometry_audit", {})
        if test.get("validation_scope") != scope or audit.get("validation_scope") != scope:
            raise ValidationScopeError("A physical test or its geometry audit used a different scope")
        if scope["mode"] == "equivalent_elastic" and audit.get("schema_version") != "slim_geometry_audit_v2":
            raise ValidationScopeError("Reduced geometry evidence is mislabeled as complete original geometry")
    return scope
