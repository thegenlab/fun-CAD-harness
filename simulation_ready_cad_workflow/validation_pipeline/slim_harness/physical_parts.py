"""Named physical assembly inventory; not len(scene.bodies)."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


PART_INVENTORY_REQUIRED = "PART_INVENTORY_REQUIRED"
_COUPON_CATEGORIES = {"coupon", "particle", "test_particle"}


def load_project_scope(project) -> dict[str, Any] | None:
    path = Path(project) / "scope.json"
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def is_time_only_continuation(project, tier=None) -> bool:
    """Existing printable-machine continuations keep original tests/parts.

    Ordinary medium still means 2 tests / 15 parts. Time-only medium uses the
    project's own scope.json or a frozen custom tier, not those ordinary caps.
    """
    scope = load_project_scope(project)
    if scope and scope.get("medium_time_only") is True:
        return True
    name = getattr(tier, "name", "") or ""
    return "original_scope_medium_time" in name


def _row_is_coupon(row: dict[str, Any]) -> bool:
    category = str(row.get("category") or "").strip().lower()
    return category in _COUPON_CATEGORIES


def _unique_names(rows: list[dict[str, Any]], key: str) -> list[str]:
    issues = []
    seen: set[str] = set()
    for row in rows:
        name = row.get(key)
        if not isinstance(name, str) or not name.strip():
            issues.append("Physical part rows require a unique non-empty name")
            continue
        if name in seen:
            issues.append(f"Duplicate physical part name {name!r}")
        seen.add(name)
    return issues


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _scene_bodies(scene: dict[str, Any]) -> dict[str, dict[str, Any]]:
    bodies: dict[str, dict[str, Any]] = {}
    for row in scene.get("bodies") or []:
        if isinstance(row, dict) and isinstance(row.get("id"), str) and row["id"]:
            bodies[row["id"]] = row
    return bodies


def _declared_source(body: dict[str, Any] | None) -> str | None:
    if not body:
        return None
    for key in ("source", "source_file"):
        value = body.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _same_source(left: str, right: str) -> bool:
    return Path(left).as_posix() == Path(right).as_posix()


def _source_available(root: Path, declared: str) -> bool:
    path = Path(declared)
    if path.is_absolute():
        return path.is_file()
    try:
        candidate = (root / declared).resolve()
        return candidate.is_file() and candidate.is_relative_to(root.resolve())
    except (OSError, ValueError):
        return False


def _available_solid_ids(body: dict[str, Any] | None) -> set[str]:
    if not body:
        return set()
    ids: set[str] = set()
    raw = body.get("source_solid_ids")
    if isinstance(raw, list):
        ids.update(item for item in raw if isinstance(item, str) and item)
    components = body.get("source_components")
    if isinstance(components, list):
        for item in components:
            if isinstance(item, dict) and isinstance(item.get("solid_id"), str) and item["solid_id"]:
                ids.add(item["solid_id"])
    return ids


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return None
    return value


def _instance_quantity(value: Any, *, default: int | None = 1) -> int | None:
    if value is None:
        return default
    return _positive_int(value)


def _instance_count(rows: list[dict[str, Any]]) -> int:
    total = 0
    for row in rows:
        quantity = _instance_quantity(row.get("quantity"), default=1)
        total += quantity if quantity is not None else 0
    return total


def _parse_source_member(part_name: str, member: Any) -> str | None:
    if not isinstance(member, str) or not member:
        return None
    prefix = part_name + ":"
    if not member.startswith(prefix):
        return None
    solid = member[len(prefix):]
    if not solid.startswith("solid_"):
        return None
    suffix = solid[6:]
    if not suffix.isdigit() or int(suffix) < 1:
        return None
    return solid


def _validate_source_members(
    part_name: str,
    row: dict[str, Any],
    body: dict[str, Any] | None,
    issues: list[str],
) -> None:
    members = row.get("source_members")
    count = row.get("source_solid_count")
    if members is None and count is None:
        return
    parsed: list[str] = []
    if members is not None:
        if not isinstance(members, list) or not members:
            issues.append(f"{part_name}: source_members must be a nonempty list")
        else:
            seen: set[str] = set()
            for member in members:
                solid = _parse_source_member(part_name, member)
                if solid is None:
                    issues.append(
                        f"{part_name}: nonexistent source_member {member!r}"
                    )
                    continue
                if member in seen:
                    issues.append(f"{part_name}: duplicate source_member {member!r}")
                    continue
                seen.add(member)
                parsed.append(solid)
            if parsed and len(parsed) == len(members):
                expected = [f"solid_{index}" for index in range(1, len(parsed) + 1)]
                if parsed != expected:
                    issues.append(
                        f"{part_name}: nonexistent source_member; declared "
                        f"{members!r} is not {part_name}:solid_1..{len(parsed)}"
                    )
            available = _available_solid_ids(body)
            if available:
                missing = [solid for solid in parsed if solid not in available]
                if missing:
                    issues.append(
                        f"{part_name}: source_members not present in available "
                        f"source metadata: {', '.join(missing)}"
                    )
    bound = _positive_int(count) if count is not None else None
    if count is not None and bound is None:
        issues.append(f"{part_name}: source_solid_count must be a positive integer")
    elif bound is not None and members is not None and len(members) != bound:
        issues.append(
            f"{part_name}: source_solid_count {bound} does not match "
            f"{len(members)} source_members"
        )


def _validate_body_source(
    root: Path,
    body_id: str,
    declared: Any,
    scene_body: dict[str, Any] | None,
    issues: list[str],
) -> None:
    if not isinstance(declared, str) or not declared.strip():
        issues.append(f"body_sources omits scene body {body_id!r}")
        return
    declared = declared.strip()
    scene_source = _declared_source(scene_body)
    if scene_source and not _same_source(declared, scene_source):
        issues.append(
            f"body_sources[{body_id!r}] is {declared!r} but scene source is {scene_source!r}"
        )
    if not _source_available(root, declared):
        issues.append(f"body_sources[{body_id!r}] file {declared!r} is not available")


def _required(basis: str, rows: list[dict[str, Any]], issues: list[str], **extra: Any) -> dict[str, Any]:
    payload = {
        "status": PART_INVENTORY_REQUIRED,
        "basis": basis,
        "parts": rows,
        "issues": issues,
        "count": 0,
        "machine_part_count": 0,
        "excluded_coupon_count": 0,
    }
    payload.update(extra)
    return payload


def _parts_from_asset(path: Path, scene: dict[str, Any], root: Path) -> dict[str, Any] | None:
    value = _load_json(path)
    if not isinstance(value, dict) or not isinstance(value.get("parts"), list):
        return None
    rows = []
    for item in value["parts"]:
        if not isinstance(item, dict):
            return None
        name = item.get("name")
        body = item.get("body")
        if not isinstance(name, str) or not name or not isinstance(body, str) or not body:
            return None
        quantity = item.get("quantity")
        rows.append({
            "name": name,
            "body": body,
            "role": item.get("role"),
            "category": item.get("category") or "machine",
            "quantity": quantity,
            "source_solid_count": item.get("source_solid_count"),
            "source_members": item.get("source_members"),
        })
    bodies = _scene_bodies(scene)
    sources = value.get("body_sources") if isinstance(value.get("body_sources"), dict) else {}
    issues = _unique_names(rows, "name")
    covered: set[str] = set()
    for row in rows:
        quantity = _instance_quantity(row.get("quantity"), default=1)
        if quantity is None:
            issues.append(
                f"{row['name']}: quantity must be a positive integer physical instance count"
            )
        else:
            row["quantity"] = quantity
        body_id = row["body"]
        scene_body = bodies.get(body_id)
        if scene_body is None:
            issues.append(
                f"Named parts reference bodies absent from the scene: {body_id}"
            )
        else:
            covered.add(body_id)
        _validate_source_members(row["name"], row, scene_body, issues)
    omitted = sorted(set(bodies) - covered)
    if omitted:
        issues.append("Named parts omit scene bodies: " + ", ".join(omitted))
    if not sources:
        issues.append("parts.json requires body_sources covering every scene body")
    else:
        for body_id, scene_body in bodies.items():
            _validate_body_source(root, body_id, sources.get(body_id), scene_body, issues)
    if issues:
        return _required("parts.json", rows, issues, body_sources=sources)
    return {"status": "ready", "basis": "parts.json", "parts": rows, "issues": [],
            "body_sources": sources}


def _parts_from_print_inventory(
    path: Path, scene: dict[str, Any], root: Path,
) -> dict[str, Any] | None:
    value = _load_json(path)
    if not isinstance(value, dict) or not isinstance(value.get("parts"), list):
        return None
    rows = []
    for item in value["parts"]:
        if not isinstance(item, dict):
            return None
        name = item.get("id") or item.get("name")
        if not isinstance(name, str) or not name:
            return None
        body = item.get("body_id") if item.get("body_id") is not None else item.get("body")
        source = item.get("source") if isinstance(item.get("source"), str) and item.get("source") else None
        stl = item.get("stl") if isinstance(item.get("stl"), str) and item.get("stl") else None
        rows.append({
            "name": name,
            "body": body,
            "category": item.get("category") or "machine",
            "quantity": item.get("quantity", 1),
            "source": source,
            "stl": stl,
        })
    bodies = _scene_bodies(scene)
    issues = _unique_names(rows, "name")
    covered: set[str] = set()
    for row in rows:
        quantity = _instance_quantity(row.get("quantity"), default=1)
        if quantity is None:
            issues.append(
                f"{row['name']}: quantity must be a positive integer physical instance count"
            )
        else:
            row["quantity"] = quantity
        coupon = _row_is_coupon(row)
        body_id = row.get("body")
        if body_id is not None and not isinstance(body_id, str):
            issues.append(f"{row['name']}: body must be a string scene body id")
            body_id = None
        if isinstance(body_id, str) and body_id:
            if body_id not in bodies:
                issues.append(
                    f"{row['name']}: names a body absent from the scene: {body_id}"
                )
            else:
                covered.add(body_id)
                scene_source = _declared_source(bodies[body_id])
                source = row.get("source")
                if isinstance(source, str) and source and scene_source and not _same_source(source, scene_source):
                    issues.append(
                        f"{row['name']}: source {source!r} does not match scene "
                        f"source {scene_source!r}"
                    )
        elif not coupon:
            issues.append(f"{row['name']}: machine part requires a declared scene body")
        for field in ("source", "stl"):
            declared = row.get(field)
            if declared is None:
                continue
            if not isinstance(declared, str) or not declared.strip():
                issues.append(f"{row['name']}: {field} must be a nonempty path")
            elif not _source_available(root, declared):
                issues.append(f"{row['name']}: {field} {declared!r} is not available")
        if not coupon:
            source = row.get("source")
            stl = row.get("stl")
            if not (isinstance(source, str) and source) and not (isinstance(stl, str) and stl):
                issues.append(f"{row['name']}: machine part requires a declared source or stl")
    omitted = sorted(set(bodies) - covered)
    if omitted:
        issues.append("Named print parts omit scene bodies: " + ", ".join(omitted))
    if issues:
        return _required(path.name, rows, issues)
    return {"status": "ready", "basis": path.as_posix(), "parts": rows, "issues": []}


def _parts_from_grouped_csv(root: Path, scene: dict[str, Any]) -> dict[str, Any] | None:
    declared = list(scene.get("authoring_sources", []))
    bodies = _scene_bodies(scene)
    csv_paths = []
    for relative in declared:
        path = (root / relative).resolve()
        if path.suffix.lower() != ".csv" or not path.is_file() or not path.is_relative_to(root.resolve()):
            continue
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if "id" not in (reader.fieldnames or []):
                continue
            rows = list(reader)
        if not rows:
            continue
        csv_paths.append((path, rows))
    if len(csv_paths) != 1:
        return None
    path, raw_rows = csv_paths[0]
    rows = []
    for item in raw_rows:
        name = item.get("id")
        if not isinstance(name, str) or not name:
            return None
        rows.append({
            "name": name,
            "body": item.get("rigid_body") or name,
            "category": item.get("category") or "machine",
            "source": item.get("source"),
        })
    issues = _unique_names(rows, "name")
    owners = {row["body"] for row in rows}
    if bodies and owners != set(bodies) and not owners <= set(bodies):
        issues.append("Grouped physical-part CSV does not map onto scene bodies")
    if issues:
        return _required(path.name, rows, issues)
    return {"status": "ready", "basis": path.as_posix(), "parts": rows, "issues": []}


def _candidate_files(root: Path, scene: dict[str, Any]) -> list[Path]:
    names = []
    for relative in [*scene.get("authoring_sources", []), *scene.get("artifacts", [])]:
        if not isinstance(relative, str):
            continue
        path = (root / relative).resolve()
        if path.is_file() and path.is_relative_to(root.resolve()):
            names.append(path)
    for extra in (root / "parts.json", root / "all_print_inventory.json", root / "print" / "inventory.json"):
        if extra.is_file():
            names.append(extra.resolve())
    unique = []
    seen: set[Path] = set()
    for path in names:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    return unique


def resolve_physical_part_inventory(
    scene_path,
    scene: dict[str, Any],
    *,
    task_text: str = "",
    project=None,
) -> dict[str, Any]:
    """Return named independently made/purchased assembly parts, or a gap."""
    # task_text is accepted by the existing workflow call site. Coupons are
    # classified only from explicit inventory category, never from task prose.
    _ = task_text
    root = Path(scene_path).resolve().parent
    found = None
    for path in _candidate_files(root, scene):
        name = path.name.lower()
        parsed = None
        if name == "parts.json":
            parsed = _parts_from_asset(path, scene, root)
        elif name in {"all_print_inventory.json", "inventory.json"}:
            parsed = _parts_from_print_inventory(path, scene, root)
        if parsed is not None:
            found = parsed
            found["inventory_file"] = str(path)
            break
    if found is None:
        found = _parts_from_grouped_csv(root, scene)
        if found is not None:
            found["inventory_file"] = found.get("basis")
    if found is None:
        return {
            "status": PART_INVENTORY_REQUIRED,
            "basis": None,
            "parts": [],
            "count": 0,
            "machine_part_count": 0,
            "excluded_coupon_count": 0,
            "purchased_parts_policy": "unspecified",
            "issues": [PART_INVENTORY_REQUIRED + ": no source-bound named physical-part inventory"],
        }
    if found.get("status") == PART_INVENTORY_REQUIRED:
        found.setdefault("count", 0)
        found.setdefault("machine_part_count", 0)
        found.setdefault("excluded_coupon_count", 0)
        found.setdefault("purchased_parts_policy", "unspecified")
        return found
    machine = []
    excluded = []
    for row in found["parts"]:
        if _row_is_coupon(row):
            excluded.append(row)
        else:
            machine.append(row)
    found.update(
        status="ready",
        parts=machine,
        excluded_coupons=excluded,
        count=_instance_count(machine),
        machine_part_count=_instance_count(machine),
        excluded_coupon_count=_instance_count(excluded),
        purchased_parts_policy="unspecified",
        counted_as="independently_made_or_purchased_assembly_parts",
    )
    if project is not None:
        found["time_only_continuation"] = is_time_only_continuation(project)
    return found


def check_physical_part_inventory(inventory: dict[str, Any], tier, *, project=None) -> list[str]:
    """Ordinary tiers enforce max named parts. Time-only continuations do not."""
    if inventory.get("status") == PART_INVENTORY_REQUIRED:
        return list(inventory.get("issues") or [PART_INVENTORY_REQUIRED])
    if project is not None and is_time_only_continuation(project, tier):
        return []
    if tier is None:
        return []
    count = int(inventory.get("machine_part_count", inventory.get("count", 0)))
    if count > tier.max_parts:
        return [
            f"Named physical-part inventory has {count} assembly parts but tier "
            f"{tier.name!r} allows at most {tier.max_parts}"
        ]
    return []
