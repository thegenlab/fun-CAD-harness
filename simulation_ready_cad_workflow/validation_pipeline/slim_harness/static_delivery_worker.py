"""Isolated real CAD import and manifest checks. No author test output is trusted."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import sys


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def facts(solid):
    box = solid.bounding_box()
    values = {"bounds_mm": [*box.min, *box.max], "volume_mm3": solid.volume, "area_mm2": solid.area,
              "center_mm": list(solid.center()), "faces": len(solid.faces()), "edges": len(solid.edges())}
    if not solid.is_valid or values["volume_mm3"] <= 0:
        raise ValueError("Imported STEP contains invalid or nonpositive-volume solid")
    return values


def same_inventory_fact(left, right):
    if any(left[key] != right[key] for key in ("faces", "edges")):
        return False
    for key in ("bounds_mm", "center_mm", "volume_mm3", "area_mm2"):
        a, b = left[key], right[key]
        if not isinstance(a, list):
            a, b = [a], [b]
        if not all(math.isfinite(x) and math.isfinite(y) and math.isclose(x, y, rel_tol=1e-7, abs_tol=1e-5) for x, y in zip(a, b)):
            return False
    return True


def consume_matches(expected, actual):
    remaining = list(range(len(actual)))
    pairs = []
    for row in expected:
        candidates = [index for index in remaining if same_inventory_fact(row, actual[index])]
        if not candidates:
            raise ValueError("Complete assembly has no matching leaf for an expected source solid")
        index = candidates[0]
        remaining.remove(index)
        pairs.append({"assembly_solid_index_1based": index + 1, "equivalent_inventory_candidates": [n + 1 for n in candidates]})
    if remaining:
        raise ValueError("Complete assembly includes extra unmatched solids: " + str(remaining))
    return pairs


def unique(rows, label):
    values = {row["id"]: row for row in rows}
    if len(values) != len(rows):
        raise ValueError(label + " contains duplicate IDs")
    return values


def physical_part_groups(root, request, bodies, manifest):
    with (root / request["parts_csv"]).open(encoding="utf-8-sig", newline="") as stream:
        parts = unique(list(csv.DictReader(stream)), "parts CSV")
    groups = {}
    for part in parts.values():
        owner = part.get("rigid_body") or part["id"]
        if owner not in bodies:
            raise ValueError("Physical part names an absent rigid body: " + part["id"])
        if part["source"] not in request["source_files"]:
            raise ValueError("Physical part source is not in the bound snapshot: " + part["id"])
        groups.setdefault(owner, []).append(part)
    if set(groups) != set(bodies):
        raise ValueError("Physical parts do not cover the complete scene body inventory")
    for key, rows in groups.items():
        total = sum(int(row["solids"]) for row in rows)
        if total != int(manifest[key]["solids"]):
            raise ValueError("Physical part/manifest solid count disagrees for " + key +
                             ": parts=" + str(total) + ", manifest=" + str(manifest[key]["solids"]))
        if len(rows) == 1 and rows[0]["source"] != bodies[key]["source"]:
            raise ValueError("Part source mapping disagrees for " + key)
    return groups


def metadata(root, request):
    scene = read(root / "scene.json")
    bodies = unique(scene["bodies"], "scene bodies")
    joints = unique(scene["joints"], "scene joints")
    manifest = unique(read(root / request["body_manifest"]), "body manifest")
    connections = read(root / request["joint_manifest"])
    if set(bodies) != set(manifest):
        raise ValueError("Scene and body manifest IDs disagree")
    physical_part_groups(root, request, bodies, manifest)
    if unique(connections, "joint manifest") != joints:
        raise ValueError("Scene and complete connection manifest disagree")
    for key, body in bodies.items():
        if body["source"] != manifest[key]["source"]:
            raise ValueError("Body source mapping disagrees for " + key)
    for joint in joints.values():
        if not {joint["body_a"], joint["body_b"]} <= set(bodies):
            raise ValueError("Connection references a missing body")
    for path in request["documents"]:
        if not (root / path).read_text(encoding="utf-8-sig").strip():
            raise ValueError("Empty required document: " + path)
    return scene, bodies, manifest


def run(payload):
    import build123d as b
    import OCP

    request = payload["request"]
    original, rebuilt = Path(payload["design"]), Path(payload["rebuilt_design"])
    scene, bodies, manifest = metadata(original, request)
    rebuilt_scene, rebuilt_bodies, rebuilt_manifest = metadata(rebuilt, request)
    if scene != rebuilt_scene:
        raise ValueError("Executing the delivered source does not reproduce the current scene/connection/test metadata")
    original_groups = physical_part_groups(original, request, bodies, manifest)
    rebuilt_groups = physical_part_groups(rebuilt, request, rebuilt_bodies, rebuilt_manifest)
    if original_groups != rebuilt_groups:
        raise ValueError("Executing the source changed physical part ownership or source metadata")
    expected_ids = set(bodies) - set(request["excluded_context_body_ids"])
    if not expected_ids or not set(request["excluded_context_body_ids"]) <= set(bodies):
        raise ValueError("Assembly inclusion contract names unknown bodies")
    result = {"errors": [], "body_correspondence": [], "excluded_context_body_ids": request["excluded_context_body_ids"],
              "metadata": {"body_ids": list(bodies), "joint_ids": [j["id"] for j in scene["joints"]], "current_scene_reproduced": True},
              "scope": "All original and rebuilt full assemblies opened with the CAD kernel. Each source solid is accounted for by bounds, volume, area, center, face and edge inventory within stated tolerances. This is inventory correspondence, not a Boolean proof of shape equivalence or documentation semantics.",
              "inventory_tolerance": {"relative": 1e-7, "absolute": 1e-5}}
    assemblies = {}
    for key, root in (("original_assembly", original), ("rebuilt_assembly", rebuilt)):
        shape = b.import_step(root / request["assembly"])
        solids = list(shape.solids())
        if not solids:
            raise ValueError("Full STEP assembly contains no solids")
        assemblies[key] = [facts(solid) for solid in solids]
        result[key] = {"path": str(root / request["assembly"]), "sha256": sha(root / request["assembly"]),
                       "solid_count": len(solids), "top_label": shape.label,
                       "children": [{"label": child.label, "solid_count": len(child.solids())} for child in shape.children],
                       "solids": assemblies[key]}
        del solids, shape
    expected = []
    spans = []
    for key, body in bodies.items():
        original_part = b.import_step(original / body["source"])
        body_facts = [facts(solid) for solid in original_part.solids()]
        if len(body_facts) != int(manifest[key]["solids"]):
            raise ValueError("Imported source solid count disagrees with manifest: " + key)
        fresh_part = b.import_step(rebuilt / rebuilt_bodies[key]["source"])
        fresh_facts = [facts(solid) for solid in fresh_part.solids()]
        consume_matches(body_facts, fresh_facts)
        if len(fresh_facts) != int(rebuilt_manifest[key]["solids"]):
            raise ValueError("Rebuilt part solid count disagrees with rebuilt manifest: " + key)
        # Multiple physical parts may belong to one declared rigid body. Check
        # their actual STEP solids, not just an owner string or summed count.
        if len(original_groups[key]) > 1:
            for root, groups, expected_facts in ((original, original_groups, body_facts),
                                                (rebuilt, rebuilt_groups, fresh_facts)):
                physical_facts = []
                for part in groups[key]:
                    source_part = b.import_step(root / part["source"])
                    inventory = [facts(solid) for solid in source_part.solids()]
                    if len(inventory) != int(part["solids"]):
                        raise ValueError("Imported physical part count disagrees for " + part["id"])
                    physical_facts.extend(inventory)
                    del source_part
                consume_matches(physical_facts, expected_facts)
        if key in expected_ids:
            spans.append((key, len(expected), len(body_facts)))
            expected.extend(body_facts)
        del original_part, fresh_part
    mapping = consume_matches(expected, assemblies["original_assembly"])
    rebuilt_mapping = consume_matches(expected, assemblies["rebuilt_assembly"])
    for key, start, count in spans:
        result["body_correspondence"].append({"body_id": key, "source": bodies[key]["source"], "source_solid_count": count,
             "physical_part_ids": [row["id"] for row in original_groups[key]],
             "original_assembly_matches": mapping[start:start + count], "rebuilt_assembly_matches": rebuilt_mapping[start:start + count]})
    runtime_paths = {str(Path(sys.executable).resolve())}
    for name, module in tuple(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if path and (name == "build123d" or name.startswith(("build123d.", "OCP"))) and Path(path).is_file():
            runtime_paths.add(str(Path(path).resolve()))
    result["runtime_bindings"] = {path: sha(path) for path in sorted(runtime_paths)}
    result["runtime_versions"] = {"python": sys.version, "build123d": b.__version__, "OCP": getattr(OCP, "__version__", "unknown")}
    return result


def main():
    payload = read(sys.argv[1])
    try:
        result = run(payload)
    except Exception as exc:
        result = {"errors": [f"{type(exc).__name__}: {exc}"]}
    Path(payload["output"]).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"errors": result["errors"]}, ensure_ascii=False))
    return 1 if result["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
