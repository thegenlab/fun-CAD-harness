"""Cheap source-only grouped-manifest contradictions for the existing F7 task.

No receipt is granted; missing or ambiguous evidence remains the normal F7
evidence request. Only a declared current body manifest with explicit physical
part ownership is checked, without CAD execution or a fixed revision path.
"""
import csv
import json
from pathlib import Path


def grouped_manifest_findings(scene_path, criteria):
    from .static_delivery import is_static_delivery_criterion
    if not any(is_static_delivery_criterion(row) for row in criteria["criteria"]):
        return []
    scene_path=Path(scene_path).resolve()
    root=scene_path.parent
    scene=json.loads(scene_path.read_text(encoding="utf-8-sig"))
    bodies={row["id"]:row for row in scene["bodies"]}
    declared=list(scene.get("authoring_sources",[]))
    candidates=[]
    for relative in declared:
        if Path(relative).name!="body_manifest.json":continue
        path=(root/relative).resolve()
        if not path.is_file():continue
        try:
            rows=json.loads(path.read_text(encoding="utf-8-sig"))
            if not isinstance(rows,list):continue
            manifest={row["id"]:row for row in rows}
            for row in rows:int(row["solids"])
        except (ValueError,KeyError,TypeError):continue
        if len(manifest)!=len(rows) or set(manifest)!=set(bodies):continue
        if all(manifest[key].get("source")==body["source"] for key,body in bodies.items()):
            candidates.append((path,manifest))
    # No guessed latest revision and no design failure for missing evidence.
    if len(candidates)!=1:return []
    manifest_path,manifest=candidates[0]
    csv_candidates=[]
    for relative in declared:
        if Path(relative).suffix.lower()!=".csv":continue
        parts_path=(root/relative).resolve()
        if not parts_path.is_file():continue
        with parts_path.open(encoding="utf-8-sig",newline="") as stream:
            reader=csv.DictReader(stream)
            if "rigid_body" not in (reader.fieldnames or []):continue
            parts=list(reader)
        try:
            if len({row["id"] for row in parts})!=len(parts):continue
            owners={row.get("rigid_body") or row["id"] for row in parts}
            for row in parts:int(row["solids"])
        except (ValueError,KeyError,TypeError):continue
        if owners!=set(bodies):continue
        if not any((row.get("rigid_body") or row["id"])!=row["id"] for row in parts):continue
        csv_candidates.append((parts_path,parts))
    if len(csv_candidates)!=1:return []
    parts_path,parts=csv_candidates[0]
    counts={key:sum(int(row["solids"]) for row in parts if (row.get("rigid_body") or row["id"])==key) for key in bodies}
    findings=[]
    for key,count in counts.items():
        actual=int(manifest[key]["solids"])
        if count!=actual:
            findings.append({"kind":"source_delivery_metadata_inconsistent","criterion_id":"F7","body_id":key,
                "manifest_path":str(manifest_path),"manifest_source_solid_count":actual,
                "parts_csv_path":str(parts_path),"physical_part_ids":[row["id"] for row in parts if (row.get("rigid_body") or row["id"])==key],
                "physical_parts_declared_solid_count":count,
                "observed_fact":f"The current source body manifest declares {actual} solids for {key}; its explicitly owned physical parts declare {count} in total.",
                "required_source_action":"Correct the author-owned reproducible manifest generator and corresponding delivery without changing geometry, nominal mass-model scope or frozen criteria.",
                "scope":"Source metadata contradiction only. No CAD rebuilt, no geometry/physical verdict and no F7 execution receipt."})
    return findings
