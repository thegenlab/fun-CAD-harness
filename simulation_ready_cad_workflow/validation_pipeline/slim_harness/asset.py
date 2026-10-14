"""Named source parts for Slim, adapted from the earlier Asset Builder prototype.

The author names parts and rigid bodies. One STEP per body removes the need to
author transient solid_N mappings; the existing Slim intake owns its inventory.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import copy
import json
import runpy
import shutil
import sys


@dataclass
class Part:
    shape: Any
    body: str
    role: str = "dynamic"
    density_kg_m3: float = 7800.0
    color: list[float] | None = None
    source_solid_count: int | None = None
    source_members: list[str] | None = None


@dataclass
class Asset:
    parts: dict[str, Part]
    units: str = "mm"
    joints: list[dict] = field(default_factory=list)
    tests: list[dict] = field(default_factory=list)
    gravity_m_s2: list[float] = field(default_factory=lambda: [0, -9.81, 0])
    description: str = ""
    dependencies: list[str] = field(default_factory=list)
    equivalent_elasticity: dict | None = None


def load_asset(source: Path) -> Asset:
    sys.path.insert(0, str(source.parent))
    try:
        namespace = runpy.run_path(str(source))
        build = namespace.get("build")
        if not callable(build):
            raise ValueError("model.py must define build() returning a Slim Asset")
        asset = build()
    finally:
        sys.path.pop(0)
    if not isinstance(asset, Asset) or not asset.parts:
        raise ValueError("build() must return a nonempty Slim Asset")
    return asset


def build_file(source: Path, output: Path) -> dict:
    from build123d import Compound, export_step
    source = Path(source).resolve()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if output/source.name != source: shutil.copy2(source,output/source.name)
    asset = load_asset(source)
    if asset.units not in {"m", "mm"}: raise ValueError("Slim scene units must be m or mm; CAD source remains millimetres")
    sources = []
    for relative in [source.name, *asset.dependencies]:
        path = (source.parent / relative).resolve()
        if Path(relative).is_absolute() or not path.is_relative_to(source.parent) or not path.is_file():
            raise ValueError(f"Local source dependency must be a file below the model directory: {relative}")
        target = output / relative
        if relative in {"scene.json", "parts.json"} or relative.startswith("body-"):
            raise ValueError(f"Source dependency conflicts with generated geometry: {relative}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.resolve() != path: shutil.copy2(path, target)
        sources.append(relative)
    groups = {}
    inventory = []
    for name, part in sorted(asset.parts.items()):
        if not isinstance(name, str) or not name or not isinstance(part, Part):
            raise ValueError("parts must map stable names to Part objects")
        if not isinstance(part.body, str) or not part.body or part.role not in {"fixed", "dynamic"}:
            raise ValueError(f"{name}: name a physical body and fixed/dynamic role")
        members = list(part.shape.solids())
        if not members or any(not solid.is_valid or solid.volume <= 0 for solid in members):
            raise ValueError(f"{name}: source must contain valid positive-volume solids")
        bound_count = len(members)
        bound_members = [f"{name}:solid_{index}" for index in range(1, bound_count + 1)]
        if part.source_solid_count is not None and part.source_solid_count != bound_count:
            raise ValueError(
                f"{name}: source_solid_count {part.source_solid_count} does not match "
                f"{bound_count} source solids")
        if part.source_members is not None and list(part.source_members) != bound_members:
            raise ValueError(f"{name}: source_members do not match the source solids in {name}")
        group = groups.setdefault(part.body, {"parts": [], "shapes": [], "role": part.role,
            "density": part.density_kg_m3, "color": part.color})
        if (group["role"], group["density"]) != (part.role, part.density_kg_m3):
            raise ValueError(f"{part.body}: one rigid body requires a consistent role and density")
        shape = copy.copy(part.shape)
        shape.label = name
        group["shapes"].append(shape)
        group["parts"].append(name)
        inventory.append({"name": name, "body": part.body, "role": part.role,
                          "source_solid_count": bound_count,
                          "source_members": bound_members})
    bodies = []
    for index, (name, group) in enumerate(sorted(groups.items()), 1):
        filename = f"body-{index:03d}.step"
        assembly = Compound(children=group["shapes"])
        assembly.label = name
        if not export_step(assembly, output / filename):
            raise ValueError(f"Failed to export {name}")
        body = {"id": name, "source": filename, "fixed": group["role"] == "fixed",
                "density_kg_m3": group["density"]}
        if group["color"] is not None: body["color"] = group["color"]
        bodies.append(body)
    (output / "parts.json").write_text(json.dumps({"description": asset.description, "parts": inventory,
        "body_sources": {b["id"]: b["source"] for b in bodies}}, ensure_ascii=False, indent=2), encoding="utf-8")
    scene = {"schema_version": "slim_scene_v1", "units": asset.units, "bodies": bodies,
             "joints": asset.joints, "gravity_m_s2": asset.gravity_m_s2,
             "authoring_sources": [*sources, "parts.json"]}
    if asset.tests: scene["tests"] = asset.tests
    if asset.equivalent_elasticity is not None: scene["equivalent_elasticity"] = asset.equivalent_elasticity
    (output / "scene.json").write_text(json.dumps(scene, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"scene": str(output / "scene.json"), "parts": len(inventory), "bodies": len(bodies)}


if __name__ == "__main__":
    from .asset import build_file as build_named_asset
    result = build_named_asset(Path(sys.argv[1]), Path(sys.argv[2]))
    Path(sys.argv[3]).write_text(json.dumps(result), encoding="utf-8")
