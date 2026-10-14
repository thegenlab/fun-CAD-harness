"""真实 CAD 的公共准备与原生装载；不包含任何时间推进或功能通过判断。

prepare 使用既有 OCCT/几何编译器，build_native 只构造原机构并接回源内部更新。
package 的 contact 声明不会被升级成理想运动副。未知物理字段必须报错。
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
from typing import Any


class SourcePhysicsError(ValueError):
    """源数据缺失或当前公共构造器尚未实现；不返回削减后的可运行模型。"""


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write(path, value):
    from simulation_ready_cad_workflow.validation_pipeline.mechanism.io import (
        write_json_atomic,
    )

    write_json_atomic(Path(path), value)


def _known(value, allowed, label):
    unknown = set(value) - set(allowed)
    if unknown:
        raise SourcePhysicsError(
            f"UNSUPPORTED_SOURCE_FIELDS {label}: {sorted(unknown)}"
        )


def _positive(value, label):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise SourcePhysicsError(
            f"SOURCE_PARAMETER_REQUIRED {label}: positive finite value"
        )
    return float(value)


def _scene_physics(scene):
    # 顶层载荷、FE、私有回调等若被忽略会改变机械，故只接受已实现的数据字段。
    _known(
        scene,
        {
            "schema_version",
            "units",
            "bodies",
            "joints",
            "gravity_m_s2",
            "authoring_sources",
            "tests",
            "equivalent_elasticity",
            "spur_gears",
            "name",
            "description",
            "artifacts",
            "declared_code",
            "delivery_assets",
            "dial_display",
            "documentation",
            "notes",
            "delivery_artifacts",
            "review_requests",
            "rotor_pair_coverage",
            "source_landmarks_mm",
            "design_notes",  # 已观察的范围/内力索引说明；本构仍从原source_laws安装。
            "assembly_pose",  # 已观察的作者装配说明；不用于设置原生位姿。
        },
        "scene",
    )
    for b in scene["bodies"]:
        _known(
            b,
            {
                "id",
                "source",
                "solid_ids",
                "fixed",
                "density_kg_m3",
                "color",
                "description",
                "role",
            },
            "body " + b["id"],
        )
    for j in scene.get("joints", []):
        _known(
            j,
            {
                "id",
                "type",
                "body_a",
                "body_b",
                "anchor",
                "axis",
                "region_a",
                "region_b",
                "damping",
                "support",
                "description",
                "note",  # BM62已观察的关节文字说明；仅保留，不执行或改变位姿。
            },
            "joint " + j["id"],
        )


def _combine_groups(geometry, parts, groups, grounded, output):
    """把同一源刚性组的全部实体合并，逐件密度和完整平行轴惯量不丢失。"""
    import numpy as np
    import trimesh
    from simulation_ready_cad_workflow.validation_pipeline.slim_harness.geometry import (
        _write_obj,
    )

    records = {r["id"]: r for r in geometry["bodies"]}
    combined = []
    for index, group in enumerate(groups):
        ids = group["part_ids"]
        rows = [records[x] for x in ids]
        mass = sum(r["mass_kg"] for r in rows)
        center = (
            sum((r["mass_kg"] * np.asarray(r["position_m"]) for r in rows), np.zeros(3))
            / mass
        )
        inertia = np.zeros((3, 3))
        meshes = []
        components = []
        analytic = []
        for row in rows:
            offset = np.asarray(row["position_m"]) - center
            inertia += np.asarray(row["inertia_kg_m2"]) + row["mass_kg"] * (
                float(offset @ offset) * np.eye(3) - np.outer(offset, offset)
            )
            mesh = trimesh.load(row["mesh_file"], force="mesh", process=False)
            mesh.vertices += offset
            meshes.append(mesh)
            for comp in row["source_components"]:
                item = copy.deepcopy(comp)
                sub = trimesh.load(item["mesh_file"], force="mesh", process=False)
                sub.vertices += offset
                target = (
                    output / "group_components" / f"{index:03}_{len(components):04}.obj"
                )
                item.update(mesh_file=str(target), **_write_obj(target, sub))
                components.append(item)
            for item in row.get("source_analytic_material", []):
                item = copy.deepcopy(item)
                item["center_local_m"] = (
                    np.asarray(item["center_local_m"]) + offset
                ).tolist()
                analytic.append(item)
        mesh = trimesh.util.concatenate(meshes)
        target = output / f"group-{index:03}.obj"
        export = _write_obj(target, mesh)
        row = copy.deepcopy(rows[0])
        volume = sum(x["volume_m3"] for x in rows)
        row.update(
            id=group["rigid_group_id"],
            fixed=group["rigid_group_id"] in grounded,
            position_m=center.tolist(),
            source_position_m=center.tolist(),
            mass_kg=mass,
            volume_m3=volume,
            density_kg_m3=mass / volume,
            inertia_kg_m2=inertia.tolist(),
            mesh_file=str(target),
            audit_mesh_file=str(target),
            mesh_sha256=export["sha256"],
            source_solid_ids=[s for r in rows for s in r["source_solid_ids"]],
            source_components=components,
            source_analytic_material=analytic,
            source_part_properties=[
                dict(
                    part_id=r["id"],
                    mass_kg=r["mass_kg"],
                    density_kg_m3=r["density_kg_m3"],
                    position_m=r["position_m"],
                    inertia_kg_m2=r["inertia_kg_m2"],
                )
                for r in rows
            ],
        )
        row["fidelity"].update(
            triangle_count=export["triangle_count"],
            contained_analytic_material_supplements=len(analytic),
        )
        combined.append(row)
    geometry = copy.deepcopy(geometry)
    geometry["bodies"] = combined
    for binding in geometry["source_bindings"]:
        binding["solid_owners"] = {
            s: r["id"] for r in combined for s in r["source_solid_ids"]
        }
    return geometry


def _package_scene(
    path,
    output,
    connection_interpretations=None,
    source_adapter_plan=None,
    source_geometry_evidence=None,
    source_fixture=None,
):
    from simulation_ready_cad_workflow.validation_pipeline.slim_harness.geometry import (
        prepare_geometry,
    )

    package = _read(path)
    capsules = {}
    family_specs = {}
    for reference in package.get("mechanism_capsule_files", []):
        capsule = _read(path.parent / reference)
        capsules[reference] = capsule
        family_file = capsule.get("reduced_model_claim", {}).get("family_spec_file")
        if family_file and (path.parent / family_file).is_file():
            family_specs[family_file] = _read(path.parent / family_file)
    adapter_source = None
    if source_adapter_plan is not None:
        plan_path = Path(source_adapter_plan).resolve(strict=True)
        plan = _read(plan_path)
        if Path(plan["package_dir"]).resolve() != path.parent:
            raise SourcePhysicsError("SOURCE_ADAPTER_PLAN_PACKAGE_MISMATCH")
        normalized = plan["adapter_spec"]["family_spec"]
        matching = [
            k
            for k, v in family_specs.items()
            if v.get("mechanism_kind") == normalized.get("mechanism_kind")
        ]
        if len(matching) != 1:
            raise SourcePhysicsError("SOURCE_ADAPTER_FAMILY_MATCH_REQUIRED")
        adapter_source = {
            "path": str(plan_path),
            "family_spec_as_delivered": copy.deepcopy(family_specs[matching[0]]),
            "normalized_source_family_spec": normalized,
        }
        family_specs[matching[0]] = normalized
    # 已知源本构由legacy_source_laws显式恢复，未知类型仍拒绝而非丢掉。
    has_detent_law = any(
        x.get("latch_press_joint", {}).get("spring_stiffness_n_m")
        for x in family_specs.values()
    )
    has_pen_law = any(
        x.get("mechanism_kind") == "click_pen_cam_latch" for x in family_specs.values()
    )
    has_nail_law = any(
        x.get("mechanism_kind") == "nail_clipper_compound_lever"
        for x in family_specs.values()
    )
    has_bicycle_law = any(
        x.get("mechanism_kind") == "bicycle_chain_sprocket_drive"
        for x in family_specs.values()
    )
    if has_bicycle_law and source_fixture != "bicycle_steering_at_rest":
        raise SourcePhysicsError(
            "BICYCLE_SOURCE_COMPOSITION_REQUIRES_DECLARED_STEERING_FIXTURE"
        )
    has_scissor_law = any(
        x.get("mechanism_kind") == "hydraulic_scissor_lift"
        for x in family_specs.values()
    )
    has_axis_spring = any(
        x.get("mechanism_kind") == "axis_spring" for x in family_specs.values()
    )
    contract_path = (
        path.parent / package.get("assembly_contract_file", "assembly_contract.json")
    ).resolve()
    contract = _read(contract_path)
    _known(
        contract,
        {
            "schema_version",
            "units",
            "world_frame",
            "parts",
            "rigid_groups",
            "grounded_rigid_group_ids",
            "connections",
            "frames",
        },
        "assembly_contract",
    )
    if contract.get("units") != {"length": "m", "mass": "kg", "angle": "rad"}:
        raise SourcePhysicsError(
            "UNSUPPORTED_CONTRACT_UNITS: expected explicit m/kg/rad"
        )
    # 不能忽略藏在包顶层的额外物理模型；常规索引/生成源/测试文件由原样快照保存。
    for key in (
        "physics",
        "springs",
        "fea",
        "equivalent_elasticity",
        "joints",
        "loads",
        "initial_state",
        "contact_materials",
    ):
        if key in package:
            raise SourcePhysicsError(
                "UNSUPPORTED_PACKAGE_PHYSICS "
                + key
                + "; use a source slim_scene declaration"
            )
    parts = contract["parts"]
    groups = contract["rigid_groups"]
    partmap = {p["part_id"]: p for p in parts}
    if len(partmap) != len(parts):
        raise SourcePhysicsError("DUPLICATE_SOURCE_PART")
    groupmap = {g["rigid_group_id"]: g for g in groups}
    if len(groupmap) != len(groups):
        raise SourcePhysicsError("DUPLICATE_RIGID_GROUP")
    membership = [x for g in groups for x in g["part_ids"]]
    if len(membership) != len(set(membership)) or set(membership) != set(partmap):
        raise SourcePhysicsError(
            "RIGID_GROUP_PART_COVERAGE: each original part must belong exactly once"
        )
    grounded = contract.get("grounded_rigid_group_ids")
    if not isinstance(grounded, list) or not set(grounded) <= set(groupmap):
        raise SourcePhysicsError("GROUNDED_GROUP_DECLARATION_REQUIRED")
    assembly = (path.parent / package["assembly_file"]).resolve()
    bodies = []
    for group in groups:
        _known(
            group,
            {"rigid_group_id", "part_ids", "connection_evidence", "description"},
            "rigid_group",
        )
        for pid in group["part_ids"]:
            part = partmap[pid]
            _known(
                part,
                {
                    "part_id",
                    "occurrence_id",
                    "role",
                    "rigid_group_id",
                    "source_solid_ids",
                    "declared_bounds_m",
                    "material",
                    "description",
                },
                "part " + pid,
            )
            if part.get("rigid_group_id") != group["rigid_group_id"]:
                raise SourcePhysicsError("INCONSISTENT_PART_GROUP " + pid)
            if part.get("role") not in (
                "fixed",
                "dynamic",
                "moving",
                "rigid",
                "passive",
                "context",
                None,
            ):
                raise SourcePhysicsError(
                    "UNSUPPORTED_PART_ROLE " + pid + ": " + str(part.get("role"))
                )
            density = _positive(
                part.get("material", {}).get("density_kg_m3"), "density " + pid
            )
            material = part.get("material", {})
            if set(material) - {"name", "density_kg_m3", "description"}:
                raise SourcePhysicsError(
                    "UNSUPPORTED_MATERIAL_LAW "
                    + pid
                    + ": "
                    + str(
                        sorted(set(material) - {"name", "density_kg_m3", "description"})
                    )
                )
            bodies.append(
                dict(
                    id=pid,
                    source=os.path.relpath(assembly, output),
                    solid_ids=part["source_solid_ids"],
                    density_kg_m3=density,
                    fixed=group["rigid_group_id"] in grounded,
                )
            )
    connections = contract.get("connections", [])
    legacy_fixed = []
    interpretations = connection_interpretations or {}
    applied_interpretations = []
    if set(interpretations) - {x["connection_id"] for x in connections}:
        raise SourcePhysicsError("INTERPRETATION_NAMES_UNKNOWN_CONNECTION")
    for connection in connections:
        _known(
            connection,
            {
                "connection_id",
                "between_rigid_group_ids",
                "connection_type",
                "evidence",
                "description",
            },
            "connection",
        )
        pair = connection["between_rigid_group_ids"]
        if len(pair) != 2 or len(set(pair)) != 2 or not set(pair) <= set(groupmap):
            raise SourcePhysicsError("INVALID_CONNECTION_GROUPS")
        kind = connection["connection_type"]
        interpretation = interpretations.get(connection["connection_id"])
        if interpretation is not None:
            # 仅明确批准的单连接诊断解释；不改变原contract，不推断弹簧/理想运动副。
            if (
                interpretation.get("mode") != "source_geometry_contact"
                or interpretation.get("authorized_by") != "main_agent"
                or interpretation.get("source_package_path") != str(path)
                or interpretation.get("connection_id") != connection["connection_id"]
                or not interpretation.get("approval_record")
                or not interpretation.get("approval_text")
                or not interpretation.get("scope")
            ):
                raise SourcePhysicsError(
                    "EXPLICIT_SOURCE_CONNECTION_INTERPRETATION_APPROVAL_REQUIRED"
                )
            if kind not in ("joint", "revolute", "prismatic", "cylindrical"):
                raise SourcePhysicsError(
                    "INTERPRETATION_ONLY_FOR_EXPLICIT_UNIMPLEMENTED_JOINT_DECLARATION"
                )
            applied_interpretations.append(
                dict(
                    source_connection=copy.deepcopy(connection),
                    interpretation=copy.deepcopy(interpretation),
                )
            )
            continue
        if has_bicycle_law and kind in (
            "revolute",
            "revolute_bearing",
            "prismatic",
            "analytical_transmission",
        ):
            continue
        if has_scissor_law and kind in (
            "revolute",
            "prismatic",
            "prismatic_roller_track",
            "cylindrical",
        ):
            continue
        if has_axis_spring and kind == "plain_revolute_contact_with_torsion_spring":
            continue
        if has_pen_law and kind in (
            "prismatic",
            "prismatic_with_compression_spring",
            "cylindrical_indexing_cam",
            "cam_contact",
            "cam_follower_contact",
        ):
            continue
        if has_nail_law and kind == "revolute":
            continue
        if has_detent_law and kind in ("prismatic", "spring_return_prismatic"):
            continue  # 原两有限bushing由已知detent源builder恢复，不改成理想副。
        if kind in (
            "contact",
            "plain_revolute_contact",
            "plain_prismatic_contact",
            "plain_revolute_double_bearing",
            "prismatic_guide_shoes",
            "rack_pinion_mesh",
            "plain_revolute_pin",
            "crown_bevel_gear_mesh",
            "discrete_detent_contact",
            "prismatic_roller_track",
            "revolute_journal",
            "one_way_face_clutch",
            "one_way_pawls",
        ):
            pass  # 原实体接触负责全部自由度限制，绝不补理想轴承/止挡。
        elif kind == "fixed":
            legacy_fixed.append(connection)
        else:
            raise SourcePhysicsError(
                f'UNSUPPORTED_CONNECTION {connection["connection_id"]}: {kind}; explicit source native/equivalent declaration required'
            )
    scene = dict(
        schema_version="slim_scene_v1",
        units="m",
        bodies=bodies,
        joints=[],
        tests=[],
        gravity_m_s2=contract["world_frame"]["gravity_m_s2"],
    )
    staging = output / "source_geometry_scene.json"
    _write(staging, scene)
    geometry = prepare_geometry(staging, output / "geometry_parts")
    if geometry["errors"]:
        raise SourcePhysicsError(
            "SOURCE_GEOMETRY_ERRORS " + json.dumps(geometry["errors"])
        )
    geometry = _combine_groups(
        geometry, parts, groups, set(grounded), output / "geometry_groups"
    )
    from benchmark_templates.legacy_source_laws import prepare_laws

    legacy_laws, law_frictions = prepare_laws(
        family_specs,
        geometry,
        output / "source_law_geometry",
        path.parent,
        capsules=capsules,
        contract=contract,
        source_geometry_evidence=source_geometry_evidence,
        source_fixture=source_fixture,
    )
    scene["bodies"] = [
        dict(
            id=r["id"],
            source=os.path.relpath(assembly, output),
            solid_ids=r["source_solid_ids"],
            fixed=r["fixed"],
            density_kg_m3=r["density_kg_m3"],
        )
        for r in geometry["bodies"]
    ]
    scene["legacy_fixed_connections"] = legacy_fixed
    scene["connection_interpretations"] = applied_interpretations
    scene["legacy_source_laws"] = legacy_laws
    source_friction = {
        float(spec["friction"]) for spec in family_specs.values() if "friction" in spec
    }
    source_friction.update(law_frictions)
    if len(source_friction) > 1:
        raise SourcePhysicsError("SOURCE_CONTACT_FRICTION_CONFLICT")
    scene["source_contact_profile"] = (
        {"friction": source_friction.pop()} if source_friction else {}
    )
    for law in legacy_laws:
        scene["source_contact_profile"].update(law.get("contact_profile", {}))
    return (
        scene,
        geometry,
        dict(
            kind="package",
            package_path=str(path),
            contract_path=str(contract_path),
            package=package,
            contract=contract,
            capsules=capsules,
            family_specs=family_specs,
            source_adapter_plan=adapter_source,
            source_fixture=source_fixture,
            source_geometry_evidence=(
                str(source_geometry_evidence) if source_geometry_evidence else None
            ),
        ),
    )


def prepare(
    source_path,
    output,
    *,
    connection_interpretations=None,
    source_adapter_plan=None,
    source_geometry_evidence=None,
    source_fixture=None,
):
    """只在全新 output 中写几何和 prepared.json，返回该文件 Path；源目录只读。"""
    from simulation_ready_cad_workflow.validation_pipeline.slim_harness.geometry import (
        prepare_geometry,
    )
    from simulation_ready_cad_workflow.validation_pipeline.slim_harness.contracts import (
        load_model_scene,
        compile_scene,
    )

    path = Path(source_path).resolve(strict=True)
    output = Path(output).resolve()
    if path.is_dir():
        options = [
            path / name
            for name in ("scene.json", "package.json")
            if (path / name).is_file()
        ]
        if len(options) != 1:
            raise SourcePhysicsError(
                "SOURCE_ENTRY_AMBIGUOUS: pass the exact scene.json or package.json"
            )
        path = options[0]
    output.mkdir(parents=True, exist_ok=False)
    try:
        if _read(path).get("schema_version") == "slim_scene_v1":
            if (
                connection_interpretations
                or source_adapter_plan
                or source_geometry_evidence
                or source_fixture
            ):
                raise SourcePhysicsError("SCENE_EXTRA_SOURCE_OVERRIDE_NOT_SUPPORTED")
            scene = load_model_scene(path)
            _scene_physics(scene)
            geometry = prepare_geometry(path, output / "geometry")
            if geometry["errors"]:
                raise SourcePhysicsError(
                    "SOURCE_GEOMETRY_ERRORS " + json.dumps(geometry["errors"])
                )
            origin = dict(
                kind="slim_scene",
                scene_path=str(path),
                source_scene=copy.deepcopy(scene),
            )
        else:
            scene, geometry, origin = _package_scene(
                path,
                output,
                connection_interpretations,
                source_adapter_plan,
                source_geometry_evidence,
                source_fixture,
            )
        # benchmark拥有任务；编译视图只剥离源tests，完整原文已保存在origin。
        model_view = copy.deepcopy(scene)
        model_view["tests"] = []
        prepared = compile_scene(model_view, geometry)
        # 几何和连接的源绑定保留；正式接触审计/功能证书由Harness另行执行。
        prepared["cad_loader"] = dict(
            schema_version="benchmark_cad_loader_v1",
            origin=origin,
            source_geometry=geometry,
            source_tests_execution="not_executed_benchmark_owns_trials",
            source_default_profile="slim_harness.runtime.CONTACT; uncalibrated Harness defaults",
            scope="Native whole-source loading/startup only; not functional PASS or Harness certification; source tests not executed by this benchmark loader",
        )
        for law in prepared.get("legacy_source_laws", []):
            prepared["cad_loader"]["scope"] += "; " + law["scope"]
        if prepared.get("connection_interpretations"):
            prepared["cad_loader"]["scope"] += "; " + "; ".join(
                x["interpretation"]["scope"]
                for x in prepared["connection_interpretations"]
            )
        destination = output / "prepared.json"
        _write(destination, prepared)
        return destination
    except Exception as exc:
        _write(
            output / "prepare_error.json",
            dict(
                status="preparation_error",
                source=str(path),
                error=f"{type(exc).__name__}: {exc}",
            ),
        )
        raise


def _verified_diagnostic_geometry_report(source_path, scene, report_path):
    """复用本工具已生成的原几何报告；来源/实现/文件内容不符立即拒绝。"""
    import hashlib
    from simulation_ready_cad_workflow.validation_pipeline.slim_harness.geometry import (
        GEOMETRY_IMPLEMENTATION_SHA256,
    )

    report_path = Path(report_path).resolve(strict=True)
    geometry = _read(report_path)
    digests = {}

    def digest(path):
        path = Path(path).resolve(strict=True)
        if path not in digests:
            h = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(block)
            digests[path] = h.hexdigest()
        return digests[path]

    def require(condition, detail):
        if not condition:
            raise SourcePhysicsError("GEOMETRY_REPORT_MISMATCH " + detail)

    require(geometry.get("schema_version") == "slim_geometry_v1", "schema")
    require(not geometry.get("errors"), "geometry errors")
    require(
        geometry.get("geometry_implementation_sha256")
        == GEOMETRY_IMPLEMENTATION_SHA256,
        "implementation",
    )
    require(
        Path(geometry["scene_file"]).resolve() == source_path
        and geometry.get("scene_sha256") == digest(source_path),
        "source scene",
    )
    require(
        geometry.get("units") == "m" and geometry.get("input_units") == scene["units"],
        "units",
    )
    rows = {r["id"]: r for r in geometry["bodies"]}
    require(
        len(rows) == len(geometry["bodies"]) == len(scene["bodies"])
        and set(rows) == {r["id"] for r in scene["bodies"]},
        "body identities",
    )
    bindings = {
        str(Path(r["source_file"]).resolve()): r for r in geometry["source_bindings"]
    }
    for declared in scene["bodies"]:
        row = rows[declared["id"]]
        source = (source_path.parent / declared["source"]).resolve(strict=True)
        require(
            Path(row["source_file"]).resolve() == source
            and row["source_sha256"] == digest(source),
            "body source " + row["id"],
        )
        stat = source.stat()
        require(
            list(row["source_stat"]) == [stat.st_size, stat.st_mtime_ns, stat.st_ino],
            "source stamp " + row["id"],
        )
        require(row["fixed"] == declared.get("fixed", False), "fixed " + row["id"])
        if "density_kg_m3" in declared:
            require(
                row["density_kg_m3"] == declared["density_kg_m3"],
                "density " + row["id"],
            )
        binding = bindings.get(str(source))
        require(
            binding is not None
            and binding["source_sha256"] == row["source_sha256"]
            and binding["source_kind"] == row["source_kind"],
            "source binding " + row["id"],
        )
        owners = binding["solid_owners"]
        require(len(owners) == binding["solid_count"], "solid count")
        require(
            set(row["source_solid_ids"])
            == {solid for solid, owner in owners.items() if owner == row["id"]},
            "solid ownership " + row["id"],
        )
        require(
            digest(row["mesh_file"]) == row["mesh_sha256"], "body mesh " + row["id"]
        )
        require(
            Path(row.get("audit_mesh_file", row["mesh_file"])).resolve()
            == Path(row["mesh_file"]).resolve(),
            "unsupported different audit mesh",
        )
        require(
            {r["solid_id"] for r in row["source_components"]}
            == set(row["source_solid_ids"]),
            "component ownership " + row["id"],
        )
        for component in row["source_components"]:
            require(
                digest(component["mesh_file"]) == component["sha256"],
                "source component " + component["solid_id"],
            )
    return geometry, {
        "report_path": str(report_path),
        "report_sha256": digest(report_path),
        "matched_scene_hash": geometry["scene_sha256"],
        "geometry_implementation_sha256": GEOMETRY_IMPLEMENTATION_SHA256,
        "verified_file_count": len(digests),
        "checks": [
            "exact scene bytes/path",
            "implementation",
            "units",
            "body identities",
            "source bytes/stamps",
            "density",
            "fixed flags",
            "solid owners",
            "body/component mesh bytes",
        ],
        "geometry_regenerated": False,
    }


def prepare_declared_model_diagnostic(
    source_path,
    output,
    *,
    declared_ideal_joint_ids,
    affected_task_ids,
    purpose,
    geometry_report=None,
):
    """单独诊断源明确理想模型；保留实物支承缺项，绝不放宽正式prepare。

    仅显式列出的源理想副可保留材料/局部连接支承失败。schema、非法轴、
    未实现内力、坏CAD仍报错；原约束、原几何和初始相对位姿都不被修改。
    """
    import numpy as np
    from simulation_ready_cad_workflow.validation_pipeline.slim_harness.geometry import (
        prepare_geometry,
        geometry_distance_to_point,
    )
    from simulation_ready_cad_workflow.validation_pipeline.slim_harness.contracts import (
        load_declared_model_scene,
        compile_scene,
        DeliveryError,
    )

    path = Path(source_path).resolve(strict=True)
    if path.is_dir():
        path = (path / "scene.json").resolve(strict=True)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    try:
        scene = load_declared_model_scene(
            path, declared_ideal_joint_ids=declared_ideal_joint_ids
        )
        _scene_physics(scene)
        ids, tasks = set(declared_ideal_joint_ids), tuple(affected_task_ids)
        source_joints = {j["id"]: j for j in scene.get("joints", [])}
        if not ids or not ids <= set(source_joints) or not tasks or not purpose:
            raise SourcePhysicsError(
                "DECLARED_MODEL_DIAGNOSTIC_NEEDS_EXISTING_JOINT_IDS_TASKS_PURPOSE"
            )
        if any(
            source_joints[j]["type"]
            not in ("fixed", "revolute", "prismatic", "spherical")
            for j in ids
        ):
            raise SourcePhysicsError("DIAGNOSTIC_ONLY_FOR_SOURCE_DECLARED_IDEAL_JOINTS")
        reuse = None
        if geometry_report is None:
            geometry = prepare_geometry(path, output / "geometry")
        else:
            geometry, reuse = _verified_diagnostic_geometry_report(
                path, scene, geometry_report
            )
        if geometry["errors"]:
            raise SourcePhysicsError(
                "SOURCE_GEOMETRY_ERRORS " + json.dumps(geometry["errors"])
            )
        prepared = copy.deepcopy(scene)
        prepared.update(bodies=geometry["bodies"], units="m", tests=[], joints=[])
        prepared.pop("validation_scope", None)
        prepared["source_support_checks"] = []
        failures = []
        records = {b["id"]: b for b in geometry["bodies"]}
        scale = 0.001 if scene["units"] == "mm" else 1.0
        for original in scene.get("joints", []):
            single = copy.deepcopy(scene)
            single["joints"], single["tests"] = [copy.deepcopy(original)], []
            # 全场扩展在下方仍用完整原scene和原公开compiler编译；这里仅审单副。
            for key in ("spur_gears", "equivalent_elasticity", "validation_scope"):
                single.pop(key, None)
            missing_regions = [
                f"region_{side}"
                for side in ("a", "b")
                if f"region_{side}" not in original
            ]
            try:
                if missing_regions:
                    # 明确支承证明仍由原checker审查；缺region本身不伪装成schema完备。
                    from simulation_ready_cad_workflow.validation_pipeline.slim_harness.source_support import (
                        check_supports,
                    )

                    try:
                        check_supports(single, geometry)
                    except (ValueError, KeyError, TypeError) as exc:
                        raise DeliveryError(f"Source support: {exc}") from exc
                    raise DeliveryError(
                        f"Joint {original['id']} missing_support_regions: {missing_regions}"
                    )
                checked = compile_scene(single, geometry)
                joint = checked["joints"][0]
                prepared["source_support_checks"].extend(
                    checked.get("source_support_checks", [])
                )
            except DeliveryError as exc:
                error = str(exc)
                jid = original["id"]
                material_errors = (
                    f"Joint {jid} missing_support_regions:",
                    f"Joint {jid} native constraint frame misses region_",
                    f"Joint {jid} region_a is not near real source material:",
                    f"Joint {jid} region_b is not near real source material:",
                    f"Joint {jid} starts outside its local connection range:",
                    "Source support: Declared shaft surface is absent from the original source",
                    "Source support: Declared bore surface is absent from the original source",
                    "Source support: Shaft declaration has no source material",
                    "Source support: Bore is filled or has no surrounding source material",
                )
                if jid not in ids or not error.startswith(material_errors):
                    raise
                # 和原compiler相同的SI/COM局部区域变换；不改源anchor或放大radius。
                joint = copy.deepcopy(original)
                joint["anchor"] = (np.asarray(original["anchor"]) * scale).tolist()
                axis = np.asarray(original.get("axis", [0, 0, 1]), dtype=float)
                joint["axis"] = (axis / np.linalg.norm(axis)).tolist()
                for side in ("a", "b"):
                    if f"region_{side}" not in joint:
                        continue  # 未知区域保持缺失，绝不造0距离或半径。
                    region = joint[f"region_{side}"]
                    world = np.asarray(region["points"], dtype=float) * scale
                    body = records[joint[f"body_{side}"]]
                    region["radius"] *= scale
                    region["source_distance_m"] = [
                        geometry_distance_to_point(body, x.tolist())
                        for x in (*world, world.mean(axis=0))
                    ]
                    region["points"] = (world - np.asarray(body["position_m"])).tolist()
                failures.append(
                    {
                        "joint_id": jid,
                        "strict_support_error": error,
                        "missing_support_regions": missing_regions,
                        "original_source_spec": copy.deepcopy(original),
                        "diagnostic_constraint": "same source ideal joint, missing material support retained",
                    }
                )
                prepared["source_support_checks"].append(
                    {
                        "id": jid,
                        "status": (
                            "missing_support_regions"
                            if missing_regions
                            else "missing_physical_support_declared_model_diagnostic"
                        ),
                        "error": error,
                    }
                )
            if original["id"] in ids:
                joint["source_declared_exact"] = True
            prepared["joints"].append(joint)
        if scene.get("spur_gears"):
            from simulation_ready_cad_workflow.validation_pipeline.slim_harness.spur_gears import (
                compile_spur_gears,
            )

            prepared["spur_gears"] = compile_spur_gears(scene, geometry)
        if scene.get("equivalent_elasticity") is not None:
            from simulation_ready_cad_workflow.validation_pipeline.slim_harness.equivalent_elastic_compiler import (
                compile_equivalent_elasticity,
            )

            equivalent = compile_equivalent_elasticity(scene, geometry)
            prepared["equivalent_elasticity"] = equivalent["declaration"]
            prepared["validation_scope"] = equivalent["validation_scope"]
        reason = {
            "kind": "source_declared_ideal_model_diagnostic_not_physical_support_proof",
            "purpose": purpose,
            "strict_support_failures": failures,
            "declared_ideal_joint_ids": sorted(ids),
        }
        prepared["cad_loader"] = dict(
            schema_version="benchmark_cad_loader_v1",
            origin=dict(
                kind="slim_scene",
                scene_path=str(path),
                source_scene=copy.deepcopy(scene),
            ),
            source_geometry=geometry,
            source_tests_execution="not_executed_benchmark_owns_trials",
            scope="Source-declared ideal model diagnostic; missing physical support retained, not physical PASS or certification",
            declared_model_diagnostic=reason,
            verified_geometry_reuse=reuse,
            diagnostic_evidence_gaps={task: reason for task in tasks},
        )
        destination = output / "prepared.json"
        _write(destination, prepared)
        return destination
    except Exception as exc:
        _write(
            output / "prepare_error.json",
            dict(
                status="preparation_error",
                source=str(path),
                error=f"{type(exc).__name__}: {exc}",
            ),
        )
        raise


def build_native(prepared_path):
    """构造所有原实体、源约束及源等效内力，时间推进仍只有公共 native.run。"""
    import numpy as np
    from benchmark_templates.runtime import NativeModel
    from simulation_ready_cad_workflow.validation_pipeline.slim_harness import (
        runtime as sr,
    )
    from simulation_ready_cad_workflow.validation_pipeline.slim_harness.validation_scope import (
        validate_validation_scope,
    )

    prepared = (
        _read(prepared_path) if not isinstance(prepared_path, dict) else prepared_path
    )
    if (
        prepared.get("cad_loader", {}).get("schema_version")
        != "benchmark_cad_loader_v1"
    ):
        raise SourcePhysicsError("PREPARED_LOADER_CONTRACT_REQUIRED")
    c = sr.load_pychrono_core()
    system = c.ChSystemNSC()
    profile = dict(sr.CONTACT)
    profile.update(prepared.get("source_contact_profile", {}))
    c.ChCollisionModel.SetDefaultSuggestedEnvelope(profile["envelope_m"])
    c.ChCollisionModel.SetDefaultSuggestedMargin(profile["margin_m"])
    system.SetCollisionSystemType(c.ChCollisionSystem.Type_BULLET)
    system.SetNumThreads(3, 3, 1)
    system.SetGravitationalAcceleration(c.ChVector3d(*prepared["gravity_m_s2"]))
    system.SetSolverType(c.ChSolver.Type_PSOR)
    solver = system.GetSolver().AsIterative()
    solver.SetMaxIterations(
        240
        if any(
            e["kind"] == "friction_clutch"
            for e in (prepared.get("equivalent_elasticity") or {}).get("elements", [])
        )
        else 160
    )
    solver.SetTolerance(0.0)
    material = c.ChContactMaterialNSC()
    material.SetFriction(profile["friction"])
    material.SetRestitution(profile["restitution"])
    scope = prepared.get("validation_scope")
    equiv = prepared.get("equivalent_elasticity")
    if equiv is not None:
        validate_validation_scope(scope, prepared["bodies"])
        if scope["mode"] != "equivalent_elastic":
            raise SourcePhysicsError("EQUIVALENT_SCOPE_REQUIRED")
    elastic_refs = set(scope["elastic_reference_body_ids"]) if scope else set()
    records = {r["id"]: r for r in prepared["bodies"]}
    bodies = {}
    meshes = []
    shapes = []
    capsules = {}
    accumulators = {}
    for key, r in records.items():
        body = c.ChBody()
        body.SetName(key)
        body.SetFixed(r["fixed"])
        body.SetPos(c.ChVector3d(*r["position_m"]))
        body.SetRot(c.ChQuaterniond(*r["rotation_wxyz"]))
        body.SetMass(r["mass_kg"])
        inertia = np.asarray(r["inertia_kg_m2"])
        body.SetInertiaXX(c.ChVector3d(*np.diag(inertia)))
        body.SetInertiaXY(c.ChVector3d(inertia[0, 1], inertia[0, 2], inertia[1, 2]))
        component_rows = r.get("source_component_colliders")
        if component_rows is not None:
            # 忠实接回原family后端的分块/精确基本体构造；完整audit mesh与源质量不变。
            from simulation_ready_cad_workflow.validation_pipeline.mechanism.worker import (
                _make_body,
            )
            from simulation_ready_cad_workflow.validation_pipeline.mechanism.io import (
                sha256_file,
            )

            local_meshes = []
            for comp in component_rows:
                path = Path(comp["mesh_file"])
                if not path.is_file() or sha256_file(path) != comp["mesh_sha256"]:
                    raise SourcePhysicsError(
                        "SOURCE_COMPONENT_BYTES_CHANGED " + str(path)
                    )
                local_meshes.append((path, bool(comp["convex"])))
            body = _make_body(
                c,
                name=key,
                local_meshes=local_meshes,
                origin_abs_m=r["position_m"],
                fixed=r["fixed"],
                mass_kg=r["mass_kg"],
                inertia_xx=np.diag(inertia),
                material=material,
                collision_radius_m=0.0,
                physical_contact_required=True,
            )
            body.SetRot(c.ChQuaterniond(*r["rotation_wxyz"]))
            body.SetInertiaXY(c.ChVector3d(inertia[0, 1], inertia[0, 2], inertia[1, 2]))
        else:
            mesh = c.ChTriangleMeshConnected()
            if not mesh.LoadWavefrontMesh(
                str(r.get("collision_mesh_file", r["mesh_file"])), True, False
            ):
                raise SourcePhysicsError("MESH_LOAD_FAILED " + key)
            body.AddCollisionShape(
                c.ChCollisionShapeTriangleMesh(
                    material, mesh, r["fixed"], False, profile["mesh_swept_radius_m"]
                )
            )
            meshes.append(mesh)
            for supplement in r.get("source_analytic_material", []):
                if supplement.get("source_solid_id") in r.get(
                    "source_collision_suppressed_solids", []
                ):
                    continue
                if supplement["type"] != "capsule":
                    raise SourcePhysicsError(
                        "UNSUPPORTED_SOURCE_ANALYTIC_MATERIAL " + supplement["type"]
                    )
                shape = c.ChCollisionShapeCapsule(
                    material, supplement["radius_m"], supplement["straight_length_m"]
                )
                frame = c.ChFramed(
                    c.ChVector3d(*supplement["center_local_m"]),
                    sr._axis_quat(c, supplement["axis_source"]),
                )
                body.AddCollisionShape(shape, frame)
                shapes.append((shape, frame))
                capsules.setdefault(key, []).append(supplement)
        body.EnableCollision(
            key not in elastic_refs and not r.get("source_reference_only")
        )
        system.Add(body)
        bodies[key] = body
        accumulators[key] = body.AddAccumulator()
    joints = {}
    types = {
        "revolute": c.ChLinkLockRevolute,
        "prismatic": c.ChLinkLockPrismatic,
        "spherical": c.ChLinkLockSpherical,
        "fixed": c.ChLinkLockLock,
    }
    for spec in prepared.get("joints", []):
        if spec["type"] not in types:
            raise SourcePhysicsError("UNSUPPORTED_PREPARED_JOINT " + spec["type"])
        a, b = bodies[spec["body_a"]], bodies[spec["body_b"]]
        link = types[spec["type"]]()
        link.SetName(spec["id"])
        link.Initialize(
            a,
            b,
            c.ChFramed(c.ChVector3d(*spec["anchor"]), sr._axis_quat(c, spec["axis"])),
        )
        system.Add(link)
        joints[spec["id"]] = dict(
            spec=spec,
            link=link,
            active=True,
            coordinate=0.0,
            initial_offset=np.asarray(sr._vec(a.GetPos()))
            - np.asarray(sr._vec(b.GetPos())),
            rate=0.0,
            effort=0.0,
            work_j=0.0,
            damping_work_j=0.0,
            damping_effort=0.0,
            damping_power_w=0.0,
        )
    # contract fixed 是源明确给出的刚接，不捏造可脱开范围；保留此源语义。
    for conn in prepared.get("legacy_fixed_connections", []):
        x, y = conn["between_rigid_group_ids"]
        a, b = bodies[x], bodies[y]
        spec = dict(
            id=conn["connection_id"],
            type="fixed",
            body_a=x,
            body_b=y,
            anchor=sr._vec(a.GetPos()),
            axis=[0, 0, 1],
            source_permanent_fixed=True,
        )
        link = c.ChLinkLockLock()
        link.SetName(spec["id"])
        link.Initialize(a, b, c.ChFramed(a.GetPos(), a.GetRot()))
        system.Add(link)
        joints[spec["id"]] = dict(
            spec=spec, link=link, active=True, coordinate=0.0, rate=0.0
        )
    from benchmark_templates.legacy_source_laws import LegacyLaws

    legacy = (
        LegacyLaws(
            c,
            system,
            bodies,
            records,
            joints,
            prepared.get("legacy_source_laws", []),
            sr._axis_quat,
        )
        if prepared.get("legacy_source_laws")
        else None
    )
    # 初始采样早于第一次源更新；尚未施加阻尼的原始静止状态功率/功为0。
    for item in joints.values():
        item.setdefault("damping_effort", 0.0)
        item.setdefault("damping_power_w", 0.0)
        item.setdefault("damping_work_j", 0.0)
    fixed_filter = sr.build_fixed_component_filter(bodies, joints, time_s=0.0)
    equivalent = (
        sr.build_equivalent_elasticity(c, system, bodies, joints, equiv)
        if equiv is not None
        else None
    )
    supports = dict(joints)
    if equivalent is not None:
        supports.update(equivalent.support_records)
    eq_filter = (
        sr.build_equivalent_collision_scope(
            c,
            system,
            bodies,
            records,
            scope,
            support_records=equivalent.support_records,
        )
        if equivalent is not None
        else None
    )
    gears = (
        sr.build_spur_gears(
            c,
            system,
            bodies,
            supports,
            prepared["spur_gears"],
            axis_quat=sr._axis_quat,
            friction=profile["friction"],
            previous_filter=eq_filter,
        )
        if prepared.get("spur_gears")
        else None
    )
    margins = sr._initialize_source_mesh_contacts(system, bodies)
    if profile["margin_m"] != sr.CONTACT["margin_m"]:
        for key, body in bodies.items():
            if not body.IsCollisionEnabled():
                continue
            cm = body.GetCollisionModel()
            cm.SetSafeMargin(profile["margin_m"])
            actual = float(cm.GetSafeMargin())
            if not math.isclose(
                actual, profile["margin_m"], rel_tol=1e-6, abs_tol=1e-12
            ):
                raise SourcePhysicsError("SOURCE_COLLISION_MARGIN_NOT_APPLIED")
            margins[key]["effective_m"] = actual
            margins[key]["source_profile_override"] = True
    source_filter = sr.build_source_contact_filter(
        c,
        system,
        bodies,
        records,
        penetration_tolerance_m=profile["penetration_tolerance_m"],
        previous_filter=gears or eq_filter,
    )
    normals = sr.build_source_contact_normals(c, system, source_filter)
    capsule_contacts = sr.build_source_capsule_contacts(
        c, system, source_filter, records, normals, created_capsules_by_body=capsules
    )
    events = []
    update_api = {"mode": None, "fallback_reason": None}

    def update_system():
        # 已在两个实际PyChrono绑定观察到的API差异：只兼容该重载错误。
        if update_api["mode"] == "Update(True)":
            system.Update(True)
            return
        try:
            system.Update()
            chosen = "Update()"
        except TypeError as exc:
            message = str(exc)
            if "ChSystem_Update" not in message or not any(
                x in message.lower() for x in ("argument", "overload")
            ):
                raise
            system.Update(True)
            chosen = "Update(True)"
            update_api["fallback_reason"] = message
        if update_api["mode"] is None:
            model.scope += "; native force-cache API " + chosen
        update_api["mode"] = chosen

    def check_domain():
        if equivalent is None:
            return
        outside = [
            key
            for key, row in equivalent.sample()["elements"].items()
            if not row["within_working_range"]
        ]
        if outside:
            raise SourcePhysicsError(
                "EQUIVALENT_WORKING_RANGE_EXCEEDED " + str(outside)
            )

    def before_step(now, dt):
        # 只清自己的内部阻尼累加器；人手输入由公共操作器管理，二者不能相互清空。
        for key, body in bodies.items():
            body.EmptyAccumulator(accumulators[key])
        for key, item in joints.items():
            spec = item["spec"]
            a, b = bodies[spec["body_a"]], bodies[spec["body_b"]]
            if not spec.get("source_permanent_fixed") and not spec.get(
                "source_declared_exact"
            ):
                ra, rb = spec["region_a"], spec["region_b"]
                gap = (
                    sr.segment_distance(
                        *sr._world_region(c, a, ra), *sr._world_region(c, b, rb)
                    )
                    - ra["radius"]
                    - rb["radius"]
                )
                if item["active"] and gap > profile["penetration_tolerance_m"]:
                    item["active"] = False
                    item["link"].SetDisabled(True)
                    events.append(
                        dict(
                            time_s=now,
                            kind="source_connection_released",
                            joint=key,
                            gap_m=float(gap),
                        )
                    )
        fixed_filter.refresh(time_s=now)
        for key, item in joints.items():
            spec = item["spec"]
            a, b = bodies[spec["body_a"]], bodies[spec["body_b"]]
            axis = np.asarray(
                sr._vec(b.TransformDirectionLocalToParent(c.ChVector3d(*spec["axis"])))
            )
            if spec["type"] == "revolute":
                rate = float(
                    (
                        np.asarray(sr._vec(a.GetAngVelParent()))
                        - np.asarray(sr._vec(b.GetAngVelParent()))
                    )
                    @ axis
                )
            else:
                offset = np.asarray(sr._vec(a.GetPos())) - np.asarray(
                    sr._vec(b.GetPos())
                )
                velocity = (
                    np.asarray(sr._vec(a.GetPosDt()))
                    - np.asarray(sr._vec(b.GetPosDt()))
                    - np.cross(sr._vec(b.GetAngVelParent()), offset)
                )
                rate = float(velocity @ axis)
            item["rate"] = rate
            item["damping_effort"] = 0.0
            item["damping_power_w"] = 0.0
            damping = float(spec.get("damping", 0.0))
            if damping and item["active"]:
                effective = sr._effective_inertia(c, bodies, records, spec, axis)
                effort = -damping * rate / (1 + damping * dt / effective)
                item["damping_effort"] = effort
                if spec["type"] == "revolute":
                    a.AccumulateTorque(
                        accumulators[spec["body_a"]],
                        c.ChVector3d(*(axis * effort)),
                        False,
                    )
                    b.AccumulateTorque(
                        accumulators[spec["body_b"]],
                        c.ChVector3d(*(-axis * effort)),
                        False,
                    )
                elif spec["type"] == "prismatic":
                    point = b.TransformPointLocalToParent(
                        c.ChVector3d(
                            *(
                                np.asarray(spec["anchor"])
                                - records[spec["body_b"]]["position_m"]
                            )
                        )
                    )
                    a.AccumulateForce(
                        accumulators[spec["body_a"]],
                        c.ChVector3d(*(axis * effort)),
                        point,
                        False,
                    )
                    b.AccumulateForce(
                        accumulators[spec["body_b"]],
                        c.ChVector3d(*(-axis * effort)),
                        point,
                        False,
                    )
                else:
                    raise SourcePhysicsError(
                        "UNSUPPORTED_PASSIVE_DAMPING " + spec["type"]
                    )
        if legacy is not None:
            legacy.before_step(now, dt)
        if equivalent is not None:
            for e in equiv["elements"]:
                if (
                    e["kind"] == "friction_clutch"
                    and not joints[e["joint_id"]]["active"]
                ):
                    raise SourcePhysicsError(
                        "EQUIVALENT_CONNECTION_RELEASED " + e["id"]
                    )
            equivalent.before_step(dt)
        capsule_contacts.begin_step()
        if gears is not None:
            gears.apply_friction(dt)

    def commit_inputs(now, dt):
        # 公共run已加入本步全部人工输入后才刷新源力缓存，撤力也在同一时刻生效。
        # 这里只提交当前状态，不推进物理时间；TSDA捕获必须晚于这次Update。
        if legacy is not None:
            legacy.commit_inputs(now, dt, model)
        update_system()
        if equivalent is not None:
            equivalent.capture_native_step_loads()

    def after_step(now, dt):
        if legacy is not None:
            legacy.after_step(now, dt)
        if equivalent is not None:
            equivalent.after_step()
        if gears is not None:
            gears.after_step()
        for key, item in joints.items():
            spec = item["spec"]
            a, b = bodies[spec["body_a"]], bodies[spec["body_b"]]
            axis = np.asarray(
                sr._vec(b.TransformDirectionLocalToParent(c.ChVector3d(*spec["axis"])))
            )
            rate_after = 0.0
            if spec["type"] == "revolute":
                rate_after = float(
                    (
                        np.asarray(sr._vec(a.GetAngVelParent()))
                        - np.asarray(sr._vec(b.GetAngVelParent()))
                    )
                    @ axis
                )
                item["coordinate"] += rate_after * dt
            elif spec["type"] == "prismatic":
                offset = np.asarray(sr._vec(a.GetPos())) - np.asarray(
                    sr._vec(b.GetPos())
                )
                velocity = (
                    np.asarray(sr._vec(a.GetPosDt()))
                    - np.asarray(sr._vec(b.GetPosDt()))
                    - np.cross(sr._vec(b.GetAngVelParent()), offset)
                )
                rate_after = float(velocity @ axis)
                item["coordinate"] = float((offset - item["initial_offset"]) @ axis)
            # 仅记录本步已施加的源阻尼及实测前后速度，不改变施力。
            mean_rate = 0.5 * (item.get("rate", 0.0) + rate_after)
            item["damping_power_w"] = -item.get("damping_effort", 0.0) * mean_rate
            item["damping_work_j"] = (
                item.get("damping_work_j", 0.0) + item["damping_power_w"] * dt
            )
            item["rate"] = rate_after
        check_domain()
        for key, body in bodies.items():
            if not np.isfinite(
                [
                    *sr._vec(body.GetPos()),
                    *sr._vec(body.GetPosDt()),
                    *sr._vec(body.GetAngVelParent()),
                ]
            ).all():
                raise SourcePhysicsError("NONFINITE_BODY_STATE " + key)

    check_domain()
    original = prepared["cad_loader"]["origin"]
    contract = original.get("contract", {})
    objects = dict(
        bodies=bodies,
        joints=joints,
        body_records=records,
        source_frames=contract.get("frames", []),
        source_connections=contract.get("connections", prepared.get("joints", [])),
        prepared=prepared,
        source_events=events,
        equivalent=equivalent,
        spur_gears=gears,
        legacy_source_laws=legacy,
        contact_profile=profile,
        source_mesh_margins=margins,
        system_update_api=update_api,
        fixture_change=prepared["cad_loader"].get("fixture_change"),
    )
    keepalive = (
        bodies,
        meshes,
        shapes,
        material,
        joints,
        fixed_filter,
        equivalent,
        eq_filter,
        gears,
        source_filter,
        normals,
        capsule_contacts,
        legacy,
    )
    model = NativeModel(
        system,
        {},
        {},
        before_step=before_step,
        after_step=after_step,
        keepalive=keepalive,
        scope=prepared["cad_loader"]["scope"],
        initial_state="Original assembled geometry, declared fixed groups, source internal preload and gravity; t=0",
    )
    model.objects = objects
    model.commit_inputs = commit_inputs
    # 诊断允许构造不代表实体支承已成立，任务判定自动保持证据缺项。
    model.task_evidence_gaps.update(
        copy.deepcopy(prepared["cad_loader"].get("diagnostic_evidence_gaps", {}))
    )
    return model


def main():
    parser = argparse.ArgumentParser(description="真实CAD准备/原生装载，无功能PASS")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("source")
    p.add_argument("output")
    p.add_argument("--connection-review", type=Path)
    p.add_argument("--source-adapter-plan", type=Path)
    p.add_argument("--source-geometry-evidence", type=Path)
    p.add_argument("--source-fixture")
    p = sub.add_parser("build")
    p.add_argument("prepared")
    p.add_argument("--smoke-seconds", type=float, default=0.0)
    p.add_argument("--output")
    args = parser.parse_args()
    if args.command == "prepare":
        review = _read(args.connection_review) if args.connection_review else None
        interpretations = {review["connection_id"]: review} if review else None
        print(
            str(
                prepare(
                    args.source,
                    args.output,
                    connection_interpretations=interpretations,
                    source_adapter_plan=args.source_adapter_plan,
                    source_geometry_evidence=args.source_geometry_evidence,
                    source_fixture=args.source_fixture,
                )
            )
        )
        return
    model = build_native(args.prepared)
    result = dict(
        status="native_loaded",
        bodies=len(model.objects["bodies"]),
        joints=len(model.objects["joints"]),
        mass_kg=sum(b.GetMass() for b in model.objects["bodies"].values()),
        functional_pass=None,
        harness_certified=False,
        system_update_api=model.objects["system_update_api"],
    )
    if args.smoke_seconds:
        if not args.output:
            parser.error("--output required for fresh smoke output")
        from benchmark_templates.native import run, Signal

        # 启动记录也保留真实可读位移；这些是原生观测，不是期望轨迹或功能判据。
        for key, body in model.objects["bodies"].items():
            if not body.IsFixed():
                for axis, direction in [
                    ("x", (1.0, 0.0, 0.0)),
                    ("y", (0.0, 1.0, 0.0)),
                    ("z", (0.0, 0.0, 1.0)),
                ]:
                    model.signals[key + ".position_" + axis + "_m"] = Signal(
                        body, "body_position", axis=direction
                    )
        result["source_geometry"] = {
            key: dict(
                mass_kg=r["mass_kg"],
                inertia_kg_m2=r["inertia_kg_m2"],
                source_solid_ids=r["source_solid_ids"],
                fixed=r["fixed"],
            )
            for key, r in model.objects["body_records"].items()
        }
        result["startup"] = run(
            model,
            [],
            duration_s=args.smoke_seconds,
            step_s=min(args.smoke_seconds, 0.0001),
            output=args.output,
        )
    if args.output and args.smoke_seconds:
        _write(Path(args.output) / "native_startup_evidence.json", result)
    print(json.dumps(result, ensure_ascii=False))


def prepare_authored_components(
    prepared_path, source_plan_path, output, *, preserve_original_bodies=()
):
    """使用原package哈希绑定分块和既有验证器生成新prepared，不覆盖大网格版本。

    完整源mesh、质量/惯量、分组、关节和内力字段保持；只恢复源family运行器的
    碰撞构造。没有原manifest或哈希不匹配即报错，不生成新的分块。
    """
    from types import SimpleNamespace
    import numpy as np
    import trimesh
    from simulation_ready_cad_workflow.validation_pipeline.intake.step_shape_cache import (
        load_step_shape,
    )
    from simulation_ready_cad_workflow.validation_pipeline.slim_harness.geometry import (
        _write_obj,
    )
    from simulation_ready_cad_workflow.validation_pipeline.mechanism.io import (
        sha256_file,
    )
    from simulation_ready_cad_workflow.validation_pipeline.mechanism.authored_collision_components import (
        load_authored_collision_components,
        bind_exported_component_meshes,
        classify_effective_source_convexity,
    )
    from simulation_ready_cad_workflow.validation_pipeline.mechanism.family_collision import (
        _export_solid_mesh,
    )
    from simulation_ready_cad_workflow.validation_pipeline.mechanism.static_geometry_proof_cache import (
        PersistentStaticGeometryProofCache,
    )

    prepared = copy.deepcopy(_read(prepared_path))
    raw_plan = _read(source_plan_path)
    plan = SimpleNamespace(**raw_plan)
    package = Path(prepared["cad_loader"]["origin"]["package_path"]).parent.resolve()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = package / "collision_components/manifest.json"
    expected_hash = plan.artifact_hashes.get("collision_components/manifest.json")
    if not manifest_path.is_file():
        raise SourcePhysicsError(
            "SOURCE_COMPONENT_MANIFEST_MISSING " + str(manifest_path)
        )
    actual_manifest_hash = sha256_file(manifest_path)
    if actual_manifest_hash != expected_hash:
        raise SourcePhysicsError(
            f"SOURCE_COMPONENT_MANIFEST_HASH_MISMATCH {manifest_path}: expected={expected_hash}, actual={actual_manifest_hash}"
        )
    assembly_relative = prepared["cad_loader"]["origin"]["package"]["assembly_file"]
    assembly = package / assembly_relative
    expected_assembly_hash = plan.artifact_hashes.get(assembly_relative)
    actual_assembly_hash = sha256_file(assembly)
    if actual_assembly_hash != expected_assembly_hash:
        raise SourcePhysicsError(
            f"SOURCE_ASSEMBLY_HASH_MISMATCH {assembly}: expected={expected_assembly_hash}, actual={actual_assembly_hash}"
        )
    source = load_step_shape(assembly, unit="MM")
    manifest = _read(manifest_path)
    declared = {row["source_solid_id"] for row in manifest["sources"]}
    records = {r["id"]: r for r in prepared["bodies"]}
    owners = {}
    source_meshes = {}
    required = {}
    fallback = {}
    fixed_for_trial = {}
    # 同一个已授权F03试架边界；原绑定也固定这些参考，不额外固定运动部件。
    if (
        prepared["cad_loader"]["origin"].get("source_fixture")
        == "bicycle_steering_at_rest"
    ):
        for law in prepared.get("legacy_source_laws", []):
            if law["kind"] == "bicycle_source_chain_steering_fixture":
                name = law["steering"]["fixed"]
                fixed_for_trial[name] = records[name]["fixed"]
                records[name]["fixed"] = True
    for key, r in records.items():
        rotation = trimesh.transformations.quaternion_matrix(r["rotation_wxyz"])[:3, :3]
        for comp in r["source_components"]:
            sid = comp["solid_id"]
            if sid in owners:
                raise SourcePhysicsError("AMBIGUOUS_COMPONENT_SOURCE_OWNER " + sid)
            owners[sid] = key
            if r.get("source_reference_only") or key in preserve_original_bodies:
                continue
            print(
                json.dumps(
                    {
                        "source_component_stage": "original_reference_mesh",
                        "body": key,
                        "solid": sid,
                    }
                ),
                flush=True,
            )
            raw_shape = source.solids[int(sid.split("_")[1]) - 1]
            # 原family后端自己的0.20mm/.50rad源网格；不混用slim的微米级audit网格做旧分块IoU。
            mesh = _export_solid_mesh(
                raw_shape, sid, output / "source_reference_meshes" / (sid + ".stl")
            )
            source_meshes[sid] = mesh
            convex = False
            if not r["fixed"]:
                convex = (
                    classify_effective_source_convexity(
                        mesh,
                        component_id=key + ":" + sid + ":exact_source",
                        collision_envelope_m=12e-6,
                    )["passed"]
                    is True
                )
            fallback[(key, sid)] = (mesh, convex)
            if sid in declared and not r["fixed"] and not convex:
                required[(key, sid)] = (raw_shape, mesh)
    if not required:
        raise SourcePhysicsError("SOURCE_MANIFEST_HAS_NO_ACTIVE_NONCONVEX_BOUND_PARTS")
    proofs = load_authored_collision_components(
        plan,
        package=package,
        workspace=Path(os.path.commonpath([str(package), str(output)])),
        expected_sources=required,
        scratch_root=output / "proof_assets",
        characteristic_length_m=plan.characteristic_length_m,
        mesh_linear_deflection_m=0.0002,
        collision_envelope_m=12e-6,
        proof_cache=PersistentStaticGeometryProofCache(
            output.parent / "source_component_proof_cache"
        ),
    )
    converted = {}
    reports = []
    for proof in proofs:
        record = records[proof.body_id]
        rotation = trimesh.transformations.quaternion_matrix(record["rotation_wxyz"])[
            :3, :3
        ]
        position = np.asarray(record["position_m"])
        entries = []
        worlds = {}
        paths = {}
        hashes = {}
        for comp in proof.components:
            local = comp.mesh_world_m.copy()
            local.vertices = (local.vertices - position) @ rotation
            path = output / "components" / proof.body_id / (comp.component_id + ".obj")
            result = _write_obj(path, local)
            exact = trimesh.load(path, force="mesh", process=False)
            exact.vertices = exact.vertices @ rotation.T + position
            worlds[comp.component_id] = exact
            paths[comp.component_id] = path
            hashes[comp.component_id] = result["sha256"]
            entries.append(
                {
                    "source_solid_id": proof.source_solid_id,
                    "component_id": comp.component_id,
                    "mesh_file": str(path),
                    "mesh_sha256": result["sha256"],
                    "convex": True,
                    "source_step_file": str(comp.step_path),
                    "source_step_sha256": comp.step_sha256,
                    "convexity_policy": dict(comp.convexity_certificate),
                }
            )
        _, report = bind_exported_component_meshes(
            proof, worlds, exported_asset_paths=paths, exported_asset_sha256=hashes
        )
        reports.append(report)
        converted[(proof.body_id, proof.source_solid_id)] = entries
    for key, r in records.items():
        if key in preserve_original_bodies:
            r.pop("source_component_colliders", None)
            r["source_collider_restoration_scope"] = (
                "Original full TriMesh retained explicitly; rejected authored decomposition was not used"
            )
            continue
        colliders = []
        if not r.get("source_reference_only"):
            for comp in r["source_components"]:
                entries = converted.get((key, comp["solid_id"]))
                if entries is not None:
                    colliders.extend(entries)
                else:
                    mesh, convex = fallback[(key, comp["solid_id"])]
                    local = mesh.copy()
                    rotation = trimesh.transformations.quaternion_matrix(
                        r["rotation_wxyz"]
                    )[:3, :3]
                    local.vertices = (
                        local.vertices - np.asarray(r["position_m"])
                    ) @ rotation
                    path = (
                        output
                        / "components"
                        / key
                        / ("original_" + comp["solid_id"] + ".obj")
                    )
                    export = _write_obj(path, local)
                    colliders.append(
                        {
                            "source_solid_id": comp["solid_id"],
                            "component_id": "unpartitioned_" + comp["solid_id"],
                            "mesh_file": str(path),
                            "mesh_sha256": export["sha256"],
                            "convex": convex,
                            "scope": "original family backend source meshing; no invented decomposition",
                        }
                    )
        r["source_component_colliders"] = colliders
    prepared["source_authored_component_reports"] = reports
    prepared["cad_loader"]["source_collider_restoration"] = {
        "source_plan": str(Path(source_plan_path).resolve()),
        "manifest_sha256": expected_hash,
        "original_prepared": str(Path(prepared_path).resolve()),
        "matched_source_solids": len(converted),
        "preserved_original_body_meshes": list(preserve_original_bodies),
        "fixed_for_existing_F03_fixture": fixed_for_trial,
        "source_mesh_policy": "family_collision._export_solid_mesh original defaults0.20mm/.50rad; full fine audit mesh unchanged",
        "collider_count": sum(
            len(r.get("source_component_colliders", ())) for r in records.values()
        ),
        "mass_inertia_grouping_internal_laws_changed": False,
        "validation_backend": "existing load_authored_collision_components + bind_exported_component_meshes; native make_family_body collider construction",
        "convexity_scope": "existing approved-generation convexity contract; exact existing helper evidence forwarded, not a new independent convexity proof",
    }
    prepared["cad_loader"][
        "scope"
    ] += "; source-authored collider components restored through original family backend; full CAD audit mesh retained; no source mechanics changed"
    _write(output / "prepared.json", prepared)
    return output / "prepared.json"


if __name__ == "__main__":
    main()
