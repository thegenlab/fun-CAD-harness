"""Build the hash-bound, trace-free bicycle-chain runtime fixture.

This is a Harness-owned runtime fixture, not a claim that the original r04
package contained collision decomposition assets.  It starts from the exact
r04 source package, applies explicit deterministic revisions to the bearings,
swept-envelope clearance, and authored visible-chain geometry, regenerates the
authoritative STEP from that revised generator, adds deterministic convex STEP
sidecars for the two moving solids, and binds the result to the bicycle plugin
installed in the current Harness.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import shutil
import stat
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))


_TIMESTAMP = datetime(2026, 7, 28, tzinfo=timezone.utc)
_SOURCE_PACKAGE_SHA256 = (
    "c0c3d5b5510085a5aca9484da1adb56a1591286121b0a1106ca930973c9234a0"
)
_SOURCE_ARTIFACTS = {
    "assembly.step": (
        "af69f1820c755d1e400be7d00cd2bd0eaec3a06c7fa88ded2bcb496c9b2e0e11"
    ),
    "assembly_contract.json": (
        "eadbe4f9a93badd5cd88cb684f0e4525e07d496df24e85d9c480d5b631a90354"
    ),
    "family_specs/bicycle_chain_sprocket_drive.json": (
        "f1dca1c7912bf286a9ed8cf2a85275b264454f83ca3d68c7c41d9ea08279d8b3"
    ),
    "generate_cleanroom_bicycle_drivetrain.py": (
        "9d618caa5ce306f84af76fd11f0df72b004df912bc843b9ad46376299262d974"
    ),
    "machine_task_mapping.json": (
        "00b485526afaed4b52283348e12423030d6979cd3c49350ea49b7378ba9a90aa"
    ),
    "mechanism_capsules/bicycle_drivetrain.json": (
        "48bae86da38645c35ab4274dd69c4c196800b7c7d855de8733c2baaa4cf2e690"
    ),
    "mechanism_coverage.json": (
        "466ffeed890f0b71acbb40fcb0428679749cf3919ff0da10cc5fc9e34a119895"
    ),
}
_REQUIREMENTS_SHA256 = (
    "e0d8ae61ba223266bc722e57d14261725a61ece380761680c887a69a2a31fce6"
)
_CASES = (
    "task_bicycle_powered_roundtrip",
    "nominal_powered_roundtrip",
    "held_out_positive_crank_load",
    "zero_input_energy_audit",
    "reverse_or_resisting_crank",
    "passive_return_to_home",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as stream:
        stream.write(
            json.dumps(
                value,
                ensure_ascii=True,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )


def _load_source_generator(path: Path) -> Any:
    # Load the environment's matching Expat before OCP/build123d can bind a
    # different same-SONAME copy through a transitive native dependency.
    import pyexpat  # noqa: F401

    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location(
        "_bicycle_r04_source_generator",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the r04 source generator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _replace_source_function(
    source: str,
    name: str,
    replacement: str,
) -> str:
    """Replace one top-level function in the copied deterministic generator."""

    marker = f"def {name}("
    start = source.find(marker)
    if start < 0 or (start and source[start - 1] != "\n"):
        raise RuntimeError(f"r04 generator lost function {name!r}")
    end = source.find("\ndef ", start + len(marker))
    if end < 0:
        raise RuntimeError(f"r04 generator function {name!r} is not bounded")
    return source[:start] + replacement.strip() + "\n\n" + source[end + 1 :]


def _apply_finite_bearing_clearance_revision(package: Path) -> Any:
    """Create the compact, explicitly revised activation demonstrator."""

    path = package / "generate_cleanroom_bicycle_drivetrain.py"
    source = path.read_text(encoding="utf-8")
    math_import = "from math import cos, degrees, pi, sin, sqrt"
    if source.count(math_import) != 1:
        raise RuntimeError("r04 generator lost its deterministic math import")
    source = source.replace(
        math_import,
        "from math import atan2, cos, degrees, pi, sin, sqrt",
        1,
    )
    source = _replace_source_function(
        source,
        "build_rear_drive_wheel",
        """
def build_rear_drive_wheel(angle_rad: float) -> Shape:
    # This is a validator-owned activation drive disc, not a replacement for
    # the submitted bicycle's visual wheel.  The exact 40/5 mm pitch torus
    # remains exposed by 0.25 mm beyond a carrier surrounding the unchanged
    # implementation-approved 36 x 14 mm backing disc.
    sprocket_center = Vector(
        REAR_CENTER.X,
        DRIVE_PLANE_Y_MM,
        REAR_CENTER.Z,
    )
    marker_center = radial_point(
        REAR_CENTER,
        110.0,
        angle_rad,
        y=0.0,
    )
    pieces = [
        cylinder_y(REAR_CENTER, 120.0, 2.0),
        cylinder_y(REAR_CENTER, 34.0, 112.0),
        cylinder_y(
            Vector(REAR_CENTER.X, -64.625, REAR_CENTER.Z),
            20.0,
            17.75,
        ),
        cylinder_y(sprocket_center, 36.0, 14.0),
        cylinder_y(sprocket_center, 44.75, 11.4),
        torus_y(
            sprocket_center,
            REAR_PITCH_RADIUS_MM,
            5.0,
        ),
        centered_box(
            marker_center,
            70.0,
            4.0,
            14.0,
            rotation_y_deg=-degrees(angle_rad),
        ),
    ]
    return fuse_all(pieces, label="activation_rear_drive_disc_and_sprocket")
""",
    )
    source = _replace_source_function(
        source,
        "build_input_crank",
        """
def build_input_crank(angle_rad: float) -> Shape:
    # A thin backing disc nearly fills the chainring torus.  Its remaining
    # 0.25 mm exposed toroidal lip keeps the exact STEP pitch surface while
    # allowing a small independently verified convex collision inventory.
    chainring_center = Vector(
        CRANK_CENTER.X,
        DRIVE_PLANE_Y_MM,
        CRANK_CENTER.Z,
    )
    right_start = Vector(CRANK_CENTER.X, -105.0, CRANK_CENTER.Z)
    right_end = radial_point(
        CRANK_CENTER, 170.0, angle_rad, y=-105.0
    )
    right_arm_end = radial_point(
        CRANK_CENTER, 127.0, angle_rad, y=-105.0
    )
    left_start = Vector(CRANK_CENTER.X, 105.0, CRANK_CENTER.Z)
    left_end = radial_point(
        CRANK_CENTER, 170.0, angle_rad + pi, y=105.0
    )
    left_arm_end = radial_point(
        CRANK_CENTER, 127.0, angle_rad + pi, y=105.0
    )
    return fuse_all(
        [
            cylinder_y(CRANK_CENTER, 13.0, 230.0),
            cylinder_y(chainring_center, 87.75, 15.5),
            torus_y(
                chainring_center,
                INPUT_PITCH_RADIUS_MM,
                8.0,
            ),
            tube(right_start, right_arm_end, 10.0),
            tube(left_start, left_arm_end, 10.0),
            centered_box(
                right_end,
                86.0,
                32.0,
                20.0,
                rotation_y_deg=-degrees(angle_rad),
            ),
            centered_box(
                left_end,
                86.0,
                32.0,
                20.0,
                rotation_y_deg=-degrees(angle_rad),
            ),
        ],
        label="activation_input_crank_chainring_and_pedals",
    )
""",
    )
    source = _replace_source_function(
        source,
        "build_chain_path_visualizer",
        """
def build_chain_path_visualizer() -> Shape:
    # The revised authored chain solid lies in the governed sprocket pitch
    # plane.  Its closed centerline, wrap, and thickness are measured by the
    # Harness and bind the runtime moving-chain path to the finite compliant
    # transmission model; it is not decorative offset geometry.
    chainring_center = Vector(
        CRANK_CENTER.X,
        DRIVE_PLANE_Y_MM,
        CRANK_CENTER.Z,
    )
    rear_sprocket_center = Vector(
        REAR_CENTER.X,
        DRIVE_PLANE_Y_MM,
        REAR_CENTER.Z,
    )
    dx = rear_sprocket_center.X - chainring_center.X
    dz = rear_sprocket_center.Z - chainring_center.Z
    center_distance = sqrt(dx * dx + dz * dz)
    basis_u_x = dx / center_distance
    basis_u_z = dz / center_distance
    basis_v_x = basis_u_z
    basis_v_z = -basis_u_x
    radial_u = (
        INPUT_PITCH_RADIUS_MM - REAR_PITCH_RADIUS_MM
    ) / center_distance
    radial_v = sqrt(1.0 - radial_u * radial_u)
    tangent_angle = atan2(radial_v, radial_u)

    def path_point(
        center: Vector,
        radius: float,
        angle: float,
    ) -> Vector:
        return Vector(
            center.X
            + radius
            * (basis_u_x * cos(angle) + basis_v_x * sin(angle)),
            DRIVE_PLANE_Y_MM,
            center.Z
            + radius
            * (basis_u_z * cos(angle) + basis_v_z * sin(angle)),
        )

    input_plus = path_point(
        chainring_center,
        INPUT_PITCH_RADIUS_MM,
        tangent_angle,
    )
    output_plus = path_point(
        rear_sprocket_center,
        REAR_PITCH_RADIUS_MM,
        tangent_angle,
    )
    output_sweep = -2.0 * tangent_angle
    output_arc = [
        path_point(
            rear_sprocket_center,
            REAR_PITCH_RADIUS_MM,
            tangent_angle + output_sweep * index / 48.0,
        )
        for index in range(49)
    ]
    input_minus = path_point(
        chainring_center,
        INPUT_PITCH_RADIUS_MM,
        -tangent_angle,
    )
    input_sweep = -(2.0 * pi - 2.0 * tangent_angle)
    input_arc = [
        path_point(
            chainring_center,
            INPUT_PITCH_RADIUS_MM,
            -tangent_angle + input_sweep * index / 48.0,
        )
        for index in range(49)
    ]
    centerline = [
        input_plus,
        output_plus,
        *output_arc[1:],
        input_minus,
        *input_arc[1:],
    ]
    segments = [
        extended_tube(left, right, 3.0, 0.75)
        for left, right in zip(centerline, centerline[1:])
    ]
    return fuse_all(segments, label="visible_closed_chain_path")
""",
    )
    replacements = (
        (
            "centered_box(Vector(0.0, 0.0, 25.0), 2150.0, 330.0, 50.0)",
            "centered_box(Vector(0.0, 0.0, 23.0), 2150.0, 330.0, 46.0)",
        ),
        (
            """    # Finite-bearing geometry around the moving crank and rear axle. The
    # torus holes leave visible radial clearance from the moving shafts.
""",
            """    # Finite annular bearing geometry around the moving crank and
    # rear axle leaves bounded radial clearance without collision exemptions.
""",
        ),
        (
            """    for y in (-64.0, 64.0):
        crank_bearing = Vector(CRANK_CENTER.X, y, CRANK_CENTER.Z)
        rear_bearing = Vector(REAR_CENTER.X, y, REAR_CENTER.Z)
        pieces.extend(
            [
                torus_y(crank_bearing, 32.0, 8.0),
                torus_y(rear_bearing, 34.0, 8.0),
                tube(
                    Vector(CRANK_CENTER.X, y, 45.0),
                    Vector(CRANK_CENTER.X, y, CRANK_CENTER.Z - 32.0),
                    22.0,
                ),
                tube(
                    Vector(REAR_CENTER.X, y, 45.0),
                    Vector(REAR_CENTER.X, y, REAR_CENTER.Z - 34.0),
                    22.0,
                ),
            ]
        )
""",
            """    # Both shafts retain visible finite annular bearing geometry.
    for y in (-40.0, 123.6):
        crank_bearing = Vector(CRANK_CENTER.X, y, CRANK_CENTER.Z)
        pieces.extend(
            [
                annulus_y(crank_bearing, 40.0, 13.6, 16.0),
                tube(
                    Vector(CRANK_CENTER.X, y, 45.0),
                    Vector(CRANK_CENTER.X, y, CRANK_CENTER.Z - 32.0),
                    22.0,
                ),
            ]
        )

    for y in (-40.0, 62.0):
        rear_bearing = Vector(REAR_CENTER.X, y, REAR_CENTER.Z)
        pieces.extend(
            [
                annulus_y(rear_bearing, 54.0, 34.6, 16.0),
                tube(
                    Vector(REAR_CENTER.X, y, 45.0),
                    Vector(REAR_CENTER.X, y, REAR_CENTER.Z - 42.0),
                    22.0,
                ),
            ]
        )
""",
        ),
        (
            "Vector(REAR_CENTER.X + 34.0, y, REAR_CENTER.Z)",
            "Vector(REAR_CENTER.X + 50.0, y, REAR_CENTER.Z)",
        ),
        (
            "Vector(REAR_CENTER.X, y, REAR_CENTER.Z + 34.0)",
            "Vector(REAR_CENTER.X, y, REAR_CENTER.Z + 50.0)",
        ),
    )
    for original, revised in replacements:
        if source.count(original) != 1:
            raise RuntimeError(
                "r04 generator no longer matches the activation-fixture "
                f"revision anchor: {original!r}"
            )
        source = source.replace(original, revised, 1)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(source)
    contract_path = package / "assembly_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    revised_evidence = {
        "input_crank_finite_bearing": (
            "Two fixed annular bearing rings surround the crank axle with "
            "visible radial clearance and are joined to the laboratory base."
        ),
        "rear_wheel_finite_bearing": (
            "Two fixed annular bearing rings surround the rear hub with "
            "visible radial clearance and are joined to the laboratory base."
        ),
        "bicycle_chain_transmission": (
            "The validator activation fixture exposes exact 80/8 mm and "
            "40/5 mm toroidal pitch faces on solid_2 and solid_3. Thin "
            "backing discs leave a 0.25 mm toroidal lip visible while "
            "independently verified convex sidecars remain within the "
            "unchanged collider-fidelity limits."
        ),
    }
    observed_connection_ids: set[str] = set()
    for connection in contract.get("connections", ()):
        connection_id = connection.get("connection_id")
        if connection_id in revised_evidence:
            connection["evidence"] = revised_evidence[connection_id]
            observed_connection_ids.add(connection_id)
    if observed_connection_ids != set(revised_evidence):
        raise RuntimeError(
            "r04 assembly contract lost a finite-bearing connection"
        )
    rear_parts = [
        part
        for part in contract.get("parts", ())
        if part.get("part_id") == "rear_drive_wheel"
    ]
    if len(rear_parts) != 1:
        raise RuntimeError("r04 assembly contract lost the rear wheel part")
    rear_parts[0]["declared_bounds_m"] = {
        "min": [-0.9700000001, -0.0850000001, 0.2599999999],
        "max": [-0.7049999999, 0.0560000001, 0.5000000001],
    }
    visible_chain_parts = [
        part
        for part in contract.get("parts", ())
        if part.get("part_id") == "visible_chain_path"
    ]
    if len(visible_chain_parts) != 1:
        raise RuntimeError("r04 assembly contract lost the visible chain part")
    visible_chain_part = visible_chain_parts[0]
    visible_chain_part.update(
        {
            "part_id": "visible_closed_chain_path",
            "occurrence_id": "visible_closed_chain_path_occurrence",
            "rigid_group_id": "visible_closed_chain_path_group",
            "declared_bounds_m": {
                "min": [-0.8930242525959582, -0.081, 0.25697447854502675],
                "max": [-0.1669752087151123, -0.075, 0.4230238336098046],
            },
            "material": {
                "name": "authored_chain_path_geometry",
                "density_kg_m3": 7800.0,
            },
        }
    )
    visible_chain_groups = [
        group
        for group in contract.get("rigid_groups", ())
        if group.get("rigid_group_id") == "visible_chain_path_group"
    ]
    if len(visible_chain_groups) != 1:
        raise RuntimeError("r04 assembly contract lost the visible chain group")
    visible_chain_groups[0].update(
        {
            "rigid_group_id": "visible_closed_chain_path_group",
            "part_ids": ["visible_closed_chain_path"],
            "connection_evidence": (
                "One continuous authored STEP solid follows exact external "
                "tangents and pitch-circle wraps in the governed sprocket "
                "plane. The Harness measures this geometry and binds it to "
                "the finite compliant moving-chain transmission model."
            ),
        }
    )
    visible_chain_frames = [
        frame
        for frame in contract.get("frames", ())
        if frame.get("frame_id") == "visible_chain_path_reference"
    ]
    if len(visible_chain_frames) != 1:
        raise RuntimeError("r04 assembly contract lost the visible chain frame")
    visible_chain_frames[0].update(
        {
            "frame_id": "visible_closed_chain_path_reference",
            "rigid_group_id": "visible_closed_chain_path_group",
            "origin_m": [-0.55, -0.078, 0.38],
        }
    )
    chain_connections = [
        connection
        for connection in contract.get("connections", ())
        if connection.get("connection_id") == "bicycle_chain_transmission"
    ]
    if len(chain_connections) != 1:
        raise RuntimeError("r04 assembly contract lost the chain transmission")
    chain_connections[0].update(
        {
            "connection_type": (
                "bicycle_chain_and_sprocket_finite_compliant_transmission"
            ),
            "evidence": (
                "Authoritative STEP solid_2 and solid_3 expose the governed "
                "80/8 mm and 40/5 mm pitch tori in the Y=-78 mm plane. "
                "Authoritative solid_4 is the uniquely declared closed chain "
                "path in that same plane; the Harness measures its seating, "
                "wrap, thickness, and clearance and binds those measurements "
                "to the finite compliant force coordinate and moving phase."
            ),
        }
    )
    _json(contract_path, contract)
    coverage_path = package / "mechanism_coverage.json"
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    visible_chain_coverage = [
        row
        for row in coverage.get("coverage", ())
        if row.get("part_id") == "visible_chain_path"
    ]
    if len(visible_chain_coverage) != 1:
        raise RuntimeError("r04 mechanism coverage lost the visible chain row")
    visible_chain_coverage[0]["part_id"] = "visible_closed_chain_path"
    _json(coverage_path, coverage)
    mapping_path = package / "machine_task_mapping.json"
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    mapping_text = json.dumps(mapping, ensure_ascii=True)
    old_claim = "and a context-only visible chain path."
    new_claim = (
        "and an authored closed chain path whose measured geometry is bound "
        "to the finite compliant transmission model."
    )
    if mapping_text.count(old_claim) != 1:
        raise RuntimeError("r04 task mapping lost the visible chain claim")
    mapping_text = mapping_text.replace(old_claim, new_claim, 1)
    _json(mapping_path, json.loads(mapping_text))
    family_spec_path = (
        package / "family_specs/bicycle_chain_sprocket_drive.json"
    )
    installed_spec_path = (
        _REPOSITORY_ROOT
        / "simulation_ready_cad_workflow"
        / "validation_pipeline"
        / "mechanism"
        / "templates"
        / "bicycle_chain_sprocket_drive_v2.json"
    )
    installed_spec = json.loads(
        installed_spec_path.read_text(encoding="utf-8")
    )
    if (
        installed_spec.get("mechanism_kind")
        != "bicycle_chain_sprocket_drive"
        or installed_spec.get("capability_state")
        != "INSTALLED_BUILTIN_ACTIVE"
    ):
        raise RuntimeError(
            "current Harness bicycle template is not the installed active spec"
        )
    _json(family_spec_path, installed_spec)
    generator = _load_source_generator(path)
    generator.write_step(
        generator.build_assembly(
            crank_angle_rad=0.0,
            rear_angle_rad=0.0,
        ),
        package / "assembly.step",
    )
    return generator


def _convex_hull_shape(points_mm: Iterable[Iterable[float]]) -> Any:
    import numpy as np
    from OCP.BRep import BRep_Builder
    from OCP.BRepBuilderAPI import (
        BRepBuilderAPI_MakeFace,
        BRepBuilderAPI_MakePolygon,
        BRepBuilderAPI_Sewing,
    )
    from OCP.ShapeFix import ShapeFix_Solid
    from OCP.TopoDS import TopoDS, TopoDS_Solid
    from OCP.gp import gp_Pnt
    from scipy.spatial import ConvexHull

    points = np.asarray(tuple(tuple(row) for row in points_mm), dtype=float)
    hull = ConvexHull(points)
    sewing = BRepBuilderAPI_Sewing(1.0e-7, True, True, True, False)
    for simplex, equation in zip(hull.simplices, hull.equations, strict=True):
        triangle = points[simplex]
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        if float(np.dot(normal, equation[:3])) < 0.0:
            triangle = triangle[[0, 2, 1]]
        polygon = BRepBuilderAPI_MakePolygon()
        for point in triangle:
            polygon.Add(gp_Pnt(*(float(value) for value in point)))
        polygon.Close()
        face = BRepBuilderAPI_MakeFace(polygon.Wire()).Face()
        sewing.Add(face)
    sewing.Perform()
    shell = TopoDS.Shell_s(sewing.SewedShape())
    builder = BRep_Builder()
    solid = TopoDS_Solid()
    builder.MakeSolid(solid)
    builder.Add(solid, shell)
    fixed = ShapeFix_Solid(solid)
    fixed.Perform()
    return fixed.Solid()


def _convex_cylinder_between(
    start: Any,
    end: Any,
    radius_mm: float,
    *,
    side_count: int = 48,
) -> Any:
    """Return a deterministic faceted cylinder inside the exact primitive."""

    start_xyz = tuple(float(value) for value in (start.X, start.Y, start.Z))
    end_xyz = tuple(float(value) for value in (end.X, end.Y, end.Z))
    direction = tuple(
        right - left for left, right in zip(start_xyz, end_xyz, strict=True)
    )
    length = math.sqrt(sum(value * value for value in direction))
    if length <= 0.0 or side_count < 12:
        raise ValueError("convex cylinder dimensions are invalid")
    axis = tuple(value / length for value in direction)
    reference = (
        (1.0, 0.0, 0.0)
        if abs(axis[0]) < 0.9
        else (0.0, 1.0, 0.0)
    )
    first = (
        axis[1] * reference[2] - axis[2] * reference[1],
        axis[2] * reference[0] - axis[0] * reference[2],
        axis[0] * reference[1] - axis[1] * reference[0],
    )
    first_length = math.sqrt(sum(value * value for value in first))
    first = tuple(value / first_length for value in first)
    second = (
        axis[1] * first[2] - axis[2] * first[1],
        axis[2] * first[0] - axis[0] * first[2],
        axis[0] * first[1] - axis[1] * first[0],
    )
    points = []
    for center in (start_xyz, end_xyz):
        for index in range(side_count):
            angle = 2.0 * math.pi * index / side_count
            points.append(
                tuple(
                    center[coordinate]
                    + radius_mm
                    * (
                        math.cos(angle) * first[coordinate]
                        + math.sin(angle) * second[coordinate]
                    )
                    for coordinate in range(3)
                )
            )
    return _convex_hull_shape(points)


def _convex_cylinder_y(
    center: Any,
    radius_mm: float,
    length_mm: float,
    *,
    side_count: int = 48,
) -> Any:
    half = 0.5 * float(length_mm)
    vector = type(center)
    return _convex_cylinder_between(
        vector(center.X, center.Y - half, center.Z),
        vector(center.X, center.Y + half, center.Z),
        radius_mm,
        side_count=side_count,
    )


def _convex_radial_profile_y(
    center: Any,
    profile: Iterable[tuple[float, float]],
    *,
    side_count: int = 64,
) -> Any:
    """Build one convex faceted body from ``(y_offset, radius)`` rings."""

    rings = tuple((float(offset), float(radius)) for offset, radius in profile)
    if len(rings) < 2 or side_count < 12:
        raise ValueError("convex radial profile dimensions are invalid")
    return _convex_hull_shape(
        (
            (
                float(center.X) + radius * math.cos(angle),
                float(center.Y) + offset,
                float(center.Z) + radius * math.sin(angle),
            )
            for offset, radius in rings
            for angle in (
                2.0 * math.pi * index / side_count
                for index in range(side_count)
            )
        )
    )


def _convex_centered_box(
    center: Any,
    x_length_mm: float,
    y_length_mm: float,
    z_length_mm: float,
) -> Any:
    half = (
        0.5 * float(x_length_mm),
        0.5 * float(y_length_mm),
        0.5 * float(z_length_mm),
    )
    return _convex_hull_shape(
        (
            (
                float(center.X) + x_sign * half[0],
                float(center.Y) + y_sign * half[1],
                float(center.Z) + z_sign * half[2],
            )
            for x_sign in (-1.0, 1.0)
            for y_sign in (-1.0, 1.0)
            for z_sign in (-1.0, 1.0)
        )
    )


def _annulus_segments(
    center: Any,
    *,
    outer_radius_mm: float,
    inner_radius_mm: float,
    length_mm: float,
    count: int,
) -> list[Any]:
    result: list[Any] = []
    for index in range(count):
        angles = (
            2.0 * math.pi * index / count,
            2.0 * math.pi * (index + 1) / count,
        )
        points = []
        for y_value in (
            float(center.Y) - length_mm / 2.0,
            float(center.Y) + length_mm / 2.0,
        ):
            for radius in (inner_radius_mm, outer_radius_mm):
                for angle in angles:
                    points.append(
                        (
                            float(center.X) + radius * math.cos(angle),
                            y_value,
                            float(center.Z) + radius * math.sin(angle),
                        )
                    )
        result.append(_convex_hull_shape(points))
    return result


def _torus_segments(
    center: Any,
    *,
    major_radius_mm: float,
    minor_radius_mm: float,
    major_count: int,
    minor_count: int,
) -> list[Any]:
    result: list[Any] = []
    circumscribed_minor = minor_radius_mm / math.cos(
        math.pi / minor_count
    )
    for index in range(major_count):
        theta_values = (
            2.0 * math.pi * index / major_count,
            2.0 * math.pi * (index + 1) / major_count,
        )
        points = []
        for theta in theta_values:
            for minor_index in range(minor_count):
                phi = (
                    2.0 * math.pi * minor_index / minor_count
                    + math.pi / minor_count
                )
                radial = (
                    major_radius_mm
                    + circumscribed_minor * math.cos(phi)
                )
                points.append(
                    (
                        float(center.X) + radial * math.cos(theta),
                        float(center.Y)
                        + circumscribed_minor * math.sin(phi),
                        float(center.Z) + radial * math.sin(theta),
                    )
                )
        result.append(_convex_hull_shape(points))
    return result


def _component_shapes(generator: Any) -> dict[str, list[tuple[str, Any]]]:
    Vector = generator.Vector
    pi = math.pi
    crank = generator.CRANK_CENTER
    rear = generator.REAR_CENTER
    drive_y = generator.DRIVE_PLANE_Y_MM
    crank_ring = Vector(crank.X, drive_y, crank.Z)
    rear_ring = Vector(rear.X, drive_y, rear.Z)

    input_components: list[tuple[str, Any]] = [
        (
            "input.shaft",
            _convex_cylinder_y(crank, 13.0, 230.0, side_count=32),
        ),
        (
            "input.chainring_backing",
            _convex_cylinder_y(
                crank_ring,
                87.75,
                15.5,
                side_count=64,
            ),
        ),
    ]
    right_start = Vector(crank.X, -105.0, crank.Z)
    right_end = generator.radial_point(crank, 170.0, 0.0, y=-105.0)
    right_arm_end = generator.radial_point(
        crank,
        127.0,
        0.0,
        y=-105.0,
    )
    left_start = Vector(crank.X, 105.0, crank.Z)
    left_end = generator.radial_point(crank, 170.0, pi, y=105.0)
    left_arm_end = generator.radial_point(
        crank,
        127.0,
        pi,
        y=105.0,
    )
    input_components.extend(
        [
            (
                "input.right_arm",
                _convex_cylinder_between(
                    right_start,
                    right_arm_end,
                    10.0,
                    side_count=32,
                ),
            ),
            (
                "input.left_arm",
                _convex_cylinder_between(
                    left_start,
                    left_arm_end,
                    10.0,
                    side_count=32,
                ),
            ),
            (
                "input.right_pedal",
                _convex_centered_box(
                    right_end,
                    86.0,
                    32.0,
                    20.0,
                ),
            ),
            (
                "input.left_pedal",
                _convex_centered_box(
                    left_end,
                    86.0,
                    32.0,
                    20.0,
                ),
            ),
        ]
    )

    rear_components: list[tuple[str, Any]] = [
        (
            "rear.drive_disc",
            _convex_cylinder_y(rear, 120.0, 2.0, side_count=64),
        ),
        (
            "rear.hub",
            _convex_cylinder_y(rear, 34.0, 112.0, side_count=48),
        ),
        (
            "rear.pitch_connector",
            _convex_cylinder_y(
                Vector(rear.X, -64.625, rear.Z),
                20.0,
                17.75,
                side_count=40,
            ),
        ),
        (
            "rear.sprocket_backing",
            _convex_cylinder_y(
                rear_ring,
                44.75,
                11.4,
                side_count=64,
            ),
        ),
        (
            "rear.sprocket_cap_negative",
            _convex_cylinder_y(
                Vector(rear_ring.X, rear_ring.Y - 6.35, rear_ring.Z),
                36.0,
                1.3,
                side_count=48,
            ),
        ),
        (
            "rear.sprocket_cap_positive",
            _convex_cylinder_y(
                Vector(rear_ring.X, rear_ring.Y + 6.35, rear_ring.Z),
                36.0,
                1.3,
                side_count=48,
            ),
        ),
        (
            "rear.index_marker",
            _convex_centered_box(
                generator.radial_point(rear, 110.0, 0.0, y=0.0),
                70.0,
                4.0,
                14.0,
            ),
        ),
    ]
    return {
        "input_crank": input_components,
        "rear_drive_wheel": rear_components,
    }


def _export_components(
    package: Path,
    generator: Any,
) -> list[dict[str, Any]]:
    from build123d import Shape, Solid, export_step
    from OCP.TopoDS import TopoDS

    source_ids = {
        "input_crank": "solid_2",
        "rear_drive_wheel": "solid_3",
    }
    sources: list[dict[str, Any]] = []
    for body_id, components in _component_shapes(generator).items():
        rows = []
        for component_id, raw_shape in components:
            relative = (
                f"collision_components/{body_id}/{component_id}.step"
            )
            destination = package / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(raw_shape, Shape):
                vertices, _triangles = raw_shape.tessellate(
                    0.05,
                    0.10,
                )
                raw_shape = _convex_hull_shape(
                    (float(item.X), float(item.Y), float(item.Z))
                    for item in vertices
                )
            shape = Solid(TopoDS.Solid_s(raw_shape))
            if shape is None:
                raise RuntimeError(
                    f"could not cast convex component: {component_id}"
                )
            if not export_step(
                shape,
                destination,
                timestamp=_TIMESTAMP,
            ):
                raise RuntimeError(f"STEP export failed: {relative}")
            rows.append(
                {
                    "component_id": component_id,
                    "step_file": relative,
                    "sha256": _sha256(destination),
                }
            )
        sources.append(
            {
                "body_id": body_id,
                "source_solid_id": source_ids[body_id],
                "components": rows,
            }
        )
    return sources


def _copy_source_package(source: Path, destination: Path) -> None:
    manifest = source / "package.json"
    if _sha256(manifest) != _SOURCE_PACKAGE_SHA256:
        raise RuntimeError("source package.json is not the exact r04 package")
    for relative, expected in _SOURCE_ARTIFACTS.items():
        source_file = source / relative
        if _sha256(source_file) != expected:
            raise RuntimeError(f"source artifact changed: {relative}")
        output = destination / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_file, output)


def _finalize_package(
    package: Path,
    sources: list[dict[str, Any]],
) -> None:
    component_manifest = {
        "schema_version": "simulation_ready_cad_collision_components_v1",
        "sources": sources,
    }
    component_manifest_path = package / "collision_components/manifest.json"
    _json(component_manifest_path, component_manifest)
    provenance = {
        "schema_version": "bicycle_chain_runtime_fixture_provenance_v1",
        "fixture_role": "HARNESS_CAPABILITY_ACTIVATION_ONLY",
        "original_r04_project_evidence_modified": False,
        "copied_contract_and_mapping_revised": True,
        "fixture_geometry_revised_from_original_r04": True,
        "authoritative_assembly_step_bytes_preserved": False,
        "original_authoritative_assembly_step_sha256": (
            _SOURCE_ARTIFACTS["assembly.step"]
        ),
        "authoritative_assembly_step_sha256": _sha256(
            package / "assembly.step"
        ),
        "source_generator_sha256": _sha256(
            package / "generate_cleanroom_bicycle_drivetrain.py"
        ),
        "engineering_revision": (
            "explicit validator activation revision with exact exposed "
            "80/8 mm and 40/5 mm pitch tori, finite bearing apertures, exact "
            "fixed-body zero-state binding, an asymmetric rear index marker, "
            "and a uniquely declared visible closed chain path in the "
            "governed pitch plane; no collision-pair exemption"
        ),
        "added_assets": (
            "thirteen compact convex STEP sidecars independently bounded by the "
            "unchanged collider-fidelity thresholds"
        ),
        "saved_trace_included": False,
        "design_worker_requirement": (
            "future real design revisions must author these sidecars themselves"
        ),
    }
    provenance_path = package / "fixture_provenance.json"
    _json(provenance_path, provenance)
    source_manifest = json.loads(
        (package.parent / "_source_package.json").read_text(encoding="utf-8")
    )
    source_manifest["design_version"] = "r04-minimal-activation-fixture-v4"
    artifacts = [
        {
            "path": relative,
            "sha256": _sha256(package / relative),
        }
        for relative in sorted(_SOURCE_ARTIFACTS)
    ]
    added = [
        component_manifest_path,
        provenance_path,
        *sorted(
            path
            for path in (package / "collision_components").rglob("*.step")
        ),
    ]
    artifacts.extend(
        {
            "path": path.relative_to(package).as_posix(),
            "sha256": _sha256(path),
        }
        for path in added
    )
    source_manifest["artifacts"] = sorted(
        artifacts,
        key=lambda item: item["path"],
    )
    _json(package / "package.json", source_manifest)


def _zip_payload(payload_root: Path, archive: Path) -> list[dict[str, Any]]:
    members: list[dict[str, Any]] = []
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        archive,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        strict_timestamps=True,
    ) as stream:
        for path in sorted(
            item for item in payload_root.rglob("*") if item.is_file()
        ):
            relative = path.relative_to(payload_root).as_posix()
            data = path.read_bytes()
            info = zipfile.ZipInfo(
                relative,
                date_time=(2026, 7, 28, 0, 0, 0),
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (stat.S_IFREG | 0o444) << 16
            info.create_system = 3
            stream.writestr(info, data, compresslevel=9)
            members.append(
                {
                    "path": relative,
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "size_bytes": len(data),
                }
            )
    return members


def _bind_materialized_preflight(
    *,
    payload: Path,
) -> dict[str, Any]:
    """Materialize static evidence from the current installed plugin."""

    from simulation_ready_cad_workflow.validation_pipeline.intake import (
        validate_design_package,
    )
    from simulation_ready_cad_workflow.validation_pipeline.mechanism.adapters import (
        adapter_to_dict,
    )
    from simulation_ready_cad_workflow.validation_pipeline.mechanism.bicycle_chain_spec import (
        BICYCLE_CHAIN_PLUGIN_ID,
        bicycle_chain_from_dict,
    )
    from simulation_ready_cad_workflow.validation_pipeline.mechanism.family_collision import (
        write_family_collision_preflight_snapshot,
    )
    from simulation_ready_cad_workflow.validation_pipeline.mechanism.discovery import (
        discover_mechanisms,
    )
    from simulation_ready_cad_workflow.validation_pipeline.mechanism.family_plugins import (
        mechanism_adapter_for_plan,
    )
    from simulation_ready_cad_workflow.validation_pipeline.mechanism.io import (
        canonical_sha256,
    )

    intake = validate_design_package(
        payload / "package",
        payload / "requirements.json",
    )
    intake_path = payload.parent / "preflight_intake_report.json"
    intake.write_json(intake_path)
    if intake.status.value not in {"ACCEPTED", "ACCEPTED_WITH_WARNINGS"}:
        raise RuntimeError(
            "generated fixture failed fresh intake: "
            + intake.render_text()
        )
    package = payload / "package"
    plans = discover_mechanisms(package, intake_path)
    if len(plans) != 1:
        raise RuntimeError("generated fixture must discover exactly one mechanism")
    plan = plans[0]
    plugin, unmaterialized_adapter = mechanism_adapter_for_plan(plan)
    if plugin.plugin_id != BICYCLE_CHAIN_PLUGIN_ID:
        raise RuntimeError("generated fixture did not select the bicycle plugin")
    spec = bicycle_chain_from_dict(unmaterialized_adapter.family_spec)

    destination_materialized = payload / "materialized"
    materialized_adapter = plugin.materialize(
        plan,
        unmaterialized_adapter,
        workspace_root=payload,
        output_dir=destination_materialized,
    )
    preflight = write_family_collision_preflight_snapshot(
        plan,
        unmaterialized_adapter,
        materialized_adapter,
        plugin_id=BICYCLE_CHAIN_PLUGIN_ID,
        expected_bodies=spec.bodies,
        workspace_root=payload,
        materialized_root=destination_materialized,
        preflight_root=payload / "collision_preflight",
    )
    materialized_adapter_sha256 = canonical_sha256(
        adapter_to_dict(materialized_adapter)
    )
    if (
        preflight["materialized_adapter_sha256"]
        != materialized_adapter_sha256
    ):
        raise RuntimeError(
            "preflight changed the canonical materialized adapter binding"
        )
    return {
        "manifest": "collision_preflight/preflight_manifest.json",
        "manifest_sha256": _sha256(
            payload
            / "collision_preflight"
            / "preflight_manifest.json"
        ),
        "materialized_adapter_sha256": preflight[
            "materialized_adapter_sha256"
        ],
        "snapshot": preflight,
    }


def build_fixture(
    *,
    source_package: Path,
    requirements: Path,
    output_dir: Path,
) -> None:
    if _sha256(requirements) != _REQUIREMENTS_SHA256:
        raise RuntimeError("requirements are not the exact r04 requirements")
    staging = output_dir / ".fixture-build"
    if staging.exists():
        shutil.rmtree(staging)
    payload = staging / "payload"
    package = payload / "package"
    package.mkdir(parents=True)
    # Retain source package.json outside package while rewriting only its
    # artifact inventory for the geometry-enriched fixture.
    shutil.copyfile(
        source_package / "package.json",
        payload / "_source_package.json",
    )
    _copy_source_package(source_package, package)
    shutil.copyfile(requirements, payload / "requirements.json")
    generator = _apply_finite_bearing_clearance_revision(
        package
    )
    sources = _export_components(package, generator)
    _finalize_package(package, sources)
    (payload / "_source_package.json").unlink()
    preflight = _bind_materialized_preflight(
        payload=payload,
    )
    archive = output_dir / "fixture_payload.zip"
    members = _zip_payload(payload, archive)
    manifest = {
        "schema_version": "bicycle_chain_runtime_fixture_v4",
        "self_contained": True,
        "live_pychrono_required": True,
        "fresh_intake_required_per_selector": True,
        "fresh_collision_materialization_required_per_selector": False,
        "release_bound_collision_preflight_required_per_selector": True,
        "fresh_pychrono_required_per_physical_selector": True,
        "collision_preflight_manifest": preflight["manifest"],
        "collision_preflight_manifest_sha256": preflight[
            "manifest_sha256"
        ],
        "saved_trace_files": [],
        "archive_file": archive.name,
        "archive_sha256": _sha256(archive),
        "archive_size_bytes": archive.stat().st_size,
        "member_count": len(members),
        "maximum_member_size_bytes": max(
            item["size_bytes"] for item in members
        ),
        "maximum_total_uncompressed_bytes": sum(
            item["size_bytes"] for item in members
        ),
        "members": members,
        "case_ids": list(_CASES),
        "source_authority": {
            "kind": "r04_derived_explicitly_revised_installed_runtime_fixture",
            "authoritative_assembly_step_sha256": _sha256(
                package / "assembly.step"
            ),
            "original_r04_package_claimed_to_contain_sidecars": False,
            "materialized_adapter_sha256": preflight[
                "materialized_adapter_sha256"
            ],
        },
    }
    _json(output_dir / "fixture_manifest.json", manifest)
    shutil.rmtree(staging)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-package", type=Path, required=True)
    parser.add_argument("--requirements", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "bicycle_chain_runtime_v2",
    )
    args = parser.parse_args()
    build_fixture(
        source_package=args.source_package.resolve(),
        requirements=args.requirements.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
