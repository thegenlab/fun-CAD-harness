# Equivalent elasticity

An explicit equivalent model can represent passive spring storage, projected-axis
elastic response, Coulomb clutch holding/sliding, and source-derived detent
potential. Actual rigid-body motion, finite input work, contact, and the approved
functional tests still run in the native solver. Existing code checks validate
source bindings, parameters and declared model limits. Model review is limited
to simulation-task changes; it does not re-review the mechanism or equivalent model.

The result is labeled "等效弹性模型下的功能验证". It does not establish actual
deformed spring surfaces, winding/self-contact clearance, stress, fatigue, or
manufactured preload. The original assembled geometry is audited in full.
During motion, only declared elastic reference bodies and explicit finite
pair-local flexible regions are excluded from original rigid geometry claims.
Reference bodies keep their source mass, inertia and declared root connections.
Mixed bodies keep their rigid parts and other contacts.

The program creates validation scope and source bindings. Do not supply a
`validation_scope` receipt in `scene.json`; it is derived from the declaration
and actual original source files. Changes to a registered test or its source
require fresh registration review under the normal workflow.

## Author data entry

Add an explicit `scene["equivalent_elasticity"]` object:

```text
schema_version: "slim_equivalent_elasticity_v1"
units: "m"
elastic_reference_body_ids: [actual source body IDs]
flexible_contact_regions: [explicit retained-body pair regions, or []]
elements: [the models below]
```

All new coordinates/lengths use metres, even when `scene.units` is `mm`. Body-relative points use the actual original rigid-body COM frame. The author must preserve the new declaration through the delivered modeling source/data, add any new source-data files to `authoring_sources`, and keep the original approved assertions. Do not paste a compiler receipt or scope signature as author evidence.

Every element has `id`, `kind`, `source_body_ids`, and `parameter_basis`. The complete force-law and sign ABI is [Equivalent force laws](EQUIVALENT_ELASTICITY_LAWS.md).

Finite clearance-aware shaft bearings and axial stops use the typed
[`shaft_support` provider](SHAFT_SUPPORTS.md), including activity-bound local
contact replacement and dependent gear release.

| kind | Required model fields |
| --- | --- |
| `torsional_spring` | `body_a`, `body_b`, `anchor_abs_m`, unit `axis_abs`, positive `stiffness_nm_rad`, explicit `rest_angle_rad`, nonnegative `damping_nm_s_rad`, `working_range_rad:[lo,hi]` |
| `point_axis_spring` | `body_a` root, `body_b` output, `point_a_body_m`, `point_b_body_m`, unit `axis_a_body`, positive `stiffness_n_m`, `rest_displacement_m`, nonnegative `damping_n_s_m`, `response:"bilateral"/"compression_only"`, `working_range_m` |
| `friction_clutch` | actual `joint_id` and its `body_a/body_b` revolute pair, unit `axis_a_body`, nonnegative `mu`, positive `effective_radius_m`, positive `normal_stiffness_n_m`, nonnegative `preload_displacement_m` |
| `linear_spring` | the two bodies/attached points, positive `stiffness_n_m`, nonnegative `rest_length_m` and `damping_n_s_m`, `working_range_m` |
| `profile_detent` | the two bodies/attached points and unit `axis_a_body`, positive `stiffness_n_m`, nonnegative `preload_displacement_m` and `normal_damping_n_s_m`, `profile_reference_height_m`, `working_range_m`, and `profile_source` below; the compiler generates `profile_knots` |

The three primary models are two-end torsion, projected-axis elasticity and passive friction. A central-force linear spring is explicitly separate from a projected-axis leaf model. The detent uses a static geometric profile, never named OFF/ARM states or a time/event script. Working ranges describe the model's domain; they do not clamp motion or supply energy.

## Nominal stiffness and source references

`parameter_basis` requires:

```text
method: "geometry_material_nominal" or "declared_nominal"
calibration: "not_calibrated"
rationale: nonempty explanation of the equivalent parameter choice
operating_scope: nonempty approximation/work-domain explanation
source_refs: [{path: design-relative source file, locator: source location}]
```

The compiler resolves and hashes actual source files. If a ref already supplies `sha256`, it must match. Output refs use absolute paths with current hashes. A source locator explains where the author obtained a nominal assumption; the compiler does not execute author Python or claim independent dimensional/manufacturing verification from a locator alone.

For `geometry_material_nominal`, add `derivation` with `young_modulus_pa`, `width_m`, bending `thickness_m`, effective `length_m`, optional integer `parallel_count` (default one), and one formula:

- `rectangular_strip_torsion`: `k = parallel_count * E * (width * thickness³ / 12) / length`.
- `cantilever_rectangular_leaf`: `k = parallel_count * 3 * E * (width * thickness³ / 12) / length³`.

These fill the applicable stiffness field, including `normal_stiffness_n_m` for the clutch. If the field is also supplied, it must agree with the declared calculation. `declared_nominal` instead requires the explicit positive stiffness value and source/rationale; it does not silently become a calibrated value. No full rest shape or FE calibration is required. The author supplies scalar rest/preload choices and a meaningful operating range.

## Actual STEP profile entry

`profile_source` currently supports:

```text
kind: "step_radius_follower"
source_body_id: one member of the actual prismatic pair
source_solid_ids: actual original STEP solid IDs
follower_center_at_source_pose_m: [x,y,z] in original STEP source coordinates
normal_direction_source: outward unit direction for follower contact search
follower_radius_m: explicit nominal radius
normal_offset_search_m: [inside/contact side, outside side] offsets along that normal
divisions: integer 2..256
source_refs: actual source evidence references
```

The two bodies must have one actual prismatic joint, and the model coordinate axis must be parallel to it. The compiler derives the original-pose coordinate from the two actual body points, scans the corresponding original STEP, and solves the radius-follower contact height at each coordinate. The normal interval must bracket contact. Heights are offsets from the declared source follower center. The resulting h(q) is continuous piecewise-linear data; knot spacing and normal root-solve width are recorded, while interpolation remains an explicit uncalibrated approximation.

## Local flexible regions

Each region contains `id`, `body_pair:[a,b]`, `region_owner_body`, `shape:"cylinder"`, `axis_origin_source_m`, `axis_direction_source`, `radius_m`, `axial_interval_m:[lo,hi]`, and `source_refs`. Coordinates are original STEP source coordinates, not COM-local coordinates. The owner must be an explicitly modeled source on that same retained pair. The compiler calls `make_validation_scope`; it does not remove the whole mixed body.
