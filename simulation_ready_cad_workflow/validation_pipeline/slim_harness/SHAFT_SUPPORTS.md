# Finite shaft supports

`equivalent_elasticity.elements` supports the explicit `kind: "shaft_support"`.
It creates no ideal joint. Two separated, clearance-aware radial spring/damper
stations carry the shaft; material points on opposed axial stops supply finite,
compression-only retention. Chrono integrates the actual original rigid bodies.
The constitutive forces use the same Harness-owned accumulator route as the
existing projected-axis equivalent elements. This is not a native hard revolute
or a `ChLoadBodyBodyBushingGeneric` with an ineffective Python override.

Use the common `id`, `body_a` (shaft), `body_b` (fixed support),
`source_body_ids`, and source-bound `parameter_basis` fields. The current
parameter basis must be `declared_nominal`, explicitly `not_calibrated`, with
its rationale, operating scope and source references. All fields below use SI
units, irrespective of `scene.units`.

- `axis_abs`: unit shaft direction; `anchor_abs_m`: a point on that axis.
- `bearings`: exactly two separated stations. Each has `id`, `center_abs_m`,
  `clearance_m`, `shaft_radius_m`, `stiffness_n_m`, `damping_n_s_m`, `friction`.
- `axial_stops`: material-point samples, including both axial directions. Each
  has `id`, `point_a_abs_m` on the original shaft retaining face, `normal_abs`,
  `gap_m`, `stiffness_n_m`, `damping_n_s_m`, `friction`. The normal points out
  of the support face toward the shaft face. Its matching support material point
  is `point_a_abs_m - gap_m * normal_abs`. Multiple points divide a declared
  face stiffness/damping and preserve the selected tilt/rotation approximation.
- `friction_regularization_m_s`: explicit positive slip smoothing speed.
- `release`: positive `radial_offset_m`, `axial_offset_m`, and `tilt_rad` bounds
  for irreversible loss of this support. Offsets are relative to the original
  assembled placement; radial displacement is checked at both bearing stations.

The radial law has zero normal force and friction inside the declared clearance.
Outside it, `N=max(0,k*(r-clearance)+c*r_dot)`. An axial stop supplies force only
after its actual material-point gap becomes negative. Friction opposes relative
tangential velocity at its application point, with the supplied coefficient.
A shared finite-impulse cap prevents the friction points collectively driving a
free shaft through a full stop. Recorded discrete energy residuals and any
positive friction work must still be inspected during timestep validation.

`spur_gears.joint_a` and `joint_b` may refer to these typed provider IDs instead
of ideal revolute IDs. Remove the replaced ideal joints. The gear compiler
checks actual body, source-axis and center agreement; native gear engagement
reads the provider's live `active` state.

Use the existing finite `flexible_contact_regions` for only the material regions
actually covered by this equivalent model. The compiler binds regions on this
pair to `active_element_id`. Both contact points must lie in the same region.
The region stays replaced while the support is valid, including open-clearance
states with zero reaction. Release withdraws all support/stop forces, restores
the original contacts, and makes dependent gears release. Stored elastic energy
at release is reported as dissipative failure loss. Dynamic original-geometry
audits must retain the sample's equivalent-element active states.

Activity is refreshed at both ends of an integration step. A final-step release
is visible in the last sample and its source-geometry audit, while the force
records of that completed interval remain unchanged. `within_support_range`
reports the actual geometric range; `within_working_range` also accepts the
defined released, zero-force branch. Gear `active` describes the endpoint and
its `normal_force_n` retains the reaction from the just-completed solver step.

This model does not certify local pressure distribution, elastic continuum
deformation, static sticking, bearing heating/wear, or the finite stop-point
sampling beyond its tested operating domain. It preserves original mass/inertia,
source geometry outside the explicit regions, external loads and test criteria.
