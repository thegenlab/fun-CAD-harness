# Equivalent elasticity ABI v1

Scene and prepared data use `equivalent_elasticity` with `schema_version: "slim_equivalent_elasticity_v1"`, `units: "m"`, `elastic_reference_body_ids`, `flexible_contact_regions`, and `elements`.

Module: `equivalent_elasticity.py`. Pure validation: `validate_equivalent_elasticity(value, body_ids, joint_specs=()) -> None`. Construction: `build_equivalent_elasticity(c, system, bodies, joints, value) -> handle`. `bodies` maps IDs to actual Chrono bodies; `joints` is the canonical runtime mapping with `link`, `spec`, and live `active` state per joint. The handle exposes `metadata()`, `before_step(dt)`, `capture_native_step_loads()`, `after_step()`, `sample()`, `applied_load_records()` and `solved_load_records()`; the Harness owns integration and stepping.

Every element contains `id`, `kind`, `source_body_ids`, and `parameter_basis`. Compiler-owned parameter provenance may include geometry/material derivation and current source bindings. Working ranges describe the approximation's supported domain; they do not inject energy, clamp state, or impose ideal motion.

## torsional_spring

Fields: `body_a`, `body_b`, `anchor_abs_m[3]`, `axis_abs[3]` (unit), `stiffness_nm_rad > 0`, `rest_angle_rad`, `damping_nm_s_rad >= 0`, `working_range_rad[2]`.

q is body A angle minus body B angle about the common axis, referenced to the declared initial link frames. U=0.5*k*(q−q0)^2. Torque on A is −dU/dq minus passive damping; B receives the opposite. Stock RSDA's multi-turn angle failed the actual selected-runtime probe. This helper instead unwraps the actual relative body-frame pose once per accepted dynamics step and applies the passive two-ended torque through private native force accumulators, matching the verified multi-turn implementation. It does not use RSDA.GetAngle or an external constant actuator torque. Source/main-body mass is not added again.

## linear_spring (native central-force TSDA)

Fields: `body_a`, `body_b`, `point_a_body_m[3]`, `point_b_body_m[3]`, `stiffness_n_m > 0`, `rest_length_m >= 0`, `damping_n_s_m >= 0`, `working_range_m[2]`.

q is the actual distance between the two body-attached points. U=0.5*k*(q−L0)^2; forces act along the joining line. Native actuator force is zero. This is not silently substituted for a projected-axis spring.

## point_axis_spring

Fields: `body_a` (root), `body_b` (output), `point_a_body_m[3]`, `point_b_body_m[3]`, `axis_a_body[3]` (unit), `stiffness_n_m > 0`, `rest_displacement_m`, `damping_n_s_m >= 0`, `response: "bilateral" | "compression_only"`, `working_range_m[2]`.

q=eA·(pB−pA). For bilateral response x=q−q0; for compression-only x=min(q−q0,0). U=0.5*k*x². The implementation must use the full potential gradient, including the root-axis orientation derivative. Its equivalent force pair acts at pB: output receives −U′eA, root receives +U′eA, giving the correct root reaction moment. Compression-only damping must not make the contact attractive.

## friction_clutch

Fields: `joint_id`, `body_a`, `body_b`, `axis_a_body[3]`, `mu >= 0`, `effective_radius_m > 0`, `normal_stiffness_n_m > 0`, `preload_displacement_m >= 0`.

N=k_normal*preload and the native Coulomb torque bound is mu*N*r. V1 uses one Coulomb bound for static holding and sliding; it does not claim distinct static/kinetic coefficients. The existing revolute DOF is retained. The Harness uses the native shaft clutch with its verified inertia representation and PSOR configuration. These tool checks are not an author prerequisite.

## profile_detent

Fields: `body_a` (profile/root), `body_b` (slider), `point_a_body_m[3]`, `point_b_body_m[3]`, `axis_a_body[3]`, `stiffness_n_m > 0`, `preload_displacement_m >= 0`, `profile_reference_height_m`, `normal_damping_n_s_m >= 0`, `working_range_m[2]`, `profile_knots: [{"coordinate_m": q_i, "height_m": h_i}, ...]` with strictly increasing q.

q=eA·(pB−pA). h(q) is piecewise-linear and continuous from the source-derived static profile. Compression=max(0,preload+h(q)−h_ref); U=0.5*k*compression². Generalized force is −dU/dq and uses the same complete two-body gradient. No time, named OFF/ARM state, prescribed latch event or hysteresis switch is an input. At profile breakpoints use a documented one-sided segment derivative; record out-of-domain state instead of claiming source validation outside the supplied range.

## Scope and mass

Independent elastic source bodies keep their original mass/inertia and fixed-root joints as an explicitly declared lumped-root approximation. Their flexible collision representation is disabled and their unchanged shape is reference display; other rigid geometry audit remains. Mixed rigid/flexible bodies keep their rigid geometry and other contacts; only explicitly declared finite contact regions and pairs use the equivalent model.

Any shaft representation must preserve the total source inertia, including body+shaft kinetic energy. An unvalidated extra shaft inertia is not allowed. Ordinary ChShaft.SetInertia requires a positive value; do not call it with zero. The Harness partitions the original tensor between the body and native shaft, and restores its missing transverse Euler term with the zero-power body-local torque -J*(n dot omega)*(omega cross n). Native solver and inertia checks are owned by the Harness; the author supplies the physical nominal declaration and does not repeat tool calibration.

Each shaft port uses J equal to one eighth of the positive-definite rank-one bound and subtracts J*n*n^T from the body tensor. For general 3D rotation the helper also restores the missing transverse Euler term using body-local torque `-J*(n dot omega)*(omega cross n)`. It is transformed to world coordinates and recorded as `inertia_representation`; its instantaneous power is zero. It is not elastic actuation or external input work. The actual free-3D source/split comparison, including off-axis angular velocity, passed; total kinetic-energy equivalence is not inferred from a coaxial case alone.

These models preserve finite input work, two-ended reaction, stored potential and passive dissipation. They do not establish coil bind, actual large-deformation spring shape, local stress, fatigue or detailed contact pressure. All approved functional tests and actual impact measurements remain required.

## Integration and observation contract

Call `before_step(actual_dt)` after the caller has prepared its ordinary inputs, then `system.Update()`, then `capture_native_step_loads()`, then the **accepted** `DoStepDynamics(actual_dt)`, then `after_step()` and `sample()`. Capture caches actual native TSDA forces and application points before they are overwritten by the end-state update. Only `after_step` commits the torsional unwrap and time integrals. Duplicate/missing step pairs, native-time mismatch, or an accepted TSDA step without capture are errors. A failed native step must not be reported as accepted by calling `after_step`.

The helper allocates a separate accumulator per body and clears only those exact accumulator indices in `before_step`. It never clears the caller's input/gravity/load accumulators. Native TSDA and clutch forces are supplied by their native links, not reapplied manually. `applied_load_records()` gives this step's custom torsional/projected/profile forces and separately labeled inertia corrections. `solved_load_records()` after `after_step` gives the cached TSDA step forces/points plus native clutch GetReaction1/2 mapped to the two body axes and existing joint frames. Initial unsolved data returns an empty list; it does not manufacture zero force samples. Neither set is external input work. Clutch values describe the original source-body-plus-shaft port; shaft/body constraint getters with the known sign issue are not used.

If a clutch's existing revolute joint becomes inactive, `before_step` raises an explicit `MODEL_DOMAIN_EXIT` before any new dynamics step. No hidden shaft clutch continues across the released joint.

`sample()` returns schema `slim_equivalent_elastic_state_v1`, actual `time_s`, an `elements` dictionary keyed by element ID, and totals `potential_energy_j` / `dissipated_energy_j`. Element fields include `kind`, current `coordinate` / `rate`, current constitutive `generalized_force` where applicable, `deformation`, current potential, working-range status, cumulative `work_on_bodies_j`, `dissipated_energy_j`, initial potential and `discrete_energy_residual_j`. `applied_step_generalized_force` is null before a manual step exists, then holds the actual prior custom command; retain it in JSON but exclude that availability-dependent field from fixed-inventory numeric signal columns. TSDA `native_force_n` is explicitly labeled `current_state_constitutive`, not the applied prior-step force. Clutch adds N, torque limit, both native reactions, slip rate, any observed positive friction work, each shaft-minus-body speed error and axis-alignment error.

For elastic elements, `work_on_bodies_j` is a generalized-work quadrature using the step-start effort and actual coordinate increment; damping is integrated by trapezoidal rate. It is an integration diagnostic, not an exact full-body energy certificate. `discrete_energy_residual_j = U-U_initial+work+dissipation` is reported without clamping or energy top-up. Clutch work uses its native solved torque times the step's mean shaft slip; negative work accumulates as dissipation and positive work is separately exposed. The caller retains responsibility for total body/shaft kinetic energy and actual external-input work checks.
