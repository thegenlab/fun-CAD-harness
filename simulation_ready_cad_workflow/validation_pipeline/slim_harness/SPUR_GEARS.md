# Finite external spur-gear transmission

`scene.spur_gears` is an optional list of source-bound equivalent gear meshes.
It currently supports standard, unshifted external spur gears on parallel axes,
with each shaft supported by an existing revolute joint to fixed material.
All dimensions inside these rows use metres, even in a millimetre scene.

Each row supplies `id`, `body_a`, `body_b`, `joint_a`, `joint_b`, `center_a_m`,
`center_b_m`, unit `axis_abs`, integer `teeth_a`/`teeth_b` (at least 17),
`module_m`, `face_width_m`, `pressure_angle_rad`, `stiffness_n_m`,
`damping_n_s_m`, `center_tolerance_m`, `axis_tolerance_rad`, and a nonempty
`parameter_basis`. The source compiler checks the actual bearing axes,
pitch-center spacing, source tip radii, face interval and radial tooth count.
The nominal unshifted tip/root radii are `m*(z/2+1)` and `m*(z/2-1.25)`.
This source-profile check is not a manufacturing or involute-accuracy certificate.

The runtime creates a native `ChLinkLockGear` with `SetTeethCompliant` enabled.
It retains the original bodies and source inertias. Only contacts lying inside
both tooth annuli of the active declared pair are replaced. Shaft, collar,
other gear-level and unrelated material contacts remain in the native path;
the original-geometry motion audit is unchanged.

The gear permanently releases if its bearing releases, axes leave the declared
alignment tolerance, pitch-center spacing leaves the narrow tolerance, or the
gear faces lose axial overlap. Direct tooth contact then resumes. The current
helper does not attempt automatic re-engagement or a new meshing phase.

Friction uses the common native contact coefficient. A passive sampled Coulomb
force acts at nominal involute contact stations, with uniform sharing of the
preceding solved normal load. Equal/opposite forces act at the same world point;
their slip power is nonpositive, and a contact-effective-inertia bound avoids
over-stopping the relative slip in one isolated friction update. This remains
an approximation to the actual spatial tooth contact and load sharing.

The native continuous compliant law does not represent backlash. Source
backlash remains visible to geometry checks. Tooth impacts, local stress,
elastic tooth shape and fatigue are outside this representation's claim.
Report those limits and validate the approximation against a detailed reference
before making an accuracy claim.

Example settings for a 20:60 pair with module 2 mm and width 8 mm are in the
E06 preparation package. Its 1 MN/m normal stiffness and 40 N s/m damping are
declared nominal experimental settings, not calibrated material properties.

Physical traces include `spur_gears`, `gear.<id>.active`, native normal force,
gear rotations, friction loss, local contact exclusions and release events.
Native step time, loop time and trace sampling time are recorded separately.
These times are nested; do not add them together. The caller owns process,
loading, trace-commit and geometry-audit timing.
