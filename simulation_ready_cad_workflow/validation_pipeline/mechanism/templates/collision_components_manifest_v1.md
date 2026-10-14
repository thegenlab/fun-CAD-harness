# Dynamic collision-component sidecar

Copy `collision_components_manifest_v1.json` to the fixed package path
`collision_components/manifest.json`.

Each listed STEP file is a design input, not evidence and not a Harness
decision. It must contain exactly one closed, positive-volume convex solid in
the same absolute CAD coordinates as `assembly.step`. Use multiple local
components for every listed fused dynamic source solid. A whole-solid convex
hull, an AABB/box replacement, a context solid, a reused component, or a
disabled collider is not acceptable.

List the manifest and every component STEP file in `package.json.artifacts`
with their real SHA-256 digests. The manifest must not contain `PASS`,
`PASSED`, `trusted`, fidelity claims, tolerances, proxy declarations, or any
other fields. Harness imports the component STEP files itself, exports the
actual PyChrono collision meshes, and independently checks convexity, source
ownership, bidirectional surface distance, OCCT volume/centroid/inertia,
occupancy, overlap, and the complete component union against the
intake-bound authoritative STEP solid.

`source_solid_id` is the package-global identity of the authoritative CAD
solid. Declare each `solid_N` at most once, even when several task capsules
give that solid different runtime body names. `body_id` records the declaring
capsule's owner; an installed Harness plugin may bind the same source solid to
another runtime body name, but it must repeat all independent component,
hash, convexity, ownership, and union-fidelity checks. A single manifest may
therefore contain the union of the component sources needed by multiple task
capsules. Unused entries never become runtime colliders for the current task.

Source-aware generators should keep the convex construction members and pass
them to `write_source_aware_components()` while exporting their fused result
as the authoritative assembly solid. Imported axis-aligned planar STEP solids
can use the certified authoring tool:

```bash
python -m simulation_ready_cad_workflow.validation_pipeline.tools.compile_design_package \
  design --write --collision-source BODY_ID:solid_N
```

The tool preserves `assembly.step`, attempts an exact B-rep grid partition
before optional CoACD proposals, and commits only a component set accepted by
the same Harness validator described above. Collision-representation failures
must not be repaired by replacing or simplifying visible assembly geometry.
