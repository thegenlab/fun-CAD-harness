# Conditional sliding gear testcase

This harness fixture represents two external spur gears on parallel shafts.
The driver gear is free to slide axially on a sleeve. It starts above the
driven gear's face band, descends into axial overlap, and only then may the
finite compliant mesh law transmit torque.

The fixture is intentionally a test of the generic conditional-interface
contract rather than a special permission for gears:

- both endpoints bind to body-owned original-STEP component references;
- an initially inactive interface produces a CAD-stage warning, not a hard
  package error;
- geometry is evaluated at every runtime step;
- requested force or torque is replaced by exact zero outside `ENGAGED`;
- the engagement reference is captured on the transition into `ENGAGED`;
- full body-pair collision remains enabled;
- no ideal gear constraint participates in the physical-law fixture.

Run it with:

```text
python -m simulation_ready_cad_workflow.validation_pipeline.mechanism.conditional_interface_testcase
python -m simulation_ready_cad_workflow.validation_pipeline.mechanism.conditional_interface_testcase --backend pychrono
```

The command emits a JSON trace and returns non-zero if the driven gear moves
before engagement or fails to rotate after engagement. The PyChrono backend
uses finite 6-DOF bushings for both supports and asserts that the system owns
zero ideal links. It is a runtime-gate smoke test rather than collider or
certificate evidence.
