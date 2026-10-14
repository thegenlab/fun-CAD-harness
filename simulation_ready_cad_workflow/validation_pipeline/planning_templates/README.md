# Built-in planning-template library

This directory is the Harness-owned source of formally approved joint and
machine-task planning templates.

- `index.v1.json` is the closed catalog. Every admitted template has an exact
  kind, identifier, semantic version, path, and SHA-256 digest.
- `joints/` contains approved reusable joint and compound-joint templates.
- `task_suites/` contains one approved suite per machine family. A suite keeps
  the machine's tasks together and separately declares profiles, shared test
  scenarios, fixtures, joint-template bindings, acceptance criteria, and
  evidence obligations.
- `provenance/` contains immutable snapshots of approved legacy project inputs
  used to derive joint templates.
- `reviews/` contains review inventories only. Files there are not admitted
  templates, are excluded from the validator release, and are never returned
  by the catalog API.

The current formal catalog contains 11 joint templates and 13 machine task
suites. The suites contain 39 approved tasks grouped into 28 executable
scenarios. Tasks are acceptance goals; scenarios are shared executions that
may produce evidence for multiple tasks. General Harness checks such as
detachment, off-axis drift, and unintended penetration remain Harness checks
instead of being represented as artificial tasks.

Task approval, runtime availability, and historical evidence are independent:

- `approval.status = APPROVED` means the task definition was reviewed and is
  part of the formal library.
- `runtime_status = IMPLEMENTATION_REQUIRED` means the definition is admitted
  but the current runner or evaluator cannot yet authorize it.
- `runtime_status = ACTIVE` means the definition is approved for execution.
  Executability is decided by the validator registries, not by this label: the
  scenario's `test_program` must have an entry in the validator's program
  policies and every criterion's `evaluator_id` must have an implementation
  (metric registry, a family adapter module, or an approved reviewer task
  plugin). An ACTIVE scenario whose program or evaluator is unregistered fails
  closed at evaluation time.
- `source_projects[].evidence_status` describes only the historical exemplar.
  A failed exemplar does not invalidate an approved task definition. For
  example, the retractable-pen tasks are admitted while their historical source
  remains labeled `PHYSICAL_PREFLIGHT_FAILED`.
- Every criterion explicitly says whether it gates `PASS` or is only a
  `WARNING`; diagnostic observations never silently become hard thresholds.

## Legacy task-id map

Suite task identifiers were normalized at admission and often differ from the
task ids inside the source machines' `machine_task_mapping.json`. Use this map
when reading a historical package against the catalog:

| Source machine | Legacy task id (package) | Suite task id |
|---|---|---|
| auto_lock_utility_knife_v1 | extend_blade_to_working_position | task.extend_to_working_position |
| | auto_lock_blade_under_axial_load | task.hold_working_position_under_load |
| | select_second_blade_detent | task.select_second_position |
| | retract_blade_fully | task.retract_blade |
| bicycle (selfloop era) | steering_sweep | task.steering_sweep |
| | continuous_wheel_rotation | task.continuous_wheel_rotation (suite 1.0.0 only) |
| | powered_bicycle_drive_roundtrip | task.powered_chain_roundtrip |
| | rear_driven_ground_roll | task.rear_driven_ground_roll (suite 1.1.0) |
| four_bar_wiper_v1 | sweep_windshield | sweep_windshield |
| | return_wiper_home | return_wiper_home |
| hand_crank_egg_beater_v1 | spin_left_beater_from_crank | task.rotate_left_beater |
| | spin_right_beater_from_crank | task.rotate_right_beater |
| | counter_rotate_beater_pair | task.counter_rotate_beaters |
| | speed_up_beater_from_crank | task.speed_up_from_crank |
| hinged_toolbox_v1 | open_toolbox_lid | open_toolbox_lid |
| | close_toolbox_lid | close_toolbox_lid |
| mechanical_alarm_clock_v1 | (package ids unchanged) | run_three_hand_clock_logic, set_time_manually, ring_alarm_independently, set_alarm_time_by_turning_hour_hand, trigger_alarm_at_selected_hand_position |
| pedal_trash_can_v1 | open_lid_by_pedal | open_lid_by_pedal |
| | close_lid_after_release | return_lid_after_release |
| rack_pinion_gate_v1 | raise_gate | raise_gate |
| | lower_gate | lower_gate_to_bottom |
| | (added at suite 1.1.0, no legacy counterpart) | prevent_gate_drop_after_release |
| retractable_ballpoint_pen_v1 | first_click_extends_and_latches_tip | task.first_click_extend_and_latch |
| | second_click_retracts_tip | task.second_click_retract_and_latch |
| | repeat_two_complete_click_cycles | task.repeat_two_click_cycles |
| spring_kitchen_tongs_v1 | pickup_standard_test_block_from_table | tongs.pickup_standard_block |
| | hold_standard_test_block_above_table | tongs.hold_standard_block |
| | release_standard_test_block_to_table | tongs.release_standard_block |
| whole_folding_drying_rack_v1 | deploy_whole_drying_rack | task.deploy_rack |
| | fold_whole_drying_rack | task.fold_rack |

The public loader always uses this fixed code-owned root. Project-local
`mechanism_templates/`, environment variables, and caller-provided directories
cannot replace or shadow the catalog. Admission requires updating both the
template file and the closed catalog through review; copying a proposal into a
project does not register it.
