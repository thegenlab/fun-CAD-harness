"""Contract tests for the installed mechanism-adapter registry.

The registry is the single dispatch authority after the data-driven
refactor: core worker/certificate/whole-machine paths iterate
``registered_mechanism_adapters()`` instead of matching family names.
These tests freeze the registry invariants that the dispatch depends on.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from simulation_ready_cad_workflow.validation_pipeline.mechanism import (
    family_plugins,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.ideal_comparison import (
    _fields_and_threshold,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.ideal_reference import (
    ideal_model_for_plan,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.models import (
    MechanismPlan,
    MechanismValidationError,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.physical_ideal_comparison import (
    _physical_behavior,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.adapters import (
    AdapterSpec,
    adapter_to_dict,
)


_HASH = "0" * 64
_PROJECT_ADAPTER_SOURCE = """\
from simulation_ready_cad_workflow.validation_pipeline.mechanism.family_plugins import MechanismAdapter, GeneralizedCoordinateSpec

class ProjectAdapter(MechanismAdapter):
    plugin_id = "project.example_v1"
    families = frozenset({"project_family"})
    coordinate = GeneralizedCoordinateSpec("position", "velocity", "m", "m/s", "force", "N")
    worker_backend = "project_example_v1"
    whole_machine_backend = "project_example_v1"
    wrench_policy_profiles = frozenset({"task_input", "support"})

    def load_cases(self, plan):
        return ()

    def expected_reduced_ideal_links(self, plan, adapter):
        return ()

MECHANISM_ADAPTER = ProjectAdapter()
"""


def _project_plan(package: Path, source: str) -> tuple[MechanismPlan, AdapterSpec]:
    relative = "mechanism_templates/project_adapter.py"
    path = package / relative
    path.parent.mkdir(parents=True)
    path.write_text(source, encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    adapter = AdapterSpec(
        adapter_id="project_data_v1",
        family="project_family",
        required_source_hashes=frozenset(),
        supported_programs=frozenset(),
        family_spec={"project_adapter_file": relative},
    )
    plan = MechanismPlan(
        package_dir=str(package),
        intake_report="intake.json",
        intake_report_sha256=_HASH,
        package_id="project_adapter_test",
        design_version="1.0.0",
        requirements_sha256=_HASH,
        package_binding_sha256=_HASH,
        capsule_id="project_capsule",
        capsule_file="capsule.json",
        family="project_family",
        protocol_id="project_protocol",
        fixed_group_ids=("fixed",),
        moving_group_ids=("moving",),
        axis_abs=(0.0, 0.0, 1.0),
        joint_origin_abs_m=(0.0, 0.0, 0.0),
        gravity_abs_m_s2=(0.0, 0.0, -9.81),
        joint_frame_id="joint",
        input_port_id="input",
        input_frame_id="input_frame",
        input_point_abs_m=(0.0, 0.0, 0.0),
        observable_id="output",
        observable_frame_id="output_frame",
        observable_point_abs_m=(0.0, 0.0, 0.0),
        task_ids=("task",),
        characteristic_length_m=0.1,
        moving_extents_m=(0.1, 0.1, 0.1),
        adapter_id=adapter.adapter_id,
        artifact_hashes={relative: digest},
        adapter_spec=adapter_to_dict(adapter),
    )
    return plan, adapter


class MechanismRegistryContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.plugins = family_plugins.registered_mechanism_adapters()

    def test_registry_is_not_empty_and_ids_are_unique(self) -> None:
        self.assertGreater(len(self.plugins), 0)
        ids = [plugin.plugin_id for plugin in self.plugins]
        self.assertEqual(
            len(ids),
            len(set(ids)),
            f"duplicate plugin_id registration detected: {ids}",
        )
        # The registration table is keyed by plugin_id; a duplicate would be
        # silently collapsed there, so the two views must stay the same size.
        self.assertEqual(
            len(family_plugins._PLUGIN_REGISTRATIONS), len(self.plugins)
        )

    def test_every_plugin_declares_required_capabilities(self) -> None:
        for plugin in self.plugins:
            with self.subTest(plugin_id=plugin.plugin_id):
                self.assertIsInstance(plugin.plugin_id, str)
                self.assertTrue(plugin.plugin_id)
                self.assertIsInstance(plugin.families, frozenset)
                self.assertTrue(plugin.families)
                self.assertIsInstance(plugin.worker_backend, str)
                self.assertTrue(plugin.worker_backend)
                self.assertIn(
                    plugin.whole_machine_artifact_branch,
                    {"reduced", "physical"},
                )
                for attribute in (
                    "initial_geometry_evidence_keys",
                    "required_trace_audit_sections",
                    "extra_family_spec_fields",
                    "implementation_capability_ids",
                    "model_boundary_ids",
                ):
                    value = getattr(plugin, attribute)
                    self.assertIsInstance(value, frozenset, attribute)
                    self.assertTrue(
                        all(isinstance(item, str) for item in value),
                        attribute,
                    )

    def test_family_spec_kind_ownership_is_unambiguous(self) -> None:
        owners: dict[object, str] = {}
        for plugin in self.plugins:
            for kind in plugin.family_spec_kinds:
                self.assertNotIn(
                    kind,
                    owners,
                    f"mechanism_kind {kind!r} is claimed by both "
                    f"{owners.get(kind)!r} and {plugin.plugin_id!r}",
                )
                owners[kind] = plugin.plugin_id

    def test_public_simulation_interface_catalog_lists_installed_plugins(
        self,
    ) -> None:
        catalog = family_plugins.public_simulation_interface_catalog()
        plugin_ids = [item["plugin_id"] for item in catalog["plugins"]]
        self.assertEqual(
            plugin_ids,
            sorted(plugin.plugin_id for plugin in self.plugins),
        )
        self.assertTrue(catalog["binding_rule"])
        self.assertIn("Do not write", catalog["document_role"])
        self.assertEqual(
            catalog["backend_contract"]["owner"],
            "Harness MechanismAdapter",
        )
        self.assertIn("PyChrono", catalog["backend_contract"]["execution"])
        for item in catalog["plugins"]:
            with self.subTest(plugin_id=item["plugin_id"]):
                self.assertNotIn(
                    "project_adapter_file",
                    item["extra_family_spec_fields"],
                )
                self.assertTrue(item["example_backend"]["worker_backend"])
                self.assertTrue(item["example_backend"]["whole_machine_backend"])
        clipper = next(
            item
            for item in catalog["plugins"]
            if item["plugin_id"] == "nail_clipper_compound_lever_v1"
        )
        self.assertEqual(
            clipper["mechanism_kinds"],
            ["nail_clipper_compound_lever"],
        )
        self.assertIn("restoring_spring", clipper["extra_family_spec_fields"])
        self.assertIn("revolute_joints", clipper["extra_family_spec_fields"])
        linkage = next(
            item
            for item in catalog["plugins"]
            if item["plugin_id"] == "revolute_linkage_v1"
        )
        self.assertIn(
            "compliant_surface_contacts",
            linkage["extra_family_spec_fields"],
        )
        chain = next(
            item
            for item in catalog["plugins"]
            if item["plugin_id"] == "bicycle_chain_sprocket_drive_v2"
        )
        self.assertIn(
            "chain.fixed_spatial_flex_zones",
            chain["implementation_capability_ids"],
        )
        self.assertIn(
            "chain.finite_compliant_force_path",
            chain["implementation_capability_ids"],
        )
        self.assertIn(
            "chain.no_full_flexible_body_physics",
            chain["model_boundary_ids"],
        )

    def test_plugins_claiming_spec_kinds_implement_parse_family_spec(
        self,
    ) -> None:
        base = family_plugins.MechanismAdapter.parse_family_spec
        for plugin in self.plugins:
            if not plugin.family_spec_kinds:
                continue
            with self.subTest(plugin_id=plugin.plugin_id):
                self.assertIsNot(
                    type(plugin).parse_family_spec,
                    base,
                    "plugin claims family_spec_kinds but inherits the "
                    "fail-closed parse_family_spec",
                )

    def test_plugin_without_spec_kinds_fails_closed_on_parse(self) -> None:
        for plugin in self.plugins:
            if plugin.family_spec_kinds:
                continue
            if type(plugin).parse_family_spec is not (
                family_plugins.MechanismAdapter.parse_family_spec
            ):
                continue
            with self.assertRaises(MechanismValidationError) as ctx:
                plugin.parse_family_spec({})
            self.assertEqual(ctx.exception.code, "MECHANISM_KIND_UNSUPPORTED")
            return
        self.skipTest("every registered plugin parses family specs")

    def test_registration_records_are_exact_and_active(self) -> None:
        for plugin in self.plugins:
            with self.subTest(plugin_id=plugin.plugin_id):
                record = family_plugins.plugin_registration_for(
                    plugin.plugin_id
                )
                self.assertEqual(
                    record,
                    {
                        "schema_version": (
                            "simulation_ready_cad_installed_plugin_v2"
                        ),
                        "plugin_id": plugin.plugin_id,
                        "status": "ACTIVE",
                        "registration_source": "installed_harness_code",
                    },
                )
                self.assertTrue(
                    family_plugins.registration_allows_runtime_execution(
                        record
                    )
                )

    def test_unknown_plugin_registration_fails_closed(self) -> None:
        with self.assertRaises(MechanismValidationError) as ctx:
            family_plugins.plugin_registration_for("no_such_plugin_v1")
        self.assertEqual(ctx.exception.code, "MECHANISM_PLUGIN_NOT_REGISTERED")

    def test_incompatible_registration_schema_fails_closed(self) -> None:
        plugin_id = self.plugins[0].plugin_id
        tampered = dict(family_plugins._PLUGIN_REGISTRATIONS)
        tampered[plugin_id] = {
            **tampered[plugin_id],
            "schema_version": "simulation_ready_cad_installed_plugin_v1",
        }
        with mock.patch.object(
            family_plugins, "_PLUGIN_REGISTRATIONS", tampered
        ):
            with self.assertRaises(MechanismValidationError) as ctx:
                family_plugins.plugin_registration_for(plugin_id)
        self.assertEqual(
            ctx.exception.code, "MECHANISM_PLUGIN_REGISTRATION_INVALID"
        )

    def test_candidate_lookups_fail_closed(self) -> None:
        self.assertEqual(
            family_plugins.candidate_mechanism_adapter_inventory(), []
        )
        with self.assertRaises(MechanismValidationError) as ctx:
            family_plugins.registered_candidate_definition("anything_v1")
        self.assertEqual(ctx.exception.code, "MECHANISM_PLUGIN_NOT_REGISTERED")

    def test_core_dispatch_modules_carry_no_family_literals(self) -> None:
        """Core dispatch paths must not compare against family/plugin names.

        Family-specific behavior belongs in the registered adapters; adding a
        mechanism must not require editing these shared modules.
        """

        import re
        from pathlib import Path

        mechanism_dir = Path(family_plugins.__file__).resolve().parent
        core_modules = (
            "worker.py",
            "certificate.py",
            "whole_machine.py",
            "whole_machine_worker.py",
            "discovery.py",
            "ideal_reference.py",
            "ideal_comparison.py",
            "physical_ideal_comparison.py",
            "task_motion_evidence.py",
        )
        # Structural coordinate vocabulary is generic, not a family name.
        allowed_literals = {"compound", "prismatic", "revolute"}
        pattern = re.compile(
            r"(?:plan\.family|\bfamily\b|mechanism_kind|plugin_id)"
            r"\s*[=!]=\s*['\"]([A-Za-z0-9_.-]+)['\"]"
        )
        offenders: list[str] = []
        for name in core_modules:
            source = (mechanism_dir / name).read_text(encoding="utf-8")
            for line_number, line in enumerate(source.splitlines(), start=1):
                for match in pattern.finditer(line):
                    if match.group(1) in allowed_literals:
                        continue
                    offenders.append(f"{name}:{line_number}: {line.strip()}")
        self.assertEqual(
            offenders,
            [],
            "family/plugin literal dispatch found in core modules:\n"
            + "\n".join(offenders),
        )

    def test_spec_kind_resolution_matches_registry_order(self) -> None:
        for plugin in self.plugins:
            for kind in plugin.family_spec_kinds:
                resolved = family_plugins.family_spec_plugin_for_kind(kind)
                self.assertIs(resolved, plugin)
        self.assertIsNone(
            family_plugins.family_spec_plugin_for_kind("no_such_kind_v1")
        )


class ProjectMechanismAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        family_plugins._PROJECT_ADAPTERS.clear()
        family_plugins._PROJECT_ADAPTER_CACHE.clear()

    def tearDown(self) -> None:
        family_plugins._PROJECT_ADAPTERS.clear()
        family_plugins._PROJECT_ADAPTER_CACHE.clear()

    def test_hash_bound_project_adapter_loads(self) -> None:
        built_in_count = len(family_plugins.registered_mechanism_adapters())
        with tempfile.TemporaryDirectory() as temp:
            plan, adapter = _project_plan(Path(temp), _PROJECT_ADAPTER_SOURCE)
            plugin, parsed_adapter = family_plugins.mechanism_adapter_for_plan(plan)
        self.assertEqual(plugin.plugin_id, "project.example_v1")
        self.assertEqual(parsed_adapter, adapter)
        self.assertEqual(
            len(family_plugins.registered_mechanism_adapters()),
            built_in_count,
        )

    def test_tampered_project_adapter_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            package = Path(temp)
            plan, adapter = _project_plan(package, _PROJECT_ADAPTER_SOURCE)
            (package / "mechanism_templates/project_adapter.py").write_text(
                _PROJECT_ADAPTER_SOURCE + "\n# changed\n", encoding="utf-8"
            )
            with self.assertRaises(MechanismValidationError) as ctx:
                family_plugins.mechanism_adapter_for(plan, adapter)
        self.assertEqual(
            ctx.exception.code, "PROJECT_MECHANISM_ADAPTER_HASH_MISMATCH"
        )

    def test_missing_adapter_export_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            plan, adapter = _project_plan(Path(temp), "VALUE = 1\n")
            with self.assertRaises(MechanismValidationError) as ctx:
                family_plugins.mechanism_adapter_for(plan, adapter)
        self.assertEqual(
            ctx.exception.code, "PROJECT_MECHANISM_ADAPTER_EXPORT_INVALID"
        )
        self.assertEqual(
            ctx.exception.failure_envelope["owner"],
            "worker_plan",
        )

    def test_registration_and_wrench_profiles_use_loaded_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            plan, adapter = _project_plan(Path(temp), _PROJECT_ADAPTER_SOURCE)
            plugin = family_plugins.mechanism_adapter_for(plan, adapter)
            registration = family_plugins.plugin_registration_for(plugin.plugin_id)
            policies = family_plugins.wrench_policies_for_plugin(plugin.plugin_id)
        self.assertTrue(
            family_plugins.registration_allows_runtime_execution(registration)
        )
        self.assertEqual(registration["registration_source"], "project_design_package")
        self.assertTrue(policies)
        self.assertTrue(
            all(policy.policy_id.startswith("project.example_v1.") for policy in policies)
        )

    def test_ideal_model_uses_project_adapter_for_current_plan(self) -> None:
        expected = mock.Mock(model_id="project.ideal_v1")
        expected.evaluate_samples = mock.Mock()
        with tempfile.TemporaryDirectory() as temp:
            plan, _adapter = _project_plan(Path(temp), _PROJECT_ADAPTER_SOURCE)
            plugin, _parsed = family_plugins.mechanism_adapter_for_plan(plan)
            with mock.patch.object(
                plugin, "ideal_model", return_value=expected
            ):
                model = ideal_model_for_plan(plan)
        self.assertIs(model, expected)

    def test_project_adapter_callback_failure_routes_to_project_author(self) -> None:
        source = _PROJECT_ADAPTER_SOURCE.replace(
            "    def load_cases(self, plan):",
            "    def ideal_model(self, plan, family_spec):\n"
            "        raise RuntimeError('project callback boom')\n\n"
            "    def load_cases(self, plan):",
        )
        with tempfile.TemporaryDirectory() as temp:
            plan, _adapter = _project_plan(Path(temp), source)
            with self.assertRaises(MechanismValidationError) as caught:
                ideal_model_for_plan(plan)

        failure = caught.exception.failure_envelope
        self.assertEqual(failure["owner"], "design_package")
        self.assertEqual(
            failure["source"]["registration_source"],
            "project_design_package",
        )

    def test_project_adapter_typed_rejection_survives_isolation_for_triage(
        self,
    ) -> None:
        source = _PROJECT_ADAPTER_SOURCE.replace(
            "from simulation_ready_cad_workflow.validation_pipeline.mechanism.family_plugins import MechanismAdapter, GeneralizedCoordinateSpec",
            "from simulation_ready_cad_workflow.validation_pipeline.mechanism.family_plugins import MechanismAdapter, GeneralizedCoordinateSpec\n"
            "from simulation_ready_cad_workflow.validation_pipeline.mechanism.models import MechanismValidationError",
        ).replace(
            "    def load_cases(self, plan):",
            "    def ideal_model(self, plan, family_spec):\n"
            "        raise MechanismValidationError('PROJECT_INPUT_REJECTED', 'bad input')\n\n"
            "    def load_cases(self, plan):",
        )
        with tempfile.TemporaryDirectory() as temp:
            plan, _adapter = _project_plan(Path(temp), source)
            with self.assertRaises(MechanismValidationError) as caught:
                ideal_model_for_plan(plan)

        failure = caught.exception.failure_envelope
        self.assertEqual(failure["owner"], "unclassified")
        self.assertEqual(
            failure["source"]["registration_source"],
            "project_design_package",
        )

    def test_isolated_project_supports_does_not_coerce_invalid_bool(self) -> None:
        source = _PROJECT_ADAPTER_SOURCE.replace(
            "    def load_cases(self, plan):",
            "    def supports(self, plan, adapter):\n"
            "        return 'yes'\n\n"
            "    def load_cases(self, plan):",
        )
        with tempfile.TemporaryDirectory() as temp:
            plan, _adapter = _project_plan(Path(temp), source)
            with self.assertRaises(MechanismValidationError) as caught:
                family_plugins.mechanism_adapter_for_plan(plan)

        self.assertEqual(
            caught.exception.failure_envelope["owner"],
            "design_package",
        )

    def test_ideal_comparison_uses_project_adapter_for_current_plan(self) -> None:
        expected = (("project_position",), 0.25, "m", "project_adapter")
        with tempfile.TemporaryDirectory() as temp:
            plan, _adapter = _project_plan(Path(temp), _PROJECT_ADAPTER_SOURCE)
            plugin, _parsed = family_plugins.mechanism_adapter_for_plan(plan)
            with mock.patch.object(
                plugin, "ideal_comparison_protocol", return_value=expected
            ):
                protocol = _fields_and_threshold(plan)
        self.assertEqual(protocol, expected)

    def test_physical_behavior_uses_project_adapter_for_current_plan(self) -> None:
        expected = ({"source": "project_adapter"}, {"limit": 0.5}, [])
        with tempfile.TemporaryDirectory() as temp:
            plan, _adapter = _project_plan(Path(temp), _PROJECT_ADAPTER_SOURCE)
            plugin, _parsed = family_plugins.mechanism_adapter_for_plan(plan)
            with mock.patch.object(
                plugin, "physical_behavior", return_value=expected
            ):
                behavior = _physical_behavior(
                    plan, mock.Mock(load_case_id="project-load"), {}
                )
        self.assertEqual(behavior, expected)

    def test_project_adapter_can_declare_an_existing_test_load_policy(self) -> None:
        source = _PROJECT_ADAPTER_SOURCE.replace(
            'frozenset({"task_input", "support"})',
            'frozenset({"passive_test_load"})',
        )
        with tempfile.TemporaryDirectory() as temp:
            plan, adapter = _project_plan(Path(temp), source)
            plugin = family_plugins.mechanism_adapter_for(plan, adapter)
            policies = family_plugins.wrench_policies_for_plugin(plugin.plugin_id)
        self.assertEqual(
            [policy.policy_id for policy in policies],
            [
                "project.example_v1.passive_test_load.force",
                "project.example_v1.passive_test_load.torque",
            ],
        )

    def test_project_adapter_rejects_an_unimplemented_wrench_policy(self) -> None:
        source = _PROJECT_ADAPTER_SOURCE.replace(
            'frozenset({"task_input", "support"})',
            'frozenset({"invented_policy"})',
        )
        with tempfile.TemporaryDirectory() as temp:
            plan, adapter = _project_plan(Path(temp), source)
            with self.assertRaises(MechanismValidationError) as caught:
                family_plugins.mechanism_adapter_for(plan, adapter)
        self.assertEqual(
            caught.exception.code,
            "PROJECT_MECHANISM_ADAPTER_WRENCH_PROFILES_INVALID",
        )

    def test_same_process_revisions_reuse_identical_adapter_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan_r01, adapter_r01 = _project_plan(
                root / "r01", _PROJECT_ADAPTER_SOURCE
            )
            plan_r02, adapter_r02 = _project_plan(
                root / "r02", _PROJECT_ADAPTER_SOURCE
            )
            plugin_r01 = family_plugins.mechanism_adapter_for(
                plan_r01, adapter_r01
            )
            plugin_r02 = family_plugins.mechanism_adapter_for(
                plan_r02, adapter_r02
            )
            plugin_r03 = family_plugins.mechanism_adapter_for(
                plan_r01, adapter_r01
            )
        self.assertEqual(plugin_r01.plugin_id, "project.example_v1")
        self.assertEqual(plugin_r02.plugin_id, plugin_r01.plugin_id)
        self.assertEqual(
            plugin_r01._expected_sha256, plugin_r02._expected_sha256
        )
        self.assertEqual(
            plugin_r01._package_dir.resolve(), (root / "r01").resolve()
        )
        self.assertEqual(
            plugin_r02._package_dir.resolve(), (root / "r02").resolve()
        )
        self.assertIsNot(plugin_r02, plugin_r01)
        self.assertIs(plugin_r03, plugin_r01)
        registration = family_plugins.plugin_registration_for(
            plugin_r01.plugin_id
        )
        self.assertEqual(
            registration["registration_source"], "project_design_package"
        )

    def test_same_process_revisions_rebind_updated_adapter_digest(self) -> None:
        updated = _PROJECT_ADAPTER_SOURCE.replace(
            'worker_backend = "project_example_v1"',
            'worker_backend = "project_example_v2"',
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan_r01, adapter_r01 = _project_plan(
                root / "r01", _PROJECT_ADAPTER_SOURCE
            )
            plan_r02, adapter_r02 = _project_plan(root / "r02", updated)
            plan_r03, adapter_r03 = _project_plan(root / "r03", updated)
            plugin_r01 = family_plugins.mechanism_adapter_for(
                plan_r01, adapter_r01
            )
            plugin_r02 = family_plugins.mechanism_adapter_for(
                plan_r02, adapter_r02
            )
            plugin_r03 = family_plugins.mechanism_adapter_for(
                plan_r03, adapter_r03
            )
            again_r01 = family_plugins.mechanism_adapter_for(
                plan_r01, adapter_r01
            )
        self.assertEqual(plugin_r01.plugin_id, "project.example_v1")
        self.assertEqual(plugin_r02.plugin_id, plugin_r01.plugin_id)
        self.assertEqual(plugin_r01.worker_backend, "project_example_v1")
        self.assertEqual(plugin_r02.worker_backend, "project_example_v2")
        self.assertEqual(plugin_r03.worker_backend, plugin_r02.worker_backend)
        self.assertEqual(
            plugin_r02._package_dir.resolve(), (root / "r02").resolve()
        )
        self.assertEqual(
            plugin_r03._package_dir.resolve(), (root / "r03").resolve()
        )
        self.assertIsNot(plugin_r02, plugin_r03)
        self.assertIs(again_r01, plugin_r01)
        self.assertIsNot(plugin_r01, plugin_r02)
        self.assertEqual(
            family_plugins.plugin_registration_for(plugin_r02.plugin_id)[
                "plugin_id"
            ],
            "project.example_v1",
        )

    def test_isolated_probe_times_out_on_import_loop(self) -> None:
        hanging = "while True:\n    pass\n"
        with tempfile.TemporaryDirectory() as temp:
            plan, adapter = _project_plan(Path(temp), hanging)
            with self.assertRaises(MechanismValidationError) as caught:
                family_plugins.load_project_mechanism_adapter(
                    plan.package_dir,
                    "mechanism_templates/project_adapter.py",
                    plan.artifact_hashes["mechanism_templates/project_adapter.py"],
                    probe_timeout_seconds=0.3,
                )
        self.assertEqual(caught.exception.code, "BOUNDED_SUBPROCESS_TIMEOUT")

    def test_isolated_probe_reports_missing_dependency(self) -> None:
        missing = "import definitely_not_a_simulation_ready_module\n"
        with tempfile.TemporaryDirectory() as temp:
            plan, adapter = _project_plan(Path(temp), missing)
            with self.assertRaises(MechanismValidationError) as caught:
                family_plugins.load_project_mechanism_adapter(
                    plan.package_dir,
                    "mechanism_templates/project_adapter.py",
                    plan.artifact_hashes["mechanism_templates/project_adapter.py"],
                    probe_timeout_seconds=5.0,
                )
        self.assertEqual(
            caught.exception.code, "PROJECT_MECHANISM_ADAPTER_LOAD_FAILED"
        )

    def test_isolated_callback_times_out(self) -> None:
        looping = _PROJECT_ADAPTER_SOURCE.replace(
            "def load_cases(self, plan):\n        return ()",
            "def load_cases(self, plan):\n        while True:\n            pass",
        )
        with tempfile.TemporaryDirectory() as temp:
            plan, adapter = _project_plan(Path(temp), looping)
            plugin = family_plugins.mechanism_adapter_for(plan, adapter)
            plugin._call_timeout_seconds = 0.3
            with self.assertRaises(MechanismValidationError) as caught:
                plugin.load_cases(plan)
        self.assertEqual(caught.exception.code, "BOUNDED_SUBPROCESS_TIMEOUT")

    def test_isolated_ideal_model_evaluate_samples_roundtrip(self) -> None:
        source = _PROJECT_ADAPTER_SOURCE.replace(
            "def load_cases(self, plan):\n        return ()",
            "def load_cases(self, plan):\n        return ()\n\n"
            "    def ideal_model(self, plan, family_spec):\n"
            "        class _Model:\n"
            "            model_id = 'project.example.ideal'\n"
            "            def evaluate_samples(self, plan, samples):\n"
            "                out = []\n"
            "                for sample in samples:\n"
            "                    item = dict(sample)\n"
            "                    item['ideal'] = True\n"
            "                    out.append(item)\n"
            "                return tuple(out), {'domain': 'test'}\n"
            "        return _Model()",
        )
        with tempfile.TemporaryDirectory() as temp:
            plan, adapter = _project_plan(Path(temp), source)
            plugin = family_plugins.mechanism_adapter_for(plan, adapter)
            model = plugin.ideal_model(plan, adapter.family_spec)
            samples, domain = model.evaluate_samples(
                plan, ({"t": 0.0}, {"t": 1.0})
            )
        self.assertEqual(domain, {"domain": "test"})
        self.assertEqual(
            [item["ideal"] for item in samples],
            [True, True],
        )

    def test_isolated_callback_rss_limit(self) -> None:
        bloated = _PROJECT_ADAPTER_SOURCE.replace(
            "def load_cases(self, plan):\n        return ()",
            "def load_cases(self, plan):\n"
            "        blob = bytearray(400 * 1024 * 1024)\n"
            "        import time\n"
            "        time.sleep(0.4)\n"
            "        return () if blob else ()",
        )
        with tempfile.TemporaryDirectory() as temp:
            plan, adapter = _project_plan(Path(temp), bloated)
            plugin = family_plugins.mechanism_adapter_for(plan, adapter)
            plugin._max_rss_bytes = 96 * 1024 * 1024
            with self.assertRaises(MechanismValidationError) as caught:
                plugin.load_cases(plan)
        self.assertEqual(caught.exception.code, "BOUNDED_SUBPROCESS_RSS")

    def test_isolated_load_cases_rejects_non_data_items(self) -> None:
        bad = _PROJECT_ADAPTER_SOURCE.replace(
            "def load_cases(self, plan):\n        return ()",
            "def load_cases(self, plan):\n        return (object(),)",
        )
        with tempfile.TemporaryDirectory() as temp:
            plan, adapter = _project_plan(Path(temp), bad)
            plugin = family_plugins.mechanism_adapter_for(plan, adapter)
            with self.assertRaises(MechanismValidationError) as caught:
                plugin.load_cases(plan)
        self.assertEqual(
            caught.exception.code, "PROJECT_ADAPTER_RESULT_NOT_DATA"
        )

    def test_adapter_call_timeout_is_longer_for_cad_lanes(self) -> None:
        from simulation_ready_cad_workflow.validation_pipeline.mechanism.project_adapter_isolation import (
            adapter_call_timeout_seconds,
        )

        self.assertEqual(adapter_call_timeout_seconds("load_cases", 60.0), 60.0)
        self.assertEqual(
            adapter_call_timeout_seconds("materialize", 60.0), 1800.0
        )
        self.assertEqual(
            adapter_call_timeout_seconds("simulate", 60.0), 1800.0
        )

    def test_isolated_parse_family_spec_exposes_bodies_and_to_dict(self) -> None:
        source = _PROJECT_ADAPTER_SOURCE.replace(
            "def load_cases(self, plan):\n        return ()",
            "def load_cases(self, plan):\n        return ()\n\n"
            "    def parse_family_spec(self, raw):\n"
            "        class ParsedSpec:\n"
            "            bodies = (\n"
            "                type('B', (), {"
            "'fixed': True, 'group_ids': ('g_fixed',)})(),\n"
            "                type('B', (), {"
            "'fixed': False, 'group_ids': ('g_move',)})(),\n"
            "            )\n"
            "            def to_dict(self):\n"
            "                return {'bodies': [\n"
            "                    {'body_id': 'fixed', 'fixed': True, "
            "'group_ids': ['g_fixed']},\n"
            "                    {'body_id': 'moving', 'fixed': False, "
            "'group_ids': ['g_move']},\n"
            "                ]}\n"
            "        return ParsedSpec()",
        )
        with tempfile.TemporaryDirectory() as temp:
            plan, adapter = _project_plan(Path(temp), source)
            plugin = family_plugins.mechanism_adapter_for(plan, adapter)
            parsed = plugin.parse_family_spec({"mechanism_kind": "project_family"})
        self.assertTrue(parsed.bodies[0].fixed)
        self.assertEqual(tuple(parsed.bodies[1].group_ids), ("g_move",))
        self.assertEqual(
            parsed.to_dict()["bodies"][0]["body_id"],
            "fixed",
        )


if __name__ == "__main__":
    unittest.main()
