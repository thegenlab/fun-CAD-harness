from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from ..cold_start import provider_transport as transport
from ..cold_start.worker import RESPONSE_SCHEMA
from ..runtime.codex_linux_sandbox import (
    app_thread_sandbox,
    app_turn_sandbox_policy,
    cli_sandbox_command,
    filesystem_sandbox_starts,
)


MODULE = "simulation_ready_cad_workflow.validation_pipeline.runtime.codex_linux_sandbox."


class CodexLinuxSandboxTests(unittest.TestCase):
    def test_unprivileged_and_sys_admin_keep_workspace_write(self):
        with patch(MODULE + "os.geteuid", return_value=1000):
            self.assertTrue(filesystem_sandbox_starts())
            self.assertEqual(app_thread_sandbox(readonly=False), "workspace-write")
            self.assertEqual(
                app_turn_sandbox_policy(readonly=False, writable_roots=["/work"]),
                {"type": "workspaceWrite", "writableRoots": ["/work"], "networkAccess": False},
            )
            self.assertEqual(cli_sandbox_command(readonly=True),
                             ["--sandbox", "read-only", "-c", 'approval_policy="never"'])
            self.assertEqual(cli_sandbox_command(readonly=False), [])
        with patch(MODULE + "os.geteuid", return_value=0), patch(
            MODULE + "Path.read_text", return_value="CapEff:\ta82425fb\n"
        ):
            # 0xa82425fb is the observed container mask plus SYS_ADMIN (bit 21).
            self.assertTrue(filesystem_sandbox_starts())

    def test_declared_container_isolation_skips_bubblewrap(self):
        with patch(MODULE + "os.name", "posix"), patch(MODULE + "os.geteuid", return_value=1000), \
                patch.dict("os.environ", {"SIMREADY_CONTAINER_ISOLATION": "1"}):
            self.assertFalse(filesystem_sandbox_starts())
            self.assertEqual(cli_sandbox_command(readonly=False),
                             ["--sandbox", "danger-full-access", "-c", 'approval_policy="never"'])

    def test_root_without_sys_admin_uses_full_access_so_tools_start(self):
        with patch(MODULE + "os.name", "posix"), patch(MODULE + "os.geteuid", return_value=0), patch(
            MODULE + "Path.read_text", return_value="CapEff:\ta80425fb\n"
        ):
            self.assertFalse(filesystem_sandbox_starts())
            self.assertEqual(app_thread_sandbox(readonly=True), "danger-full-access")
            self.assertEqual(app_turn_sandbox_policy(readonly=False, writable_roots=["/work"]),
                             {"type": "dangerFullAccess"})
            self.assertEqual(cli_sandbox_command(readonly=False),
                             ["--sandbox", "danger-full-access", "-c", 'approval_policy="never"'])

    def test_app_turn_sends_startable_policy_when_bwrap_cannot_remount(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        (root / "prompt.txt").write_text("fixture", encoding="utf-8")
        (root / "schema.json").write_text(json.dumps(RESPONSE_SCHEMA), encoding="utf-8")
        request = {
            "provider": "codex_app", "executable": "codex", "model": "fixture-model",
            "reasoning_effort": "medium", "cwd": str(root),
            "prompt_path": str(root / "prompt.txt"), "schema_path": str(root / "schema.json"),
            "readonly": False, "writable_roots": [str(root)], "session_id": None, "images": [],
        }
        events = [
            {"id": 1, "result": {}},
            {"id": 2, "result": {"thread": {"id": "owned-thread"}}},
            {"method": "item/completed", "params": {"threadId": "owned-thread", "turnId": "turn-1",
             "item": {"type": "agentMessage", "id": "answer", "text": json.dumps(
                 {"status": "ready", "summary": "fixture", "evidence_paths": [], "issues": []})}}},
            {"id": 3, "result": {"turn": {"id": "turn-1"}}},
            {"method": "turn/completed", "params": {"threadId": "owned-thread",
             "turn": {"id": "turn-1", "status": "completed"}}},
        ]
        child = Mock(stdin=io.StringIO(), stdout=io.StringIO("\n".join(json.dumps(e) for e in events) + "\n"),
                     poll=Mock(return_value=0), wait=Mock(return_value=0))
        with patch.object(transport, "_launch", return_value=child), patch.object(transport, "emit"), patch(
            "simulation_ready_cad_workflow.validation_pipeline.runtime.codex_linux_sandbox.filesystem_sandbox_starts",
            return_value=False,
        ):
            transport.app_turn(request)
        sent = [json.loads(line) for line in child.stdin.getvalue().splitlines()]
        self.assertEqual(sent[2]["params"]["sandbox"], "danger-full-access")
        self.assertEqual(sent[-1]["params"]["sandboxPolicy"], {"type": "dangerFullAccess"})

    def test_transport_file_loads_without_package_context(self):
        path = Path(transport.__file__).resolve()
        spec = importlib.util.spec_from_file_location("provider_transport_as_script", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.app_thread_sandbox(readonly=False), app_thread_sandbox(readonly=False))
        self.assertEqual(
            module.app_turn_sandbox_policy(readonly=True),
            app_turn_sandbox_policy(readonly=True),
        )


if __name__ == "__main__":
    unittest.main()
