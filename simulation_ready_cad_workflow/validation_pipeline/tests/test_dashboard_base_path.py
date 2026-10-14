from __future__ import annotations

import http.client
import json
import tempfile
import threading
import unittest
from pathlib import Path

from simulation_ready_cad_workflow.validation_pipeline.dashboard.render import render_live_shell
from simulation_ready_cad_workflow.validation_pipeline.dashboard.server import DashboardServer, maintenance_routes
from simulation_ready_cad_workflow.validation_pipeline.dashboard.url_paths import prefix_url_fields


class DashboardBasePathTests(unittest.TestCase):
    def test_media_links_are_prefixed_without_rewriting_source_or_artifact_paths(self):
        payload = {
            "preview": {"model_url": "/api/preview-file?run=one&name=model.glb"},
            "urls": ["/api/workbench-image?id=one", "https://example.com/model.glb"],
            "source": "/api/example is literal source text",
            "source_path": "/api/original.step",
        }
        result = prefix_url_fields(payload, "/harness/dashboard")
        self.assertEqual(result["preview"]["model_url"], "/harness/dashboard/api/preview-file?run=one&name=model.glb")
        self.assertEqual(result["urls"], ["/harness/dashboard/api/workbench-image?id=one", "https://example.com/model.glb"])
        self.assertEqual(result["source"], payload["source"])
        self.assertEqual(result["source_path"], payload["source_path"])
        self.assertEqual(payload["preview"]["model_url"], "/api/preview-file?run=one&name=model.glb")
        self.assertIs(prefix_url_fields(payload, ""), payload)

    def test_root_deployment_retains_root_assets_and_api(self):
        html = render_live_shell()
        self.assertIn('href="/static/app.css"', html)
        self.assertIn('"apiBase":"/api"', html)

    def test_prefixed_http_routes_assets_navigation_and_api(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = root / ".runtime_runs"
            runs.mkdir()
            server = DashboardServer(("127.0.0.1", 0), workspace_root=root, runs_root=runs, base_path="/harness/dashboard/")
            server.runs = lambda: []
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            connection = http.client.HTTPConnection(*server.server_address, timeout=5)
            try:
                # The legacy maintenance page exists only with the private Loop packages.
                for route in ["/", "/index.html?run=example"] + (["/maintenance"] if maintenance_routes else []):
                    connection.request("GET", "/harness/dashboard" + route)
                    response = connection.getresponse()
                    html = response.read().decode()
                    self.assertEqual(response.status, 200)
                    self.assertIn('href="/harness/dashboard/static/app.css"', html)
                    self.assertIn('"apiBase":"/harness/dashboard/api"', html)
                    self.assertIn('"basePath":"/harness/dashboard"', html)
                connection.request("GET", "/harness/dashboard/static/app.js")
                response = connection.getresponse()
                self.assertEqual(response.status, 200)
                self.assertIn(b"boot.basePath", response.read())
                connection.request("GET", "/harness/dashboard/api/runs")
                response = connection.getresponse()
                self.assertEqual(response.status, 200)
                self.assertEqual(json.loads(response.read())["runs"], [])
                connection.request("GET", "/harness/dashboard/maintenance/proposals")
                response = connection.getresponse()
                html=response.read()
                if maintenance_routes:
                    self.assertEqual(response.status, 200)
                    self.assertIn(b'"/harness/dashboard/api/runs"', html)
                    self.assertIn(b'"/harness/dashboard/maintenance?run=', html)
                    self.assertNotIn(b'"/maintenance?run=', html)
                else:
                    self.assertEqual(response.status, 404)
                connection.request("GET", "/harness/demo/")
                response = connection.getresponse()
                self.assertEqual(response.status, 404)
                response.read()
            finally:
                connection.close()
                server.shutdown()
                server.server_close()
                thread.join()


if __name__ == "__main__":
    unittest.main()
