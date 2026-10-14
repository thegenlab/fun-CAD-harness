from __future__ import annotations

import unittest

import numpy as np

from simulation_ready_cad_workflow.validation_pipeline.dashboard.trace_pair_video import (
    _moving_chain_marker_positions_mm,
)


class TracePairVideoChainTests(unittest.TestCase):
    def test_markers_interpolate_hash_bound_saved_phase(self) -> None:
        contract = {
            "contract_sha256": "a" * 64,
            "centerline_points_abs_m": [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.1, 0.1, 0.0],
                [0.0, 0.1, 0.0],
                [0.0, 0.0, 0.0],
            ],
            "cumulative_lengths_m": [0.0, 0.1, 0.2, 0.3, 0.4],
            "total_length_m": 0.4,
            "visual_marker_count": 4,
            "material_motion_per_phase_sign": -1,
        }
        trace = {"moving_chain_path_contract": contract}
        samples = [
            {
                "time_s": 0.0,
                "moving_chain_path_state": {
                    "contract_sha256": "a" * 64,
                    "phase_unwrapped_m": 0.0,
                },
            },
            {
                "time_s": 1.0,
                "moving_chain_path_state": {
                    "contract_sha256": "a" * 64,
                    "phase_unwrapped_m": 0.05,
                },
            },
        ]

        markers = _moving_chain_marker_positions_mm(trace, samples, 0.5)

        self.assertEqual(markers.shape, (4, 3))
        np.testing.assert_allclose(markers[0], [0.0, 25.0, 0.0])
        np.testing.assert_allclose(markers[1], [75.0, 0.0, 0.0])
        np.testing.assert_allclose(markers[2], [100.0, 75.0, 0.0])
        np.testing.assert_allclose(markers[3], [25.0, 100.0, 0.0])

    def test_contract_hash_mismatch_is_not_rendered_as_evidence(self) -> None:
        trace = {
            "moving_chain_path_contract": {
                "contract_sha256": "a" * 64,
                "centerline_points_abs_m": [
                    [0.0, 0.0, 0.0],
                    [0.1, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
                "cumulative_lengths_m": [0.0, 0.1, 0.2],
                "total_length_m": 0.2,
                "visual_marker_count": 2,
                "material_motion_per_phase_sign": -1,
            }
        }
        samples = [
            {
                "time_s": 0.0,
                "moving_chain_path_state": {
                    "contract_sha256": "b" * 64,
                    "phase_unwrapped_m": 0.0,
                },
            }
        ]

        with self.assertRaisesRegex(ValueError, "another contract"):
            _moving_chain_marker_positions_mm(trace, samples, 0.0)


if __name__ == "__main__":
    unittest.main()
