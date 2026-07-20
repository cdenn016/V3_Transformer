"""Standalone regression checks for the parameter-matched ablation sweep."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ablation


class ParameterMatchedSelectionTests(unittest.TestCase):

    def test_target_and_tolerance_validation_is_strict(self) -> None:
        self.assertEqual(ablation._validated_target_n_params(30_000_000), 30_000_000)
        self.assertEqual(ablation._validated_param_relative_deviation(0.02), 0.02)
        for value in (True, 0, -1, 3.5, "30000000"):
            with self.subTest(target=value), self.assertRaises(ValueError):
                ablation._validated_target_n_params(value)
        for value in (True, -0.01, 1.0, math.inf, "0.02"):
            with self.subTest(tolerance=value), self.assertRaises(ValueError):
                ablation._validated_param_relative_deviation(value)

    def test_grid_expands_in_declared_order_and_derives_kl_max(self) -> None:
        sweep = {
            "description": "fixture",
            "match_by": "embed_dim",
            "parameter_grid": {
                "embed_dim": [20, 24],
                "n_heads": [3, 4],
            },
        }
        self.assertEqual(
            ablation._parameter_grid_overrides(sweep),
            [
                {"embed_dim": 20, "n_heads": 3, "kl_max": 160},
                {"embed_dim": 20, "n_heads": 4, "kl_max": 160},
                {"embed_dim": 24, "n_heads": 3, "kl_max": 192},
                {"embed_dim": 24, "n_heads": 4, "kl_max": 192},
            ],
        )

    def test_grid_rejects_unknown_and_empty_fields(self) -> None:
        for grid in (
            {"not_a_config_field": [1], "embed_dim": [20]},
            {"embed_dim": [], "n_heads": [2]},
        ):
            with self.subTest(grid=grid), self.assertRaises(ValueError):
                ablation._parameter_grid_overrides({
                    "description": "fixture",
                    "match_by": "embed_dim",
                    "parameter_grid": grid,
                })

    def test_selector_keeps_one_closest_candidate_per_width(self) -> None:
        sweep_name = "parameter_fixture"
        sweep = {
            "description": "fixture",
            "match_by": "embed_dim",
            "parameter_grid": {
                "embed_dim": [20, 40],
                "n_heads": [2, 4],
            },
        }
        counts = {(20, 2): 98, (20, 4): 101, (40, 2): 105, (40, 4): 99}
        with patch.dict(ablation.SWEEPS, {sweep_name: sweep}), patch.dict(
            ablation.CONFIG,
            {
                "target_n_params": 100,
                "max_param_relative_deviation": 0.05,
            },
        ), patch.object(
            ablation,
            "_realized_n_params_for_overrides",
            side_effect=lambda ov: counts[(ov["embed_dim"], ov["n_heads"])],
        ):
            selection = ablation._parameter_match_selection(sweep_name)

        self.assertEqual(
            [
                (row["overrides"]["embed_dim"], row["overrides"]["n_heads"])
                for row in selection["selected"]
            ],
            [(20, 4), (40, 4)],
        )
        self.assertEqual(
            [row["param_difference"] for row in selection["selected"]],
            [1, -1],
        )
        self.assertEqual(
            [row["param_relative_deviation"] for row in selection["selected"]],
            [0.01, 0.01],
        )

    def test_selector_breaks_equal_distance_ties_by_declared_order(self) -> None:
        sweep_name = "parameter_tie_fixture"
        sweep = {
            "description": "fixture",
            "match_by": "embed_dim",
            "parameter_grid": {
                "embed_dim": [20, 40],
                "n_heads": [2, 4],
            },
        }
        counts = {(20, 2): 99, (20, 4): 101, (40, 2): 99, (40, 4): 101}
        with patch.dict(ablation.SWEEPS, {sweep_name: sweep}), patch.dict(
            ablation.CONFIG,
            {
                "target_n_params": 100,
                "max_param_relative_deviation": 0.02,
            },
        ), patch.object(
            ablation,
            "_realized_n_params_for_overrides",
            side_effect=lambda ov: counts[(ov["embed_dim"], ov["n_heads"])],
        ):
            selection = ablation._parameter_match_selection(sweep_name)

        self.assertEqual(
            [row["overrides"]["n_heads"] for row in selection["selected"]],
            [2, 2],
        )

    def test_selector_summarizes_invalid_pairs(self) -> None:
        sweep_name = "parameter_invalid_fixture"
        sweep = {
            "description": "fixture",
            "match_by": "embed_dim",
            "parameter_grid": {
                "embed_dim": [20, 24],
                "n_heads": [3, 4, 6],
            },
        }
        with patch.dict(ablation.SWEEPS, {sweep_name: sweep}), patch.dict(
            ablation.CONFIG,
            {
                "target_n_params": 100,
                "max_param_relative_deviation": 0.01,
            },
        ), patch.object(
            ablation,
            "_realized_n_params_for_overrides",
            return_value=100,
        ):
            selection = ablation._parameter_match_selection(sweep_name)

        rejected = selection["rejected"]
        self.assertTrue(any(row["reason"] == "config" for row in rejected))
        self.assertTrue(any("n_heads" in row["error"] for row in rejected))

    def test_selector_fails_with_closest_rows_when_too_few_widths_match(self) -> None:
        sweep_name = "parameter_sparse_fixture"
        sweep = {
            "description": "fixture",
            "match_by": "embed_dim",
            "parameter_grid": {
                "embed_dim": [20, 40],
                "n_heads": [2, 4],
            },
        }
        counts = {(20, 2): 120, (20, 4): 130, (40, 2): 140, (40, 4): 150}
        with patch.dict(ablation.SWEEPS, {sweep_name: sweep}), patch.dict(
            ablation.CONFIG,
            {
                "target_n_params": 100,
                "max_param_relative_deviation": 0.05,
            },
        ), patch.object(
            ablation,
            "_realized_n_params_for_overrides",
            side_effect=lambda ov: counts[(ov["embed_dim"], ov["n_heads"])],
        ):
            with self.assertRaisesRegex(ValueError, "closest rejected candidates") as caught:
                ablation._parameter_match_selection(sweep_name)
        self.assertIn("embed_dim=20", str(caught.exception))
        self.assertIn("embed_dim=40", str(caught.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
