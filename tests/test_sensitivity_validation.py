import unittest
from copy import deepcopy
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pysyncon import Dataprep, Synth
from pysyncon.sensitivity import run_sensitivity


class TestSensitivityValidation(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(46)
        donors = rng.normal(10, 2, (12, 3))
        observed = donors @ np.array([0.2, 0.3, 0.5])
        observed += np.sin(np.arange(12)) / 5
        observed[8:] += 3
        outcomes = np.column_stack([observed, donors])
        self.data = pd.DataFrame(
            [
                (u, t + 1, outcomes[t, u], outcomes[t, u] ** 2)
                for u in range(4)
                for t in range(12)
            ],
            columns=["unit", "time", "y", "x"],
        )
        self.base = dict(
            predictors=["x"],
            predictors_op="mean",
            dependent="y",
            unit_variable="unit",
            time_variable="time",
            treatment_identifier=0,
            controls_identifier=[1, 2, 3],
            time_predictors_prior=range(1, 9),
            time_optimize_ssr=range(1, 9),
            special_predictors=[("y", [2, 4, 6, 8], "mean")],
        )

    def study(self, **kwargs):
        args = dict(
            data=self.data,
            base_dataprep=self.base,
            specifications=[],
            treatment_time=9,
            att_periods=range(9, 13),
            validation_windows=[range(5, 9)],
        )
        args.update(kwargs)
        return run_sensitivity(**args)

    def test_manual_training_fit_predictions_and_full_refit_agree(self):
        result = self.study()
        cfg = deepcopy(self.base)
        cfg.update(
            time_predictors_prior=range(1, 5),
            time_optimize_ssr=range(1, 5),
            special_predictors=[("y", [2, 4], "mean")],
        )
        native = Synth()
        native.fit(dataprep=Dataprep(foo=self.data[self.data.time < 5], **cfg))
        np.testing.assert_allclose(result.validation_weights.weight, native.W)
        panel = self.data.pivot(index="time", columns="unit", values="y")
        expected = panel.loc[5:8, [1, 2, 3]] @ native.W
        np.testing.assert_allclose(result.validation_paths.synthetic, expected)
        score = np.sqrt(np.mean((panel.loc[5:8, 0] - expected) ** 2))
        self.assertAlmostEqual(result.validation.rmse[0], score)
        self.assertAlmostEqual(result.summary.validation_rmse[0], score)
        self.assertEqual(result.validation.status.tolist(), ["ok"])
        self.assertEqual(result.validation.n_validation.tolist(), [4])
        fit = result.validation_fits["baseline"]["fold_1"]
        self.assertLess(fit.dataprep.foo.time.max(), 5)
        full = self.study(validation_windows=None)
        pd.testing.assert_frame_equal(result.weights, full.weights)
        pd.testing.assert_frame_equal(result.paths, full.paths)
        pd.testing.assert_frame_equal(
            result.summary[full.summary.columns], full.summary
        )
        self.assertEqual(result.metadata["treatment_time"], 9)

    def test_heldout_outcomes_and_predictors_never_reach_training(self):
        original = self.study()
        altered = self.data.copy()
        selected = (altered.unit == 0) & altered.time.between(5, 8)
        altered.loc[selected, "y"] += 100
        altered.loc[altered.time >= 5, "x"] *= 1000
        updated = self.study(data=altered)
        pd.testing.assert_frame_equal(
            original.validation_weights, updated.validation_weights
        )
        np.testing.assert_array_equal(
            original.validation_paths.synthetic, updated.validation_paths.synthetic
        )
        self.assertGreater(updated.validation.rmse[0], original.validation.rmse[0] + 90)
        self.assertFalse(np.allclose(original.weights.weight, updated.weights.weight))
        self.assertEqual(
            updated.validation_registry["baseline"]["fold_1"]["special_predictors"],
            [("y", [2, 4], "mean")],
        )

    def test_post_outcomes_do_not_change_validation_even_when_missing(self):
        original = self.study()
        data = self.data.copy()
        data.loc[data.time >= 9, ["y", "x"]] = np.nan
        updated = self.study(data=data)
        pd.testing.assert_frame_equal(original.validation, updated.validation)
        pd.testing.assert_frame_equal(
            original.validation_paths, updated.validation_paths
        )
        pd.testing.assert_frame_equal(
            original.validation_weights, updated.validation_weights
        )
        self.assertEqual(updated.summary.status.tolist(), ["failed"])

    def test_rolling_windows_pool_errors_and_preserve_every_attempt(self):
        result = self.study(
            validation_windows=[[5], [6, 7, 8]],
            specifications=[
                {"name": "late", "time_optimize_ssr": [6, 7, 8]},
                {"name": "bad", "predictors": ["absent"]},
            ],
        )
        self.assertEqual(len(result.validation), 6)
        paths = result.validation_paths.query("spec_id == 'baseline'")
        expected = np.sqrt(np.mean(paths.gap**2))
        self.assertAlmostEqual(result.summary.validation_rmse[0], expected)
        self.assertTrue(result.summary.validation_rmse.iloc[1:].isna().all())
        self.assertEqual(
            result.summary.validation_status.tolist(), ["ok", "failed", "failed"]
        )
        self.assertEqual(result.validation.query("status == 'failed'").shape[0], 4)
        self.assertTrue(
            result.validation.query("status == 'failed'").message.str.len().gt(0).all()
        )
        self.assertEqual(result.metadata["validation_windows"], [[5], [6, 7, 8]])
        self.assertEqual(
            result.validation_registry["late"]["fold_1"]["time_optimize_ssr"], []
        )

    def test_one_failed_fold_has_no_partial_aggregate(self):
        result = self.study(
            validation_windows=[[4, 5], [6, 7, 8]],
            specifications=[
                {"name": "lag", "special_predictors": [("y", [5, 8], "mean")]},
            ],
        )
        folds = result.validation.query("spec_id == 'lag'")
        self.assertEqual(folds.status.tolist(), ["failed", "ok"])
        self.assertTrue(np.isnan(result.summary.validation_rmse[1]))
        self.assertTrue(np.isfinite(result.summary.att[1]))

    def test_invalid_windows_fail_before_any_estimation(self):
        for windows in ([], [[]], [[1]], [[8, 9]], [[5, 5]], [[99]], [[5, 6], [6, 7]]):
            with self.subTest(windows=windows), patch.object(Synth, "fit") as fit:
                with self.assertRaisesRegex(ValueError, "validation"):
                    self.study(validation_windows=windows)
                fit.assert_not_called()

    def test_invalid_designs_cannot_be_repaired_by_clipping(self):
        result = self.study(
            specifications=[
                {"name": "leaky", "special_predictors": [("y", [2, 10], "mean")]},
                {"name": "identity", "dependent": "x"},
            ]
        )
        self.assertEqual(result.validation.status.tolist(), ["ok", "failed", "failed"])
        self.assertEqual(
            result.summary.validation_status.tolist(), ["ok", "failed", "failed"]
        )

    def test_missing_validation_observation_is_a_failed_fold(self):
        self.data.loc[(self.data.unit == 1) & (self.data.time == 6), "y"] = np.nan
        result = self.study()
        self.assertEqual(result.validation.status.tolist(), ["failed"])
        self.assertTrue(np.isnan(result.summary.validation_rmse[0]))
        self.assertEqual(len(result.validation_weights), 3)
        self.assertIn("fold_1", result.validation_fits["baseline"])

    def test_declared_threshold_flags_subset_without_discarding_results(self):
        result = self.study(validation_max_rmse=0)
        self.assertFalse(result.summary.validation_pass[0])
        self.assertEqual(result.dispersion()["attempted"], 1)
        self.assertEqual(result.dispersion(validation_only=True)["attempted"], 0)
        result = self.study(validation_max_rmse=1000)
        self.assertTrue(result.summary.validation_pass[0])
        self.assertEqual(result.dispersion(validation_only=True)["attempted"], 1)
        for limit in (-1, np.nan, np.inf):
            with self.subTest(limit=limit), self.assertRaisesRegex(
                ValueError, "validation_max_rmse"
            ):
                self.study(validation_max_rmse=limit)
        with self.assertRaisesRegex(ValueError, "validation"):
            self.study(validation_windows=None, validation_max_rmse=1)
        with self.assertRaisesRegex(ValueError, "validation_max_rmse"):
            self.study().dispersion(validation_only=True)

    def test_validation_plot_labels_points_and_unplottable_designs(self):
        result = self.study(
            specifications=[{"name": "empty", "time_optimize_ssr": [7, 8]}]
        )
        ax = result.plot_validation()
        self.assertEqual(len(ax.collections[0].get_offsets()), 1)
        self.assertTrue(any("empty" in t.get_text() for t in ax.texts))
        plt.close(ax.figure)
        with self.assertRaisesRegex(ValueError, "validation"):
            self.study(validation_windows=None).plot_validation()

    def test_warning_folds_keep_scores_and_are_marked_in_the_plot(self):
        result = self.study(
            fit_options={"optim_options": {"maxiter": 1}}, validation_max_rmse=1
        )
        self.assertEqual(result.validation.status.tolist(), ["warning"])
        self.assertEqual(result.summary.validation_status.tolist(), ["warning"])
        self.assertFalse(result.validation_warnings.empty)
        self.assertTrue(np.isfinite(result.summary.validation_rmse[0]))
        fig, ax = plt.subplots()
        self.assertIs(result.plot_validation(ax), ax)
        self.assertEqual(len(ax.collections), 1)
        self.assertEqual(len(ax.lines), 2)
        plt.close(fig)

    def test_nonfinite_validation_gap_has_no_score_or_paths(self):
        self.data.loc[self.data.time.between(5, 8), "y"] = -1e308
        self.data.loc[(self.data.unit == 0) & self.data.time.between(5, 8), "y"] = 1e308
        result = self.study()
        self.assertEqual(result.validation.status.tolist(), ["failed"])
        self.assertIn("nonfinite trajectory", result.validation.message[0])
        self.assertTrue(result.validation_paths.empty)
        self.assertTrue(np.isnan(result.summary.validation_rmse[0]))
        ax = result.plot_validation()
        self.assertFalse(ax.collections)
        plt.close(ax.figure)

    def test_missing_training_outcomes_fail_validation(self):
        for options in ({}, {"custom_V": [0.5, 0.5]}):
            with self.subTest(options=options):
                panel = self.data.copy()
                panel.loc[(panel.unit == 1) & (panel.time == 3), "y"] = np.nan
                result = self.study(data=panel, fit_options=options)
                self.assertEqual(result.validation.status.tolist(), ["failed"])
                self.assertTrue(np.isnan(result.summary.validation_rmse.iloc[0]))
        panel = self.data.copy()
        panel.loc[(panel.unit == 1) & (panel.time == 1), "y"] = np.nan
        result = self.study(
            data=panel,
            specifications=[{"name": "short", "time_optimize_ssr": [2, 3, 4]}],
        )
        self.assertEqual(result.validation.status.tolist(), ["failed", "failed"])
        self.assertIn("fold_1", result.validation_fits["short"])

    def test_window_order_and_input_mutation_do_not_change_the_registered_study(self):
        windows = [[6, 7, 8], [4, 5]]
        result = self.study(validation_windows=windows)
        reference = self.study(validation_windows=list(reversed(windows)))
        pd.testing.assert_frame_equal(
            result.validation_weights, reference.validation_weights
        )
        windows[0].append(9)
        self.assertEqual(result.metadata["validation_windows"], [[4, 5], [6, 7, 8]])


if __name__ == "__main__":
    unittest.main()
