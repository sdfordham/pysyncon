import unittest
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pysyncon import Dataprep, Synth, AugSynth
from pysyncon.sensitivity import run_sensitivity


class TestSensitivity(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(294)
        self.data = pd.DataFrame(
            [
                {
                    "unit": u,
                    "year": t,
                    "outcome": u
                    + 0.3 * t
                    + rng.normal(0, 0.2)
                    + (2 if u == 1 and t >= 7 else 0),
                    "x": u + rng.normal(0, 0.2),
                }
                for u in range(1, 5)
                for t in range(1, 11)
            ]
        )
        self.base = dict(
            predictors=["x"],
            predictors_op="mean",
            dependent="outcome",
            unit_variable="unit",
            time_variable="year",
            treatment_identifier=1,
            controls_identifier=[2, 3, 4],
            time_predictors_prior=range(1, 7),
            time_optimize_ssr=range(1, 7),
            special_predictors=[("outcome", [2, 4, 6], "mean")],
        )

    def run_study(self, specs=(), **kwargs):
        args = dict(
            data=self.data,
            base_dataprep=self.base,
            specifications=specs,
            treatment_time=7,
            att_periods=[7, 8, 9, 10],
        )
        args.update(kwargs)
        return run_sensitivity(**args)

    def test_baseline_matches_standalone_and_paths(self):
        for options in ({}, {"custom_V": np.array([0.4, 0.6])}):
            with self.subTest(options=options):
                result = self.run_study(fit_options=options)
                dp = Dataprep(foo=self.data, **self.base)
                model = Synth()
                model.fit(dp, **options)
                np.testing.assert_allclose(result.weights.weight, model.W)
                self.assertAlmostEqual(
                    result.summary.att[0], model.att(range(7, 11))["att"]
                )
                self.assertAlmostEqual(result.summary.fit_rmse[0] ** 2, model.mspe())
                wide = self.data.pivot(index="year", columns="unit", values="outcome")
                reconstructed = (
                    wide[result.weights.donor] @ result.weights.weight.to_numpy()
                )
                np.testing.assert_allclose(result.paths.synthetic, reconstructed)
                np.testing.assert_allclose(result.paths.gap, wide[1] - reconstructed)

    def test_order_and_independent_failures(self):
        result = self.run_study(
            [
                {"name": "reversed", "controls_identifier": [4, 3, 2]},
                {"name": "bad", "predictors": ["absent"]},
                {"name": "short", "time_optimize_ssr": [4, 5, 6]},
            ]
        )
        self.assertEqual(len(result.summary), 4)
        self.assertEqual(result.failures.spec_id.tolist(), ["bad"])
        self.assertAlmostEqual(result.summary.att[0], result.summary.att[1])
        np.testing.assert_allclose(
            result.weights.query("spec_id == 'baseline'").weight,
            result.weights.query("spec_id == 'reversed'").weight,
        )
        self.assertEqual(result.summary.iloc[-1].status, "ok")
        self.assertEqual(result.registry["reversed"]["controls_identifier"], [4, 3, 2])
        self.assertEqual(self.base["controls_identifier"], [2, 3, 4])

    def test_fixed_identity_and_pretreatment_guards(self):
        overrides = [
            {"dependent": "x"},
            {"treatment_identifier": 2},
            {"foo": self.data},
            {"time_optimize_ssr": [1, 7]},
            {"time_predictors_prior": [2, 8]},
            {"special_predictors": [("outcome", [3, 7], "mean")]},
            {"time_optimize_ssr": [1, 99]},
            {"controls_identifier": [2, 2]},
            {"controls_identifier": []},
            {"controls_identifier": [1, 2]},
            {"time_predictors_prior": []},
            {"predictors": [], "special_predictors": []},
        ]
        specs = [{"name": str(i), **override} for i, override in enumerate(overrides)]
        result = self.run_study(specs)
        self.assertEqual(len(result.failures), len(specs))
        self.assertEqual(len(result.paths), 10)

    def test_contaminated_donors(self):
        result = self.run_study(
            [{"name": "clean", "controls_identifier": [2, 3]}], excluded_donors=[4]
        )
        self.assertEqual(result.failures.spec_id.tolist(), ["baseline"])
        self.assertEqual(result.summary.status.tolist(), ["failed", "ok"])

    def test_missing_att_observation_does_not_shorten_period(self):
        data = self.data.copy()
        data.loc[(data.unit == 4) & (data.year == 8), "outcome"] = np.nan
        result = self.run_study(
            [{"name": "complete", "controls_identifier": [2, 3]}], data=data
        )
        self.assertEqual(result.summary.status.tolist(), ["failed", "ok"])
        self.assertEqual(result.metadata["att_periods"], [7, 8, 9, 10])

    def test_invalid_registry_or_study_raises(self):
        for kwargs in [
            dict(specifications=[{"name": "a"}, {"name": "a"}]),
            dict(att_periods=[]),
            dict(att_periods=[7, 7]),
            dict(att_periods=[6, 7]),
            dict(att_periods=[99]),
            dict(estimator=AugSynth),
            dict(fit_options={"X0": None}),
        ]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.run_study(**kwargs)

    def test_nonconvergence_is_visible_and_retains_estimate(self):
        result = self.run_study(fit_options={"optim_options": {"maxiter": 1}})
        self.assertEqual(result.summary.status.tolist(), ["warning"])
        self.assertTrue(np.isfinite(result.summary.att[0]))
        self.assertFalse(result.fits["baseline"].optimization_result.success)
        self.assertTrue(result.warnings.message.str.contains("did not converge").any())

    def test_inner_nonconvergence_is_visible(self):
        from scipy.optimize import minimize

        def failed(*args, **kwargs):
            res = minimize(*args, **kwargs)
            res.success, res.message = False, "test failure"
            return res

        with patch("pysyncon.base.minimize", side_effect=failed):
            result = self.run_study(fit_options={"custom_V": np.array([0.5, 0.5])})
        self.assertEqual(result.summary.status.tolist(), ["warning"])
        self.assertTrue(result.warnings.message.str.contains("Donor-weight").all())

    def test_all_failed_plot_and_dispersion(self):
        result = self.run_study(excluded_donors=[2, 3, 4])
        self.assertEqual(result.dispersion()["failed"], 1)
        self.assertTrue(np.isnan(result.dispersion()["positive_share"]))
        self.assertEqual(
            result.paths.columns.tolist(),
            ["spec_id", "time", "observed", "synthetic", "gap"],
        )
        ax = result.plot()
        self.assertEqual(len(ax.collections), 0)
        self.assertEqual([t.get_text() for t in ax.texts], ["failed"])
        plt.close(ax.figure)

    def test_no_post_outcome_information_used_for_weights(self):
        original = self.run_study()
        altered = self.data.copy()
        altered.loc[altered.year >= 7, "outcome"] += 5000
        updated = self.run_study(data=altered)
        np.testing.assert_array_equal(original.weights.weight, updated.weights.weight)

    def test_result_is_independent_of_input_mutation(self):
        result = self.run_study()
        self.base["predictors"].append("new")
        self.data.loc[:, "outcome"] = 0
        self.assertEqual(result.baseline["predictors"], ["x"])
        self.assertTrue((result.fits["baseline"].dataprep.foo.outcome != 0).any())
