import unittest

import numpy as np
import pandas as pd

from pysyncon import Dataprep, AugSynth
from pysyncon.augsynth import VanillaOptimMixin
from pysyncon.utils import PlaceboTest


def make_dataprep(time_optimize_ssr=None):
    """Build a small deterministic panel: 1 treated unit (1) and 3 controls
    (2, 3, 4), 8 time periods (1..8, 6 pre-treatment and 2 post-treatment),
    and 2 predictor columns that can serve as covariates."""
    times = [1, 2, 3, 4, 5, 6, 7, 8]
    data = []
    for u in [1, 2, 3, 4]:
        effect = {1: 2.0, 2: 0.0, 3: 1.0, 4: -0.5}[u]
        for t in times:
            data.append(
                {
                    "name": u,
                    "time": t,
                    "dependent": effect + 0.3 * t + 0.1 * (t % 3),
                    "predictor1": effect + u + 0.05 * t,
                    "predictor2": (u + t) % 4 * 0.5,
                }
            )
    foo = pd.DataFrame(data)
    return Dataprep(
        foo=foo,
        predictors=["predictor1", "predictor2"],
        predictors_op="mean",
        time_predictors_prior=[1, 2, 3],
        dependent="dependent",
        unit_variable="name",
        time_variable="time",
        treatment_identifier=1,
        controls_identifier=[2, 3, 4],
        time_optimize_ssr=time_optimize_ssr
        if time_optimize_ssr is not None
        else [1, 2, 3, 4, 5, 6],
    )


class TestAugSynthRidgeUnits(unittest.TestCase):
    def setUp(self):
        self.dataprep = make_dataprep()
        self.augsynth = AugSynth()
        self.lambda_ = 0.5

    def design_mats(self):
        """(X0_stacked, X1_stacked, X0, X1, t0) as built in `fit` with
        `use_covariates=True`."""
        X0, X1 = self.dataprep.make_outcome_mats(
            time_period=self.augsynth.pre_periods
            if self.augsynth.pre_periods
            else list(self.dataprep.time_optimize_ssr)
        )
        Z0, Z1 = self.dataprep.make_covariate_mats()
        X0_d, X1_d, Z0_s, Z1_s = self.augsynth._normalize(X0, X1, Z0, Z1)
        return (
            pd.concat([X0_d, Z0_s], axis=0),
            pd.concat([X1_d, Z1_s]),
            X0,
            X1,
            X0.shape[0],
        )

    def test_normalize_centering(self):
        X0, X1 = self.dataprep.make_outcome_mats(
            time_period=list(self.dataprep.time_optimize_ssr)
        )
        Z0, Z1 = self.dataprep.make_covariate_mats()
        X0_d, X1_d, Z0_s, Z1_s = self.augsynth._normalize(X0, X1, Z0, Z1)

        # centering at the control mean (per row)
        pd.testing.assert_frame_equal(X0_d, X0.subtract(X0.mean(axis=1), axis=0))
        pd.testing.assert_series_equal(
            X1_d, X1.subtract(X0.mean(axis=1), axis=0).rename(X1.name)
        )
        Z0_c = Z0.subtract(Z0.mean(axis=1), axis=0)
        Z1_c = Z1.subtract(Z0.mean(axis=1), axis=0)
        sdx = X0_d.to_numpy().std(ddof=1)
        pd.testing.assert_frame_equal(Z0_s, Z0_c.divide(Z0_c.std(axis=1), axis=0) * sdx)
        pd.testing.assert_series_equal(
            Z1_s, Z1_c.divide(Z0_c.std(axis=1), axis=0) * sdx
        )
        # per-row means of the centered control outcomes are zero
        np.testing.assert_allclose(X0_d.mean(axis=1), 0.0, atol=1e-12)

    def test_generate_lambdas(self):
        X0_stacked, _, _, _, _ = self.design_mats()
        lambdas = self.augsynth.generate_lambdas(X0_stacked)
        self.assertEqual(len(lambdas), 21)
        sing = np.linalg.svd(X0_stacked, compute_uv=False)
        lambda_max = sing[0] ** 2.0
        scaler = (1e-8) ** (1 / 20)
        self.assertAlmostEqual(lambdas[0], lambda_max / scaler)
        self.assertAlmostEqual(lambdas[-1], lambda_max * scaler**19)
        self.assertTrue(np.all(np.diff(lambdas) < 0))

    def test_cross_validate(self):
        X0_stacked, X1_stacked, _, _, t0 = self.design_mats()
        lambdas = self.augsynth.generate_lambdas(X0_stacked)
        result = self.augsynth.cross_validate(X0_stacked, X1_stacked, lambdas, t0)

        # independent transcription of the fold loop
        errors = []
        for i in range(t0 - 1):
            hold = slice(i, i + 1)
            X0_t = X0_stacked.drop(index=X0_stacked.index[hold])
            X0_v = X0_stacked.iloc[hold]
            X1_t = X1_stacked.drop(index=X1_stacked.index[hold])
            X1_v = X1_stacked.iloc[hold]
            W, _ = VanillaOptimMixin.w_optimize(
                V_mat=np.eye(X0_t.shape[0]),
                X0=X0_t.to_numpy(),
                X1=X1_t.to_numpy(),
                qp_options={"maxiter": 2000, "ftol": 1e-12},
            )
            row = []
            for l in lambdas:
                ridge_weights = self.augsynth.solve_ridge(
                    A=X1_t, B=X0_t, W=W, lambda_=l
                )
                row.append(((X1_v - X0_v @ (W + ridge_weights)) ** 2).sum())
            errors.append(row)
        errors = np.array(errors)
        np.testing.assert_allclose(result.errors_mean, errors.mean(axis=0))
        np.testing.assert_allclose(
            result.errors_se, errors.std(axis=0, ddof=1) / np.sqrt(t0 - 1)
        )

    def test_cross_validate_raises(self):
        X0_stacked, X1_stacked, _, _, t0 = self.design_mats()
        lambdas = self.augsynth.generate_lambdas(X0_stacked)
        self.assertRaises(
            ValueError,
            self.augsynth.cross_validate,
            X0_stacked,
            X1_stacked,
            lambdas,
            t0,
            holdout_len=t0,
        )
        self.assertRaises(
            ValueError,
            self.augsynth.cross_validate,
            X0_stacked,
            X1_stacked,
            lambdas,
            t0,
            holdout_len=0,
        )

    def test_fit_reproduces_numpy_transcription(self):
        X0_stacked, X1_stacked, _, _, _ = self.design_mats()
        W, _ = VanillaOptimMixin.w_optimize(
            V_mat=np.eye(X0_stacked.shape[0]),
            X0=X0_stacked.to_numpy(),
            X1=X1_stacked.to_numpy(),
            qp_options={"maxiter": 2000, "ftol": 1e-12},
        )
        W_ridge = self.augsynth.solve_ridge(
            A=X1_stacked, B=X0_stacked, W=W, lambda_=self.lambda_
        )
        expected = W + W_ridge

        self.augsynth.fit(dataprep=self.dataprep, lambda_=self.lambda_)
        np.testing.assert_allclose(self.augsynth.W, expected, rtol=1e-10, atol=1e-10)

    def test_use_covariates_false(self):
        X0, X1 = self.dataprep.make_outcome_mats(
            time_period=list(self.dataprep.time_optimize_ssr)
        )
        X0_c = X0.subtract(X0.mean(axis=1), axis=0)
        X1_c = X1.subtract(X0.mean(axis=1), axis=0)
        W, _ = VanillaOptimMixin.w_optimize(
            V_mat=np.eye(X0_c.shape[0]),
            X0=X0_c.to_numpy(),
            X1=X1_c.to_numpy(),
            qp_options={"maxiter": 2000, "ftol": 1e-12},
        )
        W_ridge = self.augsynth.solve_ridge(A=X1_c, B=X0_c, W=W, lambda_=self.lambda_)
        expected = W + W_ridge

        self.augsynth.fit(
            dataprep=self.dataprep, lambda_=self.lambda_, use_covariates=False
        )
        np.testing.assert_allclose(self.augsynth.W, expected, rtol=1e-10, atol=1e-10)

    def test_ridge_mhat_and_bias(self):
        self.augsynth.fit(dataprep=self.dataprep, lambda_=self.lambda_)

        # independent transcription of the outcome model and bias estimate
        X0, X1 = self.dataprep.make_outcome_mats(time_period=self.augsynth.pre_periods)
        Y0, Y1 = self.dataprep.make_outcome_mats(time_period=self.augsynth.post_periods)
        Z0, Z1 = self.dataprep.make_covariate_mats()
        X0_stacked, _, _, _, _ = self.design_mats()

        design = X0_stacked.to_numpy()  # m x n_c (scaled covariates)
        y_c = Y0.subtract(Y0.mean(axis=1), axis=0).T.to_numpy()  # n_c x T_post
        N = np.linalg.inv(design @ design.T + self.lambda_ * np.eye(design.shape[0]))
        beta = N @ (design @ y_c)
        np.testing.assert_allclose(self.augsynth.beta, beta, rtol=1e-12)

        # evaluation matrix for all units: centered outcomes + the UNscaled
        # centered covariates (this replicates the R package exactly - the
        # design used for the weights and beta uses the scaled covariates)
        X_all_c = pd.concat(
            [
                X0.subtract(X0.mean(axis=1), axis=0),
                X1.subtract(X0.mean(axis=1), axis=0).rename(X1.name),
            ],
            axis=1,
        )
        Z0_c = Z0.subtract(Z0.mean(axis=1), axis=0)
        Z1_c = Z1.subtract(Z0.mean(axis=1), axis=0).rename(Z1.name)
        F_all = pd.concat([X_all_c, pd.concat([Z0_c, Z1_c], axis=1)], axis=0)
        expected_mhat = pd.DataFrame(
            F_all.T.to_numpy() @ beta,
            index=F_all.columns,
            columns=self.augsynth.post_periods,
        )
        pd.testing.assert_frame_equal(self.augsynth.ridge_mhat, expected_mhat)

        # bias_est = m1 - synw @ mhat[controls] (SCM-only weights)
        m1 = expected_mhat.loc[X1.name]
        m0 = expected_mhat.loc[list(X0.columns)]
        expected_bias = m1 - self.augsynth.synw @ m0
        pd.testing.assert_series_equal(self.augsynth.bias_est, expected_bias)
        self.assertAlmostEqual(self.augsynth.avg_bias, expected_bias.mean(), places=12)

    def test_ridge_mhat_no_covariates(self):
        self.augsynth.fit(
            dataprep=self.dataprep, lambda_=self.lambda_, use_covariates=False
        )

        X0, X1 = self.dataprep.make_outcome_mats(time_period=self.augsynth.pre_periods)
        Y0, Y1 = self.dataprep.make_outcome_mats(time_period=self.augsynth.post_periods)
        design = X0.subtract(X0.mean(axis=1), axis=0).to_numpy()
        y_c = Y0.subtract(Y0.mean(axis=1), axis=0).T.to_numpy()
        N = np.linalg.inv(design @ design.T + self.lambda_ * np.eye(design.shape[0]))
        beta = N @ (design @ y_c)
        F_all = pd.concat(
            [
                X0.subtract(X0.mean(axis=1), axis=0),
                X1.subtract(X0.mean(axis=1), axis=0).rename(X1.name),
            ],
            axis=1,
        )
        expected_mhat = pd.DataFrame(
            F_all.T.to_numpy() @ beta,
            index=F_all.columns,
            columns=self.augsynth.post_periods,
        )
        pd.testing.assert_frame_equal(self.augsynth.ridge_mhat, expected_mhat)

    def test_periods(self):
        self.augsynth.fit(dataprep=self.dataprep, lambda_=self.lambda_)
        self.assertEqual(self.augsynth.t_int, 7)
        self.assertEqual(self.augsynth.pre_periods, [1, 2, 3, 4, 5, 6])
        self.assertEqual(self.augsynth.post_periods, [7, 8])

    def test_time_optimize_ssr_must_be_pre_periods(self):
        bad_dataprep = make_dataprep(time_optimize_ssr=[1, 3, 4, 5, 6])
        augsynth = AugSynth()
        self.assertRaises(ValueError, augsynth.fit, bad_dataprep, lambda_=self.lambda_)

    def test_no_post_periods_raises(self):
        bad_dataprep = make_dataprep(time_optimize_ssr=[1, 2, 3, 4, 5, 6, 7, 8])
        augsynth = AugSynth()
        self.assertRaises(ValueError, augsynth.fit, bad_dataprep, lambda_=self.lambda_)

    def test_att_mspe_mape_mae(self):
        self.augsynth.fit(dataprep=self.dataprep, lambda_=self.lambda_)
        post = self.augsynth.post_periods
        att = self.augsynth.att(time_period=post)
        Z0, Z1 = self.dataprep.make_outcome_mats(time_period=post)
        gaps = Z1 - (Z0 * self.augsynth.W).sum(axis=1)
        self.assertAlmostEqual(att["att"], gaps.mean(), places=12)
        # pre-period fit measures
        self.assertGreater(self.augsynth.mspe(), 0.0)
        self.assertGreater(self.augsynth.mape(), 0.0)
        self.assertGreater(self.augsynth.mae(), 0.0)

    def test_placebo(self):
        self.augsynth.fit(dataprep=self.dataprep, lambda_=self.lambda_)
        placebo = PlaceboTest()
        placebo.fit(
            dataprep=self.dataprep,
            scm=AugSynth(),
            scm_options={"lambda_": self.lambda_},
        )
        self.assertEqual(placebo.gaps.shape, (8, 3))
