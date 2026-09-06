import unittest
import numpy as np
import pandas as pd

from pysyncon import Dataprep, AugSynth
from pysyncon.augsynth import VanillaOptimMixin


class TestAugSynthKansas(unittest.TestCase):
    """Validation against the published outputs of the R augsynth package for
    the Kansas example from Ben-Michael, Feller & Rothstein (2021), see
    `vignettes/singlesynth-vignette.md` in github.com/ebenmichael/augsynth
    (`augsynth(lngdpcapita ~ treated, fips, year_qtr, 2012.25, kansas,
    progfunc="Ridge", scm=T)`, no covariates, lambda chosen by
    cross-validation).

    The data is `data/kansas.csv`, converted from the R augsynth package
    `data/kansas.rda`.

    Vignette reference values (Ridge ASCM):
        ATT -0.0401, L2 imbalance 0.062, 49 donor units,
        average estimated bias 0.011.
    Vignette reference values (SCM):
        ATT -0.0294, L2 imbalance 0.083, 7 donor units.
    """

    @classmethod
    def setUpClass(cls):
        df = pd.read_csv("./data/kansas.csv")
        # data sanity checks
        assert df.shape == (5250, 11)
        fips = sorted(df.fips.unique())
        assert len(fips) == 50
        times = sorted(df.year_qtr.unique())
        assert times[0] == 1990.0 and times[-1] == 2016.0 and len(times) == 105
        assert np.allclose(np.diff(times), 0.25)
        expected_treated = ((df.fips == 20) & (df.year_qtr >= 2012.25)).astype(float)
        assert (df.treated == expected_treated).all()
        assert df.lngdpcapita.notna().all()

        pre = sorted(df.loc[df.year_qtr < 2012.25, "year_qtr"].unique())
        cls.dataprep = Dataprep(
            foo=df,
            predictors=[],  # no covariates, as in the R vignette example
            predictors_op="mean",
            dependent="lngdpcapita",
            unit_variable="fips",
            time_variable="year_qtr",
            treatment_identifier=20,
            controls_identifier=[f for f in fips if f != 20],
            time_predictors_prior=pre,  # unused; no covariates in this fit
            time_optimize_ssr=pre,
        )
        cls.augsynth = AugSynth()
        cls.augsynth.fit(cls.dataprep, use_covariates=False)

    def test_periods(self):
        self.assertEqual(self.augsynth.t_int, 2012.25)
        self.assertEqual(len(self.augsynth.pre_periods), 89)
        self.assertEqual(len(self.augsynth.post_periods), 16)

    def test_att(self):
        Z0, Z1 = self.dataprep.make_outcome_mats(time_period=self.augsynth.post_periods)
        att = (Z1 - (Z0 * self.augsynth.W).sum(axis=1)).mean()
        self.assertAlmostEqual(att, -0.0401, delta=1e-3)

    def test_l2_imbalance(self):
        X0, X1 = self.dataprep.make_outcome_mats(time_period=self.augsynth.pre_periods)
        l2 = np.sqrt(((X0.to_numpy() @ self.augsynth.W - X1.to_numpy()) ** 2).sum())
        self.assertAlmostEqual(l2, 0.062, delta=1e-3)

    def test_diagnostics(self):
        # l2_imbalance matches the published R vignette value (0.062) and
        # 1 - scaled_l2_imbalance is the published % improvement (84.7)
        self.assertAlmostEqual(self.augsynth.l2_imbalance, 0.061515, delta=1e-3)
        self.assertAlmostEqual(self.augsynth.unif_l2_imbalance, 0.402083, delta=1e-3)
        self.assertAlmostEqual(self.augsynth.scaled_l2_imbalance, 0.152992, delta=1e-3)
        self.assertAlmostEqual(1 - self.augsynth.scaled_l2_imbalance, 0.847, delta=1e-3)
        self.assertIsNone(self.augsynth.covariate_l2_imbalance)
        self.assertIsNone(self.augsynth.scaled_covariate_l2_imbalance)

    def test_n_donors(self):
        n_units = len(self.dataprep.controls_identifier)
        n_donors = (np.abs(self.augsynth.W) > 1 / (1000 * n_units)).sum()
        self.assertEqual(n_donors, 49)

    def test_lambda(self):
        lambdas = self.augsynth.cv_result.lambdas
        self.assertEqual(len(lambdas), 21)
        self.assertTrue(np.isclose(self.augsynth.lambda_, lambdas).any())

    def test_avg_bias(self):
        self.assertAlmostEqual(self.augsynth.avg_bias, 0.011, delta=1e-3)

    def test_ridge_mhat(self):
        # evaluated for all units (controls + treated) over the post periods
        self.assertEqual(self.augsynth.ridge_mhat.shape, (50, 16))
        self.assertEqual(len(self.augsynth.bias_est), 16)

    def test_scm_cross_check(self):
        # plain SCM (V = I) on the centered pre-treatment outcomes
        X0, X1 = self.dataprep.make_outcome_mats(time_period=self.augsynth.pre_periods)
        X0_c = X0.subtract(X0.mean(axis=1), axis=0)
        X1_c = X1.subtract(X0.mean(axis=1), axis=0)
        W, _ = VanillaOptimMixin.w_optimize(
            V_mat=np.eye(X0.shape[0]),
            X0=X0_c.to_numpy(),
            X1=X1_c.to_numpy(),
            qp_options={"maxiter": 2000, "ftol": 1e-12},
        )

        Z0, Z1 = self.dataprep.make_outcome_mats(time_period=self.augsynth.post_periods)
        att = (Z1 - (Z0 * W).sum(axis=1)).mean()
        self.assertAlmostEqual(att, -0.0294, delta=1e-3)

        l2 = np.sqrt(((X0.to_numpy() @ W - X1.to_numpy()) ** 2).sum())
        self.assertAlmostEqual(l2, 0.083, delta=1e-3)

        n_units = len(self.dataprep.controls_identifier)
        n_donors = (np.abs(W) > 1 / (1000 * n_units)).sum()
        self.assertEqual(n_donors, 7)


class TestAugSynthKansasCovariates(unittest.TestCase):
    """Covariate-augmented fit of the Kansas example, following the
    "Augmenting with covariates" section of the R augsynth package vignette
    `singlesynth-vignette`:

        covsyn <- augsynth(lngdpcapita ~ treated | lngdpcapita +
            log(revstatecapita) + log(revlocalcapita) +
            log(avgwklywagecapita) + estabscapita + emplvlcapita,
            fips, year_qtr, kansas, progfunc = "ridge", scm = T)

    The covariates are averaged over the pre-intervention period (dropping
    missing values), as in the R package. The vignette prints no summary
    numbers for this fit, so these values are not independently R-verified;
    they are regression baselines of the corrected (ported) algorithm, which
    is validated against the published R outputs for the no-covariate fit
    (see TestAugSynthKansas).
    """

    @classmethod
    def setUpClass(cls):
        df = pd.read_csv("./data/kansas.csv")
        # covariate data sanity checks
        assert df.revstatecapita.isna().sum() == 2850
        assert df.revlocalcapita.isna().sum() == 2850
        assert df.avgwklywagecapita.isna().sum() == 0
        assert df.estabscapita.isna().sum() == 0
        assert df.emplvlcapita.isna().sum() == 0
        pre_mask = df.year_qtr < 2012.25
        for c in [
            "revstatecapita",
            "revlocalcapita",
            "avgwklywagecapita",
            "estabscapita",
            "emplvlcapita",
        ]:
            assert (df.loc[pre_mask].groupby("fips")[c].agg("count") > 0).all()

        df["log_revstatecapita"] = np.log(df["revstatecapita"])
        df["log_revlocalcapita"] = np.log(df["revlocalcapita"])
        df["log_avgwklywagecapita"] = np.log(df["avgwklywagecapita"])

        fips = sorted(df.fips.unique())
        pre = sorted(df.loc[df.year_qtr < 2012.25, "year_qtr"].unique())
        cls.dataprep = Dataprep(
            foo=df,
            predictors=[
                "lngdpcapita",
                "log_revstatecapita",
                "log_revlocalcapita",
                "log_avgwklywagecapita",
                "estabscapita",
                "emplvlcapita",
            ],
            predictors_op="mean",
            time_predictors_prior=pre,
            dependent="lngdpcapita",
            unit_variable="fips",
            time_variable="year_qtr",
            treatment_identifier=20,
            controls_identifier=[f for f in fips if f != 20],
            time_optimize_ssr=pre,
        )
        cls.augsynth = AugSynth()
        cls.augsynth.fit(cls.dataprep)

    def test_periods(self):
        self.assertEqual(self.augsynth.t_int, 2012.25)
        self.assertEqual(len(self.augsynth.pre_periods), 89)
        self.assertEqual(len(self.augsynth.post_periods), 16)

    def test_att(self):
        Z0, Z1 = self.dataprep.make_outcome_mats(time_period=self.augsynth.post_periods)
        att = (Z1 - (Z0 * self.augsynth.W).sum(axis=1)).mean()
        self.assertAlmostEqual(att, -0.0641, delta=1e-3)

    def test_l2_imbalance(self):
        X0, X1 = self.dataprep.make_outcome_mats(time_period=self.augsynth.pre_periods)
        l2 = np.sqrt(((X0.to_numpy() @ self.augsynth.W - X1.to_numpy()) ** 2).sum())
        self.assertAlmostEqual(l2, 0.0456, delta=1e-3)

    def test_n_donors(self):
        n_units = len(self.dataprep.controls_identifier)
        n_donors = (np.abs(self.augsynth.W) > 1 / (1000 * n_units)).sum()
        self.assertEqual(n_donors, 49)

    def test_diagnostics(self):
        self.assertAlmostEqual(self.augsynth.l2_imbalance, 0.045649, delta=1e-3)
        self.assertAlmostEqual(self.augsynth.scaled_l2_imbalance, 0.113532, delta=1e-3)
        self.assertAlmostEqual(self.augsynth.covariate_l2_imbalance, 0.002594, delta=1e-3)
        self.assertAlmostEqual(
            self.augsynth.scaled_covariate_l2_imbalance, 0.012898, delta=1e-3
        )

    def test_lambda(self):
        lambdas = self.augsynth.cv_result.lambdas
        self.assertEqual(len(lambdas), 21)
        self.assertTrue(np.isclose(self.augsynth.lambda_, lambdas).any())
        self.assertAlmostEqual(self.augsynth.lambda_, 0.00511996, delta=1e-6)

    def test_avg_bias(self):
        self.assertAlmostEqual(self.augsynth.avg_bias, 0.0304, delta=1e-3)

    def test_design_rows(self):
        # 89 pre-treatment outcome rows + 6 covariate rows
        self.assertEqual(self.augsynth.beta.shape, (95, 16))

    def test_covariate_balance(self):
        # the ridge augmentation removes most of the covariate imbalance
        # left by the SCM weights (in scaled design units)
        Z0, Z1 = self.dataprep.make_covariate_mats()
        Z0_c = Z0.subtract(Z0.mean(axis=1), axis=0)
        Z1_c = Z1.subtract(Z0.mean(axis=1), axis=0)
        sdz = Z0_c.std(axis=1)
        X0, _ = self.dataprep.make_outcome_mats(time_period=self.augsynth.pre_periods)
        sdx = X0.subtract(X0.mean(axis=1), axis=0).to_numpy().std(ddof=1)

        scm_imb = (Z1_c.divide(sdz, axis=0) * sdx) - (
            (Z0_c * self.augsynth.synw).sum(axis=1).divide(sdz, axis=0) * sdx
        )
        aug_imb = (Z1_c.divide(sdz, axis=0) * sdx) - (
            (Z0_c * self.augsynth.W).sum(axis=1).divide(sdz, axis=0) * sdx
        )
        self.assertAlmostEqual(np.abs(scm_imb).sum(), 0.1011, delta=1e-2)
        self.assertAlmostEqual(np.abs(aug_imb).sum(), 0.0036, delta=1e-2)
        self.assertLess(np.abs(aug_imb).sum(), np.abs(scm_imb).sum())


class TestAugSynthKansasResidualize(unittest.TestCase):
    """Residualized covariate fit of the Kansas example (the `residualize =
    T` variant of the covariate fit in the R augsynth vignette), which is
    fitted with `lambda = asyn$lambda` (the cross-validated lambda of the
    no-covariate fit). The vignette prints no summary numbers for this fit
    either, so these values are regression baselines of the corrected
    (ported) algorithm, which is validated against the published R outputs
    for the no-covariate fit (see TestAugSynthKansas).
    """

    @classmethod
    def setUpClass(cls):
        df = pd.read_csv("./data/kansas.csv")
        df["log_revstatecapita"] = np.log(df["revstatecapita"])
        df["log_revlocalcapita"] = np.log(df["revlocalcapita"])
        df["log_avgwklywagecapita"] = np.log(df["avgwklywagecapita"])

        fips = sorted(df.fips.unique())
        pre = sorted(df.loc[df.year_qtr < 2012.25, "year_qtr"].unique())
        cls.dataprep = Dataprep(
            foo=df,
            predictors=[
                "lngdpcapita",
                "log_revstatecapita",
                "log_revlocalcapita",
                "log_avgwklywagecapita",
                "estabscapita",
                "emplvlcapita",
            ],
            predictors_op="mean",
            time_predictors_prior=pre,
            dependent="lngdpcapita",
            unit_variable="fips",
            time_variable="year_qtr",
            treatment_identifier=20,
            controls_identifier=[f for f in fips if f != 20],
            time_optimize_ssr=pre,
        )
        # the no-covariate fit's cross-validated lambda (1-SE rule), as in
        # the vignette's `lambda = asyn$lambda` (see kansas.ipynb)
        lambda_asyn = 0.07866222813499749
        cls.augsynth = AugSynth()
        cls.augsynth.fit(cls.dataprep, lambda_=lambda_asyn, residualize=True)

    def test_att(self):
        Z0, Z1 = self.dataprep.make_outcome_mats(time_period=self.augsynth.post_periods)
        att = (Z1 - (Z0 * self.augsynth.W).sum(axis=1)).mean()
        self.assertAlmostEqual(att, -0.0548, delta=1e-3)

    def test_l2_imbalance(self):
        X0, X1 = self.dataprep.make_outcome_mats(time_period=self.augsynth.pre_periods)
        l2 = np.sqrt(((X0.to_numpy() @ self.augsynth.W - X1.to_numpy()) ** 2).sum())
        self.assertAlmostEqual(l2, 0.0669, delta=1e-3)

    def test_n_donors(self):
        n_units = len(self.dataprep.controls_identifier)
        n_donors = (np.abs(self.augsynth.W) > 1 / (1000 * n_units)).sum()
        self.assertEqual(n_donors, 49)

    def test_lambda(self):
        self.assertAlmostEqual(self.augsynth.lambda_, 0.07866222813499749, delta=1e-9)

    def test_avg_bias(self):
        self.assertAlmostEqual(self.augsynth.avg_bias, 0.0064, delta=1e-3)

    def test_no_cov_weights(self):
        self.assertIsNotNone(self.augsynth.no_cov_weights)
        self.assertEqual(len(self.augsynth.no_cov_weights), len(self.augsynth.W))

    def test_design_rows(self):
        # the residualized design contains the pre-treatment outcomes only
        self.assertEqual(self.augsynth.beta.shape, (89, 16))

    def test_exact_covariate_balance(self):
        # the covariate re-add balances the covariates exactly (up to
        # numerical precision), the hallmark of residualize=TRUE
        Z0, Z1 = self.dataprep.make_covariate_mats()
        Z0_c = Z0.subtract(Z0.mean(axis=1), axis=0).to_numpy()
        Z1_c = Z1.subtract(Z0.mean(axis=1), axis=0).to_numpy()
        imb = Z1_c - Z0_c @ self.augsynth.W
        self.assertLess(np.sqrt((imb**2).sum()), 1e-8)

    def test_diagnostics(self):
        self.assertAlmostEqual(self.augsynth.l2_imbalance, 0.066929, delta=1e-3)
        self.assertAlmostEqual(self.augsynth.scaled_l2_imbalance, 0.166456, delta=1e-3)
        # exact covariate balance (centered, unscaled covariates)
        self.assertLess(self.augsynth.covariate_l2_imbalance, 1e-8)
