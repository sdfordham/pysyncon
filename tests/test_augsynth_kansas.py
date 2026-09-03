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
        assert df.shape == (5250, 6)
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
