import unittest
import pandas as pd

from pysyncon import Dataprep, Synth


class TestSynthGermany(unittest.TestCase):
    def setUp(self):
        df = pd.read_csv("./data/germany.csv")
        dataprep_train = Dataprep(
            foo=df,
            predictors=["gdp", "trade", "infrate"],
            predictors_op="mean",
            time_predictors_prior=range(1971, 1981),
            special_predictors=[
                ("industry", range(1971, 1981), "mean"),
                ("schooling", [1970, 1975], "mean"),
                ("invest70", [1980], "mean"),
            ],
            dependent="gdp",
            unit_variable="country",
            time_variable="year",
            treatment_identifier="West Germany",
            controls_identifier=[
                "USA",
                "UK",
                "Austria",
                "Belgium",
                "Denmark",
                "France",
                "Italy",
                "Netherlands",
                "Norway",
                "Switzerland",
                "Japan",
                "Greece",
                "Portugal",
                "Spain",
                "Australia",
                "New Zealand",
            ],
            time_optimize_ssr=range(1981, 1991),
        )
        synth_train = Synth()
        synth_train.fit(
            dataprep=dataprep_train, optim_method="Nelder-Mead", optim_initial="equal"
        )
        self.custom_V = synth_train.V

        self.dataprep = Dataprep(
            foo=df,
            predictors=["gdp", "trade", "infrate"],
            predictors_op="mean",
            time_predictors_prior=range(1981, 1991),
            special_predictors=[
                ("industry", range(1981, 1991), "mean"),
                ("schooling", [1980, 1985], "mean"),
                ("invest80", [1980], "mean"),
            ],
            dependent="gdp",
            unit_variable="country",
            time_variable="year",
            treatment_identifier="West Germany",
            controls_identifier=[
                "USA",
                "UK",
                "Austria",
                "Belgium",
                "Denmark",
                "France",
                "Italy",
                "Netherlands",
                "Norway",
                "Switzerland",
                "Japan",
                "Greece",
                "Portugal",
                "Spain",
                "Australia",
                "New Zealand",
            ],
            time_optimize_ssr=range(1960, 1990),
        )

        self.optim_method = "Nelder-Mead"
        self.optim_initial = "equal"
        self.weights = {
            "USA": 0.21624982,
            "UK": 0.0,
            "Austria": 0.414522077,
            "Belgium": 0.0,
            "Denmark": 0.0,
            "France": 0.0,
            "Italy": 0.0,
            "Netherlands": 0.09841208,
            "Norway": 0.0,
            "Switzerland": 0.107654851,
            "Japan": 0.163161172,
            "Greece": 0.0,
            "Portugal": 0.0,
            "Spain": 0.0,
            "Australia": 0.0,
            "New Zealand": 0.0,
        }
        self.att = {"att": -1555.1346777620479, "se": 317.6469306023242}
        self.att_time_period = range(1990, 2004)

    def test_weights(self):
        synth = Synth()
        synth.fit(
            dataprep=self.dataprep,
            optim_method=self.optim_method,
            optim_initial=self.optim_initial,
            custom_V=self.custom_V,
        )
        weights = pd.Series(self.weights, name="weights")
        pd.testing.assert_series_equal(
            weights, synth.weights(round=9), check_exact=False, atol=0.025
        )

    def test_att(self):
        synth = Synth()
        synth.fit(
            dataprep=self.dataprep,
            optim_method=self.optim_method,
            optim_initial=self.optim_initial,
            custom_V=self.custom_V,
        )
        synth_att = synth.att(time_period=self.att_time_period)

        # Allow a tolerance of 2.5%
        att_perc_delta = abs(1.0 - self.att["att"] / synth_att["att"])
        self.assertLessEqual(att_perc_delta, 0.025)

        # Allow a tolerance of 2.5%
        se_perc_delta = abs(1.0 - self.att["se"] / synth_att["se"])
        self.assertLessEqual(se_perc_delta, 0.025)
