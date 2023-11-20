import unittest
import pandas as pd

from pysyncon import Dataprep, Synth


class TestSynthTexas(unittest.TestCase):
    def setUp(self):
        df = pd.read_csv("./data/texas.csv")
        self.dataprep = Dataprep(
            foo=df,
            predictors=["income", "ur", "poverty"],
            predictors_op="mean",
            time_predictors_prior=range(1985, 1994),
            special_predictors=[
                ("bmprison", [1988], "mean"),
                ("bmprison", [1990], "mean"),
                ("bmprison", [1991], "mean"),
                ("bmprison", [1992], "mean"),
                ("alcohol", [1990], "mean"),
                ("aidscapita", [1990], "mean"),
                ("aidscapita", [1991], "mean"),
                ("black", [1990], "mean"),
                ("black", [1991], "mean"),
                ("black", [1992], "mean"),
                ("perc1519", [1990], "mean"),
            ],
            dependent="bmprison",
            unit_variable="state",
            time_variable="year",
            treatment_identifier="Texas",
            controls_identifier=[
                "Alabama",
                "Alaska",
                "Arizona",
                "Arkansas",
                "California",
                "Colorado",
                "Connecticut",
                "Delaware",
                "District of Columbia",
                "Florida",
                "Georgia",
                "Hawaii",
                "Idaho",
                "Illinois",
                "Indiana",
                "Iowa",
                "Kansas",
                "Kentucky",
                "Louisiana",
                "Maine",
                "Maryland",
                "Massachusetts",
                "Michigan",
                "Minnesota",
                "Mississippi",
                "Missouri",
                "Montana",
                "Nebraska",
                "Nevada",
                "New Hampshire",
                "New Jersey",
                "New Mexico",
                "New York",
                "North Carolina",
                "North Dakota",
                "Ohio",
                "Oklahoma",
                "Oregon",
                "Pennsylvania",
                "Rhode Island",
                "South Carolina",
                "South Dakota",
                "Tennessee",
                "Utah",
                "Vermont",
                "Virginia",
                "Washington",
                "West Virginia",
                "Wisconsin",
                "Wyoming",
            ],
            time_optimize_ssr=range(1985, 1994),
        )
        self.optim_method = "BFGS"
        self.optim_initial = "ols"
        self.weights = {
            "Alabama": 0.0,
            "Alaska": 0.0,
            "Arizona": 0.0,
            "Arkansas": 0.0,
            "California": 0.407651414,
            "Colorado": 0.0,
            "Connecticut": 0.0,
            "Delaware": 0.0,
            "District of Columbia": 0.0,
            "Florida": 0.110543548,
            "Georgia": 0.0,
            "Hawaii": 0.0,
            "Idaho": 0.0,
            "Illinois": 0.36027434,
            "Indiana": 0.0,
            "Iowa": 0.0,
            "Kansas": 0.0,
            "Kentucky": 0.0,
            "Louisiana": 0.121530698,
            "Maine": 0.0,
            "Maryland": 0.0,
            "Massachusetts": 0.0,
            "Michigan": 0.0,
            "Minnesota": 0.0,
            "Mississippi": 0.0,
            "Missouri": 0.0,
            "Montana": 0.0,
            "Nebraska": 0.0,
            "Nevada": 0.0,
            "New Hampshire": 0.0,
            "New Jersey": 0.0,
            "New Mexico": 0.0,
            "New York": 0.0,
            "North Carolina": 0.0,
            "North Dakota": 0.0,
            "Ohio": 0.0,
            "Oklahoma": 0.0,
            "Oregon": 0.0,
            "Pennsylvania": 0.0,
            "Rhode Island": 0.0,
            "South Carolina": 0.0,
            "South Dakota": 0.0,
            "Tennessee": 0.0,
            "Utah": 0.0,
            "Vermont": 0.0,
            "Virginia": 0.0,
            "Washington": 0.0,
            "West Virginia": 0.0,
            "Wisconsin": 0.0,
            "Wyoming": 0.0,
        }
        self.att = {"att": 20339.375838131393, "se": 3190.4946788704715}
        self.att_time_period = range(1993, 2001)

    def test_weights(self):
        synth = Synth()
        synth.fit(
            dataprep=self.dataprep,
            optim_method=self.optim_method,
            optim_initial=self.optim_initial,
        )
        weights = pd.Series(self.weights, name="weights")
        # Allow a tolerance of 2.5%
        pd.testing.assert_series_equal(
            weights, synth.weights(round=9), check_exact=False, atol=0.025
        )

    def test_att(self):
        synth = Synth()
        synth.fit(
            dataprep=self.dataprep,
            optim_method=self.optim_method,
            optim_initial=self.optim_initial,
        )
        synth_att = synth.att(time_period=self.att_time_period)

        # Allow a tolerance of 2.5%
        att_perc_delta = abs(1.0 - self.att["att"] / synth_att["att"])
        self.assertLessEqual(att_perc_delta, 0.025)

        # Allow a tolerance of 2.5%
        se_perc_delta = abs(1.0 - self.att["se"] / synth_att["se"])
        self.assertLessEqual(se_perc_delta, 0.025)
