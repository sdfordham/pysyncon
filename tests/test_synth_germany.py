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
        self.cis = {
            "value": {
                1991: 279.09685975333196,
                1992: 99.76203427529981,
                1993: -631.5437231770848,
                1994: -1050.2679900905205,
                1995: -1205.2549226793199,
                1996: -1467.2491625958974,
                1997: -1954.3741689815615,
                1998: -2008.3960300490326,
                1999: -2160.627036515649,
                2000: -2620.7330909274606,
            },
            "lower_ci": {
                1991: 43.148688105431994,
                1992: -136.18613737260014,
                1993: -867.4918948249846,
                1994: -1286.2161617384206,
                1995: -1441.20309432722,
                1996: -1703.1973342437975,
                1997: -2190.3223406294615,
                1998: -2244.3442016969325,
                1999: -2396.5752081635487,
                2000: -2856.6812625753605,
            },
            "upper_ci": {
                1991: 515.0450314012319,
                1992: 335.7102059231998,
                1993: -395.59555152918483,
                1994: -814.3198184426207,
                1995: -969.3067510314198,
                1996: -1231.3009909479972,
                1997: -1718.4259973336614,
                1998: -1772.4478584011324,
                1999: -1924.6788648677486,
                2000: -2384.7849192795607,
            },
        }
        self.ci_args = {
            "alpha": 0.05,
            "time_periods": [
                1991,
                1992,
                1993,
                1994,
                1995,
                1996,
                1997,
                1998,
                1999,
                2000,
            ],
            "max_iter": 50,
            "tol": 0.1,
            "verbose": False,
        }

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

    def test_cis(self):
        synth = Synth()
        synth.fit(
            dataprep=self.dataprep,
            optim_method=self.optim_method,
            optim_initial=self.optim_initial,
            custom_V=self.custom_V,
        )

        cis = pd.DataFrame.from_dict(self.cis)
        cis.index.name = "time"
        pd.testing.assert_frame_equal(
            cis,
            synth.confidence_interval(custom_V=self.custom_V, **self.ci_args),
            check_exact=False,
            atol=0.025,
        )
