import unittest
import pandas as pd

from pysyncon import Dataprep, Synth
from pysyncon.utils import PlaceboTest


class TestSynthBasque(unittest.TestCase):
    def setUp(self):
        df = pd.read_csv("./data/basque.csv")
        self.dataprep = Dataprep(
            foo=df,
            predictors=[
                "school.illit",
                "school.prim",
                "school.med",
                "school.high",
                "school.post.high",
                "invest",
            ],
            predictors_op="mean",
            time_predictors_prior=range(1964, 1970),
            special_predictors=[
                ("gdpcap", range(1960, 1970), "mean"),
                ("sec.agriculture", range(1961, 1970, 2), "mean"),
                ("sec.energy", range(1961, 1970, 2), "mean"),
                ("sec.industry", range(1961, 1970, 2), "mean"),
                ("sec.construction", range(1961, 1970, 2), "mean"),
                ("sec.services.venta", range(1961, 1970, 2), "mean"),
                ("sec.services.nonventa", range(1961, 1970, 2), "mean"),
                ("popdens", [1969], "mean"),
            ],
            dependent="gdpcap",
            unit_variable="regionname",
            time_variable="year",
            treatment_identifier="Basque Country (Pais Vasco)",
            controls_identifier=[
                "Spain (Espana)",
                "Andalucia",
                "Aragon",
                "Principado De Asturias",
                "Baleares (Islas)",
                "Canarias",
                "Cantabria",
                "Castilla Y Leon",
                "Castilla-La Mancha",
                "Cataluna",
                "Comunidad Valenciana",
                "Extremadura",
                "Galicia",
                "Madrid (Comunidad De)",
                "Murcia (Region de)",
                "Navarra (Comunidad Foral De)",
                "Rioja (La)",
            ],
            time_optimize_ssr=range(1960, 1970),
        )
        self.optim_method = "Nelder-Mead"
        self.optim_initial = "equal"
        self.weights = {
            "Spain (Espana)": 0.0,
            "Andalucia": 0.0,
            "Aragon": 0.0,
            "Principado De Asturias": 0.0,
            "Baleares (Islas)": 0.0,
            "Canarias": 0.0,
            "Cantabria": 0.0,
            "Castilla Y Leon": 0.0,
            "Castilla-La Mancha": 0.0,
            "Cataluna": 0.850816306,
            "Comunidad Valenciana": 0.0,
            "Extremadura": 0.0,
            "Galicia": 0.0,
            "Madrid (Comunidad De)": 0.149183694,
            "Murcia (Region de)": 0.0,
            "Navarra (Comunidad Foral De)": 0.0,
            "Rioja (La)": 0.0,
        }
        self.placebo_gaps = {
            "Cataluna": {
                1960.0: 0.203808058,
                1961.0: 0.22013128,
                1962.0: 0.263867425,
                1963.0: 0.305086227,
                1964.0: 0.307812892,
                1965.0: 0.310500949,
                1966.0: 0.369694004,
                1967.0: 0.423575362,
                1968.0: 0.458736716,
                1969.0: 0.488697369,
                1970.0: 0.492355223,
            },
            "Madrid (Comunidad De)": {
                1960.0: 0.927170193,
                1961.0: 1.066511653,
                1962.0: 1.011029922,
                1963.0: 0.950455684,
                1964.0: 0.945846094,
                1965.0: 0.930053083,
                1966.0: 0.772220243,
                1967.0: 0.614648344,
                1968.0: 0.557832902,
                1969.0: 0.491439776,
                1970.0: 0.441262212,
            },
            "Andalucia": {
                1960.0: -0.005071144,
                1961.0: 0.002029757,
                1962.0: -0.002976465,
                1963.0: -0.008368432,
                1964.0: -0.012947738,
                1965.0: -0.018273511,
                1966.0: -0.002324632,
                1967.0: 0.012943551,
                1968.0: 0.009046557,
                1969.0: 0.004579814,
                1970.0: 0.013673678,
            },
        }
        self.summary = pd.DataFrame(
            data=[
                [7.26559110e-02, 3.98884646e01, 2.56336977e02, 3.23825543e02],
                [1.19777358e-01, 1.03174230e03, 2.73010720e03, 2.18245335e03],
                [3.48611100e-03, 9.03586680e01, 2.23340172e02, 1.48864075e02],
                [1.02189247e-01, 2.57275251e01, 6.34368045e01, 4.71326627e01],
                [1.08267860e-02, 1.34797198e01, 3.61534897e01, 2.61630325e01],
                [5.32110000e-05, 2.46473831e01, 2.15826359e01, 2.14454579e01],
                [1.17260969e-01, 5.28546845e00, 5.27078346e00, 3.58401509e00],
                [6.33926060e-02, 6.84399996e00, 6.17934020e00, 2.10581177e01],
                [1.55350772e-01, 4.10600004e00, 2.75975796e00, 5.25223529e00],
                [9.58688000e-02, 4.50820000e01, 3.76359420e01, 2.26702353e01],
                [5.30811070e-02, 6.15000000e00, 6.95245150e00, 7.27400001e00],
                [1.63475200e-03, 3.37540001e01, 4.11037607e01, 3.66458824e01],
                [2.37097130e-02, 4.07200012e00, 5.37134427e00, 7.10294116e00],
                [1.80712657e-01, 2.46889999e02, 1.96283316e02, 9.74682350e01],
            ],
            columns=["V", "treated", "synthetic", "sample mean"],
            index=[
                "school.illit",
                "school.prim",
                "school.med",
                "school.high",
                "school.post.high",
                "invest",
                "special.1.gdpcap",
                "special.2.sec.agriculture",
                "special.3.sec.energy",
                "special.4.sec.industry",
                "special.5.sec.construction",
                "special.6.sec.services.venta",
                "special.7.sec.services.nonventa",
                "special.8.popdens",
            ],
        )
        self.treatment_time = 1975
        self.pvalue = 0.16666666666666666
        self.att = {"att": -0.6995647842110987, "se": 0.07078092130438395}
        self.att_time_period = range(1975, 1998)

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
        pd.testing.assert_frame_equal(
            self.summary, synth.summary(round=9), check_exact=False, atol=0.025
        )

    def test_placebo_weights(self):
        synth = Synth()
        placebo_test = PlaceboTest()
        placebo_test.fit(
            dataprep=self.dataprep,
            scm=synth,
            scm_options={
                "optim_method": self.optim_method,
                "optim_initial": self.optim_initial,
            },
        )

        placebo_gaps = pd.DataFrame.from_dict(self.placebo_gaps).rename_axis(
            index="year"
        )
        regions = self.placebo_gaps.keys()
        years = list(self.placebo_gaps["Cataluna"].keys())
        pd.testing.assert_frame_equal(
            placebo_gaps,
            placebo_test.gaps[regions].loc[years],
            check_exact=False,
            atol=0.025,
        )
        self.assertAlmostEqual(
            self.pvalue,
            placebo_test.pvalue(treatment_time=self.treatment_time),
            places=3,
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
