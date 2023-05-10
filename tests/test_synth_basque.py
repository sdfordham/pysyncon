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
                1960.0: -0.203808056,
                1961.0: -0.220131278,
                1962.0: -0.263867427,
                1963.0: -0.305086234,
                1964.0: -0.307812904,
                1965.0: -0.310500965,
                1966.0: -0.369694018,
                1967.0: -0.423575377,
                1968.0: -0.45873673,
                1969.0: -0.488697382,
                1970.0: -0.492355237,
            },
            "Madrid (Comunidad De)": {
                1960.0: -0.927175216,
                1961.0: -1.066517558,
                1962.0: -1.011035191,
                1963.0: -0.950460346,
                1964.0: -0.945849043,
                1965.0: -0.930054154,
                1966.0: -0.772216555,
                1967.0: -0.614639881,
                1968.0: -0.557819528,
                1969.0: -0.491421484,
                1970.0: -0.441239691,
            },
            "Andalucia": {
                1960.0: 0.005071129,
                1961.0: -0.002029762,
                1962.0: 0.002976458,
                1963.0: 0.008368423,
                1964.0: 0.012947728,
                1965.0: 0.0182735,
                1966.0: 0.002324614,
                1967.0: -0.012943574,
                1968.0: -0.009046578,
                1969.0: -0.00457983,
                1970.0: -0.013673715,
            },
        }
        self.summary = pd.DataFrame(
            data=[
                [7.300000e-02, 3.988800e+01, 2.563360e+02, 3.238260e+02],
                [1.200000e-01, 1.031742e+03, 2.730103e+03, 2.182453e+03],
                [3.000000e-03, 9.035900e+01, 2.233400e+02, 1.488640e+02],
                [1.020000e-01, 2.572800e+01, 6.343700e+01, 4.713300e+01],
                [1.100000e-02, 1.348000e+01, 3.615400e+01, 2.616300e+01],
                [0.000000e+00, 2.464700e+01, 2.158300e+01, 2.144500e+01],
                [1.170000e-01, 5.285000e+00, 5.271000e+00, 3.584000e+00],
                [6.300000e-02, 6.844000e+00, 6.179000e+00, 2.105800e+01],
                [1.550000e-01, 4.106000e+00, 2.760000e+00, 5.252000e+00],
                [9.600000e-02, 4.508200e+01, 3.763600e+01, 2.267000e+01],
                [5.300000e-02, 6.150000e+00, 6.952000e+00, 7.274000e+00],
                [2.000000e-03, 3.375400e+01, 4.110400e+01, 3.664600e+01],
                [2.400000e-02, 4.072000e+00, 5.371000e+00, 7.103000e+00],
                [1.810000e-01, 2.468900e+02, 1.962840e+02, 9.746800e+01]
            ],
            columns=['V', 'treated', 'synthetic', 'sample mean'],
            index=['school.illit', 'school.prim', 'school.med', 'school.high',
            'school.post.high', 'invest', 'special.1.gdpcap',
            'special.2.sec.agriculture', 'special.3.sec.energy',
            'special.4.sec.industry', 'special.5.sec.construction',
            'special.6.sec.services.venta', 'special.7.sec.services.nonventa',
            'special.8.popdens']
        )

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
            self.summary, synth.summary(), check_exact=False
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
