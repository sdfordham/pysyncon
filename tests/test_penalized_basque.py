import unittest
import pandas as pd

from pysyncon import Dataprep, PenalizedSynth


class TestPenalizedBasque(unittest.TestCase):
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
                "Aragon",
                "Baleares (Islas)",
                "Andalucia",
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
                "Principado De Asturias",
                "Rioja (La)",
            ],
            time_optimize_ssr=range(1960, 1970),
        )
        self.lambda_ = 0.01
        self.weights = {
            "Aragon": 0.0,
            "Baleares (Islas)": 0.21680004,
            "Andalucia": 0.0,
            "Canarias": 0.0,
            "Cantabria": 0.0,
            "Castilla Y Leon": 0.0,
            "Castilla-La Mancha": 0.0,
            "Cataluna": 0.636479454,
            "Comunidad Valenciana": 0.0,
            "Extremadura": 0.0,
            "Galicia": 0.0,
            "Madrid (Comunidad De)": 0.146720506,
            "Murcia (Region de)": 0.0,
            "Navarra (Comunidad Foral De)": 0.0,
            "Principado De Asturias": 0.0,
            "Rioja (La)": 0.0,
        }

    def test_weights(self):
        robust = PenalizedSynth()
        robust.fit(dataprep=self.dataprep, lambda_=self.lambda_)

        weights = pd.Series(self.weights, name="weights")
        # Allow a tolerance of 2.5%
        pd.testing.assert_series_equal(
            weights, robust.weights(round=9), check_exact=False, atol=0.025
        )
