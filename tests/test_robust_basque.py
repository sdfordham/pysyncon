import unittest
import pandas as pd

from pysyncon import Dataprep, RobustSynth


class TestRobustBasque(unittest.TestCase):
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
        self.lambda_ = 0.1
        self.sv_count = 2
        self.weights = {
            "Aragon": 0.042750725,
            "Baleares (Islas)": 0.095687916,
            "Andalucia": 0.05471977,
            "Canarias": 0.029348893,
            "Cantabria": 0.131449835,
            "Castilla Y Leon": 0.00534905,
            "Castilla-La Mancha": -0.023989253,
            "Cataluna": 0.172766943,
            "Comunidad Valenciana": 0.098502043,
            "Extremadura": -0.024916194,
            "Galicia": 0.000285705,
            "Madrid (Comunidad De)": 0.306908016,
            "Murcia (Region de)": 0.037554988,
            "Navarra (Comunidad Foral De)": 0.042127484,
            "Principado De Asturias": 0.144568216,
            "Rioja (La)": 0.018474723,
        }

    def test_weights(self):
        robust = RobustSynth()
        robust.fit(dataprep=self.dataprep, lambda_=self.lambda_, sv_count=self.sv_count)

        weights = pd.Series(self.weights, name="weights")
        # Allow a tolerance of 2.5%
        pd.testing.assert_series_equal(
            weights, robust.weights(round=9), check_exact=False, atol=0.025
        )
