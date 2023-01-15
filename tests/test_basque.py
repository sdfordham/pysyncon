import unittest
import pandas as pd

from pysyncon import Dataprep, Synth


class TestBasque(unittest.TestCase):
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

    def test_weights(self):
        synth = Synth()
        synth.fit(
            dataprep=self.dataprep,
            optim_method=self.optim_method,
            optim_initial=self.optim_initial,
        )
        weights = pd.Series(self.weights, name="weights")
        pd.testing.assert_series_equal(
            weights, synth.weights(round=9), check_less_precise=9
        )
