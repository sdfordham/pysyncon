import unittest
import pandas as pd

from pysyncon import Dataprep, AugSynth


class TestAugsynthBasque(unittest.TestCase):
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
                "Andalucia",
                "Aragon",
                "Baleares (Islas)",
                "Canarias",
                "Cantabria",
                "Castilla-La Mancha",
                "Castilla Y Leon",
                "Cataluna",
                "Comunidad Valenciana",
                "Extremadura",
                "Galicia",
                "Madrid (Comunidad De)",
                "Murcia (Region de)",
                "Navarra (Comunidad Foral De)",
                "Principado De Asturias",
                "Rioja (La)",
                "Spain (Espana)",
            ],
            time_optimize_ssr=range(1960, 1970),
        )
        self.optim_method = "Nelder-Mead"
        self.optim_initial = "equal"
        self.weights = {
            "Andalucia": 0.113627911,
            "Aragon": 1.774922286,
            "Baleares (Islas)": -0.713432799,
            "Canarias": 1.19397534,
            "Cantabria": 0.497825351,
            "Castilla-La Mancha": 0.131573892,
            "Castilla Y Leon": -1.405974956,
            "Cataluna": 1.31890027,
            "Comunidad Valenciana": -1.731140541,
            "Extremadura": -1.134362989,
            "Galicia": 1.982136937,
            "Madrid (Comunidad De)": 0.110801212,
            "Murcia (Region de)": -1.31476635,
            "Navarra (Comunidad Foral De)": -1.303045915,
            "Principado De Asturias": -0.02423815,
            "Rioja (La)": 1.58950474,
            "Spain (Espana)": -0.086306241,
        }

    def test_weights(self):
        augsynth = AugSynth()
        augsynth.fit(dataprep=self.dataprep)

        weights = pd.Series(self.weights, name="weights")
        # Allow a tolerance of 2.5%
        pd.testing.assert_series_equal(
            weights, augsynth.weights(round=9), check_exact=False, atol=0.025
        )
