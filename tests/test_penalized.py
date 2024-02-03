import unittest
import numpy as np
import pandas as pd

import pysyncon


class TestPenalizedSynth(unittest.TestCase):
    def setUp(self):
        self.dataprep = pysyncon.Dataprep(
            foo=pd.DataFrame(
                {
                    "time": [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
                    "name": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
                    "dependent": np.random.random(12),
                    "predictor1": np.random.random(12),
                    "predictor2": np.random.random(12),
                }
            ),
            predictors=["predictor1"],
            predictors_op="mean",
            dependent="dependent",
            unit_variable="name",
            time_variable="time",
            treatment_identifier=1,
            controls_identifier=[2, 3],
            time_predictors_prior=[2, 3],
            time_optimize_ssr=[1, 2, 3],
            special_predictors=[
                ("predictor1", [2], "mean"),
                ("predictor2", [1, 2], "median"),
                ("predictor2", [1, 2], "std"),
            ],
        )
        self.custom_V = np.full(4, 1.0)

    def test_fit_no_data(self):
        pen = pysyncon.PenalizedSynth()
        self.assertRaises(ValueError, pen.fit)

    def test_fit_custom_V(self):
        pen = pysyncon.PenalizedSynth()
        pen.fit(dataprep=self.dataprep, custom_V=self.custom_V)
