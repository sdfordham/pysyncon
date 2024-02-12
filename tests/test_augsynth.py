import unittest
import numpy as np
import pandas as pd

import pysyncon


class TestAugSynth(unittest.TestCase):
    def setUp(self):
        self.foo = pd.DataFrame(
            {
                "time": [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
                "name": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
                "dependent": np.random.random(12),
                "predictor1": np.random.random(12),
                "predictor2": np.random.random(12),
            }
        )
        self.predictors = ["predictor1"]
        self.predictors_op = "mean"
        self.dependent = "dependent"
        self.unit_variable = "name"
        self.time_variable = "time"
        self.treatment_identifier = 1
        self.treatment_identifier_list = [1, 2]
        self.controls_identifier = [2, 3]
        self.controls_identifier_alt = [3]
        self.time_predictors_prior = [2, 3]
        self.time_optimize_ssr = [1, 2, 3]
        self.special_predictors = [
            ("predictor1", [2], "mean"),
            ("predictor2", [1, 2], "median"),
            ("predictor2", [1, 2], "std"),
        ]

    def test_fit_treated(self):
        kwargs = {
            "foo": self.foo,
            "predictors": self.predictors,
            "predictors_op": self.predictors_op,
            "dependent": self.dependent,
            "unit_variable": self.unit_variable,
            "time_variable": self.time_variable,
            "time_predictors_prior": self.time_predictors_prior,
            "time_optimize_ssr": self.time_optimize_ssr,
            "special_predictors": self.special_predictors,
        }

        dataprep = pysyncon.Dataprep(
            treatment_identifier=self.treatment_identifier_list,
            controls_identifier=self.controls_identifier_alt,
            **kwargs,
        )
        augsynth = pysyncon.AugSynth()
        self.assertRaises(ValueError, augsynth.fit, dataprep)

        dataprep = pysyncon.Dataprep(
            treatment_identifier=self.treatment_identifier,
            controls_identifier=self.controls_identifier,
            **kwargs,
        )
        augsynth = pysyncon.AugSynth()
        try:
            augsynth.fit(dataprep)
        except Exception as e:
            self.fail(f"Augsynth fit with single treated failed: {e}.")

        dataprep = pysyncon.Dataprep(
            treatment_identifier=[self.treatment_identifier],
            controls_identifier=self.controls_identifier,
            **kwargs,
        )
        augsynth = pysyncon.AugSynth()
        try:
            augsynth.fit(dataprep)
        except Exception as e:
            self.fail(f"Augsynth fit with single treated in list failed: {e}.")
