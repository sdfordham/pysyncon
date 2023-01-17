import unittest
from unittest.mock import patch, Mock
import numpy as np
import pandas as pd

import pysyncon


class TestSynth(unittest.TestCase):
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

    @patch("pysyncon.synth.plt")
    def test_path_plot(self, mock_plt: Mock):
        synth = pysyncon.Synth()
        synth.fit(dataprep=self.dataprep)
        synth.path_plot()

        self.assertEqual(mock_plt.plot.call_count, 2)
        first_call, second_call = mock_plt.plot.call_args_list

        _, first_call_kwargs = first_call
        self.assertEqual(first_call_kwargs["color"], "black")
        self.assertEqual(first_call_kwargs["linewidth"], 1)

        _, second_call_kwargs = second_call
        self.assertEqual(second_call_kwargs["color"], "black")
        self.assertEqual(second_call_kwargs["linewidth"], 1)
        self.assertEqual(second_call_kwargs["linestyle"], "dashed")

        mock_plt.axvline.assert_not_called()
        mock_plt.legend.assert_called()
        mock_plt.grid.assert_called_with(True)
        mock_plt.show.assert_called()

        synth.path_plot(treatment_time=3)
        mock_plt.axvline.assert_called_once()

        _, kwargs = mock_plt.axvline.call_args
        self.assertEqual(kwargs["x"], 3)
        self.assertEqual(kwargs["ymin"], 0.05)
        self.assertEqual(kwargs["ymax"], 0.95)
        self.assertEqual(kwargs["linestyle"], "dashed")

    @patch("pysyncon.synth.plt")
    def test_gaps_plot(self, mock_plt: Mock):
        synth = pysyncon.Synth()
        synth.fit(dataprep=self.dataprep)
        synth.gaps_plot()

        self.assertEqual(mock_plt.plot.call_count, 1)
        _, kwargs = mock_plt.plot.call_args

        self.assertEqual(kwargs["color"], "black")
        self.assertEqual(kwargs["linewidth"], 1)

        mock_plt.axvline.assert_not_called()
        mock_plt.grid.assert_called_with(True)
        mock_plt.show.assert_called()

        synth.path_plot(treatment_time=3)
        mock_plt.axvline.assert_called_once()

        _, kwargs = mock_plt.axvline.call_args
        self.assertEqual(kwargs["x"], 3)
        self.assertEqual(kwargs["ymin"], 0.05)
        self.assertEqual(kwargs["ymax"], 0.95)
        self.assertEqual(kwargs["linestyle"], "dashed")
