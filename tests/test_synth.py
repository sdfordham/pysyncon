import unittest
from unittest.mock import patch, Mock
import numpy as np
import pandas as pd

import pysyncon


class TestSynth(unittest.TestCase):
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
        synth = pysyncon.Synth()
        self.assertRaises(ValueError, synth.fit, dataprep)

        dataprep = pysyncon.Dataprep(
            treatment_identifier=self.treatment_identifier,
            controls_identifier=self.controls_identifier,
            **kwargs,
        )
        synth = pysyncon.Synth()

        # Run with normal controls list
        synth.fit(dataprep)

        dataprep = pysyncon.Dataprep(
            treatment_identifier=[self.treatment_identifier],
            controls_identifier=self.controls_identifier,
            **kwargs,
        )
        synth = pysyncon.Synth()

        # Run with a list of treatment identifiers
        synth.fit(dataprep)

    def test_X0_X1_fit(self):
        synth = pysyncon.Synth()

        # Neither dataprep nor matrices set
        self.assertRaises(ValueError, synth.fit)

        # X1 needs to be pd.Series
        X0 = pd.DataFrame(np.random.rand(5, 5))
        X1 = pd.DataFrame(np.random.rand(5, 2))
        Z0 = pd.DataFrame(np.random.rand(5, 5))
        Z1 = pd.DataFrame(np.random.rand(5, 2))
        self.assertRaises(TypeError, synth.fit, X0=X0, X1=X1, Z0=Z0, Z1=Z1)

        # X1 needs to be pd.Series
        X0 = pd.DataFrame(np.random.rand(5, 5))
        X1 = pd.DataFrame(np.random.rand(5, 1))
        Z0 = pd.DataFrame(np.random.rand(5, 5))
        Z1 = pd.DataFrame(np.random.rand(5, 1))
        self.assertRaises(TypeError, synth.fit, X0=X0, X1=X1, Z0=Z0, Z1=Z1)

    @patch("pysyncon.base.plt")
    def test_path_plot(self, mock_plt: Mock):
        kwargs = {
            "foo": self.foo,
            "predictors": self.predictors,
            "predictors_op": self.predictors_op,
            "dependent": self.dependent,
            "unit_variable": self.unit_variable,
            "time_variable": self.time_variable,
            "treatment_identifier": self.treatment_identifier,
            "controls_identifier": self.controls_identifier,
            "time_predictors_prior": self.time_predictors_prior,
            "time_optimize_ssr": self.time_optimize_ssr,
            "special_predictors": self.special_predictors,
        }

        dataprep = pysyncon.Dataprep(**kwargs)
        synth = pysyncon.Synth()
        # No weight matrix set
        self.assertRaises(ValueError, synth.path_plot)

        X0, X1 = dataprep.make_covariate_mats()
        Z0, Z1 = dataprep.make_outcome_mats()
        synth.fit(X0=X0, X1=X1, Z0=Z0, Z1=Z1)
        # No Dataprep object available
        self.assertRaises(ValueError, synth.path_plot)

        synth.fit(dataprep=dataprep)
        synth.path_plot()

        self.assertEqual(mock_plt.plot.call_count, 2)
        first_call, second_call = mock_plt.plot.call_args_list

        _, first_call_kwargs = first_call
        self.assertEqual(first_call_kwargs["color"], "black")
        self.assertEqual(first_call_kwargs["linewidth"], 1)
        self.assertEqual(first_call_kwargs["label"], dataprep.treatment_identifier)

        _, second_call_kwargs = second_call
        self.assertEqual(second_call_kwargs["color"], "black")
        self.assertEqual(second_call_kwargs["linewidth"], 1)
        self.assertEqual(second_call_kwargs["linestyle"], "dashed")
        self.assertEqual(second_call_kwargs["label"], "Synthetic")

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

    @patch("pysyncon.base.plt")
    def test_gaps_plot(self, mock_plt: Mock):
        kwargs = {
            "foo": self.foo,
            "predictors": self.predictors,
            "predictors_op": self.predictors_op,
            "dependent": self.dependent,
            "unit_variable": self.unit_variable,
            "time_variable": self.time_variable,
            "treatment_identifier": self.treatment_identifier,
            "controls_identifier": self.controls_identifier,
            "time_predictors_prior": self.time_predictors_prior,
            "time_optimize_ssr": self.time_optimize_ssr,
            "special_predictors": self.special_predictors,
        }

        dataprep = pysyncon.Dataprep(**kwargs)
        synth = pysyncon.Synth()
        # No weight matrix set
        self.assertRaises(ValueError, synth.gaps_plot)

        X0, X1 = dataprep.make_covariate_mats()
        Z0, Z1 = dataprep.make_outcome_mats()
        synth.fit(X0=X0, X1=X1, Z0=Z0, Z1=Z1)
        # No Dataprep object available
        self.assertRaises(ValueError, synth.gaps_plot)

        synth.fit(dataprep=dataprep)
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

    def test_weight(self):
        synth = pysyncon.Synth()
        # No weight matrix set
        self.assertRaises(ValueError, synth.weights)

    def test_summary(self):
        kwargs = {
            "foo": self.foo,
            "predictors": self.predictors,
            "predictors_op": self.predictors_op,
            "dependent": self.dependent,
            "unit_variable": self.unit_variable,
            "time_variable": self.time_variable,
            "treatment_identifier": self.treatment_identifier,
            "controls_identifier": self.controls_identifier,
            "time_predictors_prior": self.time_predictors_prior,
            "time_optimize_ssr": self.time_optimize_ssr,
            "special_predictors": self.special_predictors,
        }

        dataprep = pysyncon.Dataprep(**kwargs)
        synth = pysyncon.Synth()
        # No weight matrix set
        self.assertRaises(ValueError, synth.summary)
        X0, X1 = dataprep.make_covariate_mats()
        Z0, Z1 = dataprep.make_outcome_mats()
        synth.fit(X0=X0, X1=X1, Z0=Z0, Z1=Z1)
        # No Dataprep object available
        self.assertRaises(ValueError, synth.summary)

        synth.V = None
        # No V matrix available
        self.assertRaises(ValueError, synth.summary)

    def test_att(self):
        synth = pysyncon.Synth()
        # No weight matrix set
        self.assertRaises(ValueError, synth.att, range(1))

    def test_metrics(self):
        kwargs = {
            "foo": self.foo,
            "predictors": self.predictors,
            "predictors_op": self.predictors_op,
            "dependent": self.dependent,
            "unit_variable": self.unit_variable,
            "time_variable": self.time_variable,
            "treatment_identifier": self.treatment_identifier,
            "controls_identifier": self.controls_identifier,
            "time_predictors_prior": self.time_predictors_prior,
            "time_optimize_ssr": self.time_optimize_ssr,
            "special_predictors": self.special_predictors,
        }

        dataprep = pysyncon.Dataprep(**kwargs)
        synth = pysyncon.Synth()

        X0, X1 = dataprep.make_covariate_mats()
        Z0, Z1 = dataprep.make_outcome_mats()
        synth.fit(X0=X0, X1=X1, Z0=Z0, Z1=Z1)
        # No Dataprep object available
        self.assertRaises(ValueError, synth.mspe)
        self.assertRaises(ValueError, synth.mape)
        self.assertRaises(ValueError, synth.mae)

        del synth

        synth = pysyncon.Synth()
        synth.dataprep = dataprep
        # No weights availble/fit not run available
        self.assertRaises(ValueError, synth.mspe)
        self.assertRaises(ValueError, synth.mape)
        self.assertRaises(ValueError, synth.mae)

    def test_confidence_intervals(self):
        kwargs = {
            "foo": self.foo,
            "predictors": self.predictors,
            "predictors_op": self.predictors_op,
            "dependent": self.dependent,
            "unit_variable": self.unit_variable,
            "time_variable": self.time_variable,
            "treatment_identifier": self.treatment_identifier,
            "controls_identifier": self.controls_identifier,
            "time_predictors_prior": self.time_predictors_prior,
            "time_optimize_ssr": self.time_optimize_ssr,
            "special_predictors": self.special_predictors,
        }

        dataprep = pysyncon.Dataprep(**kwargs)
        synth = pysyncon.Synth()
        synth.fit(dataprep=dataprep)

        # Bad option
        self.assertRaises(
            ValueError,
            synth.confidence_interval,
            alpha=0.5,
            time_periods=[4],
            tol=0.01,
            method="foo",
        )

        # Run with dataprep supplied
        synth.confidence_interval(
            alpha=0.5, time_periods=[4], dataprep=dataprep, tol=0.01
        )

        # Too few time periods for alpha value
        self.assertRaises(
            ValueError,
            synth.confidence_interval,
            alpha=0.05,
            time_periods=[4],
            tol=0.01,
            dataprep=dataprep,
        )

        # Run without dataprep supplied
        synth.confidence_interval(alpha=0.5, time_periods=[4], tol=0.01)

        # Too few time periods for alpha value
        self.assertRaises(
            ValueError,
            synth.confidence_interval,
            alpha=0.05,
            time_periods=[4],
            tol=0.01,
        )

        # Without dataprep supplied or matrices
        synth.dataprep = None
        self.assertRaises(
            ValueError, synth.confidence_interval, alpha=0.5, time_periods=[4], tol=0.01
        )

        # No pre-periods supplied
        synth.dataprep = None
        X0, X1 = dataprep.make_covariate_mats()
        Z0, Z1 = dataprep.make_outcome_mats(time_period=[1, 2, 3, 4])
        self.assertRaises(
            ValueError,
            synth.confidence_interval,
            alpha=0.5,
            time_periods=[4],
            tol=0.01,
            X0=X0,
            X1=X1,
            Z0=Z0,
            Z1=Z1,
        )

        # Bad alpha value
        self.assertRaises(
            ValueError,
            synth.confidence_interval,
            alpha=0.05,
            time_periods=[4],
            pre_periods=[1, 2, 3],
            tol=0.01,
            X0=X0,
            X1=X1,
            Z0=Z0,
            Z1=Z1,
        )

        # Dataframes supplied instead of series
        X1 = X1.to_frame()
        Z1 = Z1.to_frame()
        self.assertRaises(
            TypeError,
            synth.confidence_interval,
            alpha=0.5,
            time_periods=[4],
            pre_periods=[1, 2, 3],
            tol=0.01,
            X0=X0,
            X1=X1,
            Z0=Z0,
            Z1=Z1,
        )
