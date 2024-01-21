import unittest
from unittest.mock import patch, Mock
import numpy as np
import pandas as pd

from pysyncon import Dataprep, Synth
from pysyncon.utils import HoldoutSplitter, CrossValidationResult, PlaceboTest


class TestHoldoutSplitter(unittest.TestCase):
    def test_values(self):
        cases = [(3, 3, 1), (3, 3, 2), (5, 1, 1), (5, 1, 2)]
        for case in cases:
            with self.subTest(case=case):
                rows, columns, holdout = case
                df = pd.DataFrame(np.random.random(size=(rows, columns)))
                ser = pd.Series(np.random.random(size=rows))

                iter_len = 0
                for df_, df_h, ser_, ser_h in HoldoutSplitter(
                    df=df, ser=ser, holdout_len=holdout
                ):
                    self.assertIsInstance(df_, pd.DataFrame)
                    pd.testing.assert_frame_equal(
                        df_,
                        df.drop(index=df.index[iter_len : iter_len + holdout,]),
                    )

                    self.assertIsInstance(ser_, pd.Series)
                    pd.testing.assert_series_equal(
                        ser_,
                        ser.drop(index=ser.index[iter_len : iter_len + holdout]),
                    )

                    self.assertIsInstance(df_h, pd.DataFrame)
                    pd.testing.assert_frame_equal(
                        df_h,
                        df.iloc[iter_len : iter_len + holdout,],
                    )

                    self.assertIsInstance(ser_h, pd.Series)
                    pd.testing.assert_series_equal(
                        ser_h,
                        ser.iloc[iter_len : iter_len + holdout],
                    )
                    iter_len += 1
                self.assertEqual(iter_len - 1, rows - holdout)

    def test_errs(self):
        cases = [(1, 1, 2, 2), (2, 2, 1, 1), (3, 2, 1, 2), (2, 1, 2, 3)]
        for case in cases:
            with self.subTest(case=case):
                df_rows, df_cols, holdout, ser_rows = case

                df = pd.DataFrame(np.random.random(size=(df_rows, df_cols)))
                ser = pd.Series(np.random.random(size=ser_rows))

                self.assertRaises(
                    ValueError,
                    HoldoutSplitter,
                    df=df,
                    ser=ser,
                    holdout_len=holdout,
                )

        cases = [(1, 1, 0, 1), (2, 2, 2, 2), (3, 3, 4, 3)]
        for case in cases:
            with self.subTest(case=case):
                df_rows, df_cols, holdout, ser_rows = case

                df = pd.DataFrame(np.random.random(size=(df_rows, df_cols)))
                ser = pd.Series(np.random.random(size=ser_rows))

                self.assertRaises(
                    ValueError,
                    HoldoutSplitter,
                    df=df,
                    ser=ser,
                    holdout_len=holdout,
                )


class TestCrossValidationResult(unittest.TestCase):
    def test_best_lambda(self):
        cases = [1, 2, 3, 10]
        for case in cases:
            with self.subTest(case=case):
                cv_result = CrossValidationResult(
                    lambdas=np.random.random(size=case),
                    errors_mean=np.random.random(size=case),
                    errors_se=np.random.random(size=case),
                )

                best_lambda = cv_result.best_lambda()
                min_mean = cv_result.errors_mean.min()
                min_mean_idx = cv_result.errors_mean.argmin()
                min_mean_se = cv_result.errors_se[min_mean_idx]
                self.assertEqual(
                    best_lambda,
                    cv_result.lambdas[cv_result.errors_mean <= min_mean + min_mean_se]
                    .max()
                    .item(),
                )

                best_lambda = cv_result.best_lambda(min_1se=False)
                min_mean_idx = cv_result.errors_mean.argmin()
                self.assertEqual(best_lambda, cv_result.lambdas[min_mean_idx].item())

    @patch("pysyncon.utils.plt")
    def test_result_plot(self, mock_plt: Mock):
        cv_result = CrossValidationResult(
            lambdas=np.random.random(size=10),
            errors_mean=np.random.random(size=10),
            errors_se=np.random.random(size=10),
        )
        cv_result.plot()

        self.assertEqual(mock_plt.errorbar.call_count, 1)
        _, kwargs = mock_plt.errorbar.call_args
        self.assertEqual(kwargs["ecolor"], "black")
        self.assertEqual(kwargs["capsize"], 2)

        mock_plt.xlabel.assert_called_with("Lambda")
        mock_plt.ylabel.assert_called_with("Mean error")
        mock_plt.xscale.assert_called_with("log")
        mock_plt.yscale.assert_called_with("log")
        mock_plt.title.assert_called_with("Cross validation result")
        mock_plt.grid.assert_called()
        mock_plt.show.assert_called()


class TestPlaceboTests(unittest.TestCase):
    def setUp(self):
        # 1 -> treated, (2, 3) -> controls
        self.dataprep = Dataprep(
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
        self.synth = Synth()
        self.synth.fit(dataprep=self.dataprep)

        self.placebo_test = PlaceboTest()
        self.placebo_test.fit(dataprep=self.dataprep, scm=self.synth)

    @patch("pysyncon.utils.plt")
    def test_gaps_plot(self, mock_plt: Mock):
        self.placebo_test.gaps_plot()

        self.assertEqual(mock_plt.plot.call_count, 2)
        _, kwargs = mock_plt.plot.call_args
        self.assertEqual(kwargs["color"], "black")
        self.assertEqual(kwargs["alpha"], 1.0)
        mock_plt.axvline.assert_not_called()
        mock_plt.grid.assert_called()

    @patch("pysyncon.utils.plt")
    def test_gaps_plot_axvline(self, mock_plt: Mock):
        self.placebo_test.gaps_plot(treatment_time=3)

        mock_plt.axvline.assert_called()
        _, kwargs = mock_plt.axvline.call_args
        self.assertEqual(kwargs["ymin"], 0.05)
        self.assertEqual(kwargs["ymax"], 0.95)
        self.assertEqual(kwargs["linestyle"], "dashed")

    @patch("pysyncon.utils.plt")
    def test_gaps_plot_mspe_threshold(self, mock_plt: Mock):
        self.placebo_test.gaps_plot(treatment_time=3, mspe_threshold=1)
