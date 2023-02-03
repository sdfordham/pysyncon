from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class HoldoutSplitter:
    """Iterator that prepares the time series for cross-validation by
    progressively removing blocks of length `holdout_len`.
    """
    def __init__(self, df: pd.DataFrame, ser: pd.Series, holdout_len: int = 1):
        """Iterator that prepares the time series for cross-validation by
        progressively removing blocks of length `holdout_len`.

        Parameters
        ----------
        df : pandas.DataFrame, shape (r, c)
            Dataframe that will be split for the cross-validation.
        ser : pandas.Series, shape (r, 1)
            Series that will split for the cross-validation.
        holdout_len : int, optional
            Number of days to remove in each iteration, by default 1.

        Raises
        ------
        ValueError
            if df and ser do not have the same number of rows.
        ValueError
            if `holdout_len` is not >= 1.
        ValueError
            if `holdout_len` is larger than the number of rows of df.
        """
        if df.shape[0] != ser.shape[0]:
            raise ValueError("`df` and `ser` must have the same number of rows.")
        if holdout_len < 1:
            raise ValueError("`holdout_len` must be at least 1.")
        if holdout_len >= df.shape[0]:
            raise ValueError("`holdout_len` must be less than df.shape[0]")
        self.df = df
        self.ser = ser
        self.holdout_len = holdout_len
        self.idx = 0

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        if (self.idx + self.holdout_len) > self.df.shape[0]:
            raise StopIteration
        holdout = slice(self.idx, self.idx + self.holdout_len)

        df_holdout = self.df.iloc[holdout,]  # fmt: skip
        ser_holdout = self.ser.iloc[holdout]

        df = self.df.drop(index=self.df.index[holdout])
        ser = self.ser.drop(index=self.ser.index[holdout])

        self.idx += 1
        return df, df_holdout, ser, ser_holdout


@dataclass
class CrossValidationResult:
    """Convenience class for holding the results of the cross-validation
    procedure from the AugSynth.
    """
    lambdas: np.ndarray
    errors_mean: np.ndarray
    errors_se: np.ndarray

    def best_lambda(self, min_1se: bool = True) -> float:
        """Return the best lambda.

        Parameters
        ----------
        min_1se : bool, optional
            return the largest lambda within 1 standard error of the minimum
            , by default True

        Returns
        -------
        float
        """
        if min_1se:
            return (
                self.lambdas[
                    self.errors_mean
                    <= self.errors_mean.min()
                    + self.errors_se[self.errors_mean.argmin()]
                ]
                .max()
                .item()
            )
        return self.lambdas[self.errors_mean.argmin()].item()

    def plot(self) -> None:
        """Plots the mean errors against the lambda values with the standard
        errors as error bars.
        """
        plt.errorbar(
            x=self.lambdas,
            y=self.errors_mean,
            yerr=self.errors_se,
            ecolor="black",
            capsize=2,
        )
        plt.xlabel("Lambda")
        plt.ylabel("Mean error")
        plt.xscale("log")
        plt.yscale("log")
        plt.title("Cross validation result")
        plt.grid()
        plt.show()
