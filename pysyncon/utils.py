from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class HoldoutSplitter:
    def __init__(self, df: pd.DataFrame, ser: pd.Series, holdout_len: int = 1):
        self.df = df
        self.ser = ser
        self.holdout_len = holdout_len
        self.idx = 0

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        if (self.idx + self.holdout_len) == self.df.shape[0]:
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
    lambdas: np.ndarray
    errors_mean: np.ndarray
    errors_se: np.ndarray

    def best_lambda(self, min_1se: bool = True) -> float:
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
