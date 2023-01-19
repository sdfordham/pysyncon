from __future__ import annotations
from typing import Optional, Literal
from abc import ABCMeta, abstractmethod

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds, LinearConstraint

from .dataprep import Dataprep, IsinArg_t


class BaseSynth(metaclass=ABCMeta):
    def __init__(self) -> None:
        self.dataprep: Optional[Dataprep] = None
        self.W: Optional[np.ndarray] = None

    @abstractmethod
    def fit(*args, **kwargs) -> None:
        raise NotImplemented

    @staticmethod
    def w_optimize(
        V_mat: np.ndarray,
        X0: np.ndarray,
        X1: np.ndarray,
        Z0: np.ndarray,
        Z1: np.ndarray,
        qp_method: Literal["SLSQP"] = "SLSQP",
        qp_options: dict = {"maxiter": 1000},
    ) -> tuple[np.ndarray, float, float]:
        _, n_c = X0.shape

        P = X0.T @ V_mat @ X0
        q = -1.0 * X1.T @ V_mat @ X0

        def fun(x):
            return q.T @ x + 0.5 * x.T @ P @ x

        bounds = Bounds(lb=np.array([0.0] * n_c).T, ub=np.array([1.0] * n_c).T)
        constraints = LinearConstraint(A=np.array([1.0] * n_c), lb=1.0, ub=1.0)

        x0 = np.array([1 / n_c] * n_c)
        res = minimize(
            fun=fun,
            x0=x0,
            bounds=bounds,
            constraints=constraints,
            method=qp_method,
            options=qp_options,
        )
        W, loss_W = res["x"], res["fun"]
        loss_V = (Z1 - Z0 @ W).T @ (Z1 - Z0 @ W) / len(Z0)
        return W, loss_W.item(), loss_V.item()

    def path_plot(
        self,
        time_period: Optional[IsinArg_t] = None,
        treatment_time: Optional[int] = None,
        grid: bool = True,
    ) -> None:
        if self.W is None:
            raise ValueError("No weight matrix available; fit data first.")
        if self.dataprep is None:
            raise ValueError("dataprep must be set for automatic plots.")

        Z0, Z1 = self.dataprep.make_outcome_mats(time_period=time_period)
        ts_synthetic = (Z0 * self.W).sum(axis=1)

        plt.plot(Z1, color="black", linewidth=1, label=Z1.name)
        plt.plot(
            ts_synthetic,
            color="black",
            linewidth=1,
            linestyle="dashed",
            label="Synthetic",
        )
        plt.ylabel(self.dataprep.dependent)
        if treatment_time:
            plt.axvline(x=treatment_time, ymin=0.05, ymax=0.95, linestyle="dashed")
        plt.legend()
        plt.grid(grid)
        plt.show()

    def gaps_plot(
        self,
        time_period: Optional[IsinArg_t] = None,
        treatment_time: Optional[int] = None,
        grid: bool = True,
    ) -> None:
        if self.dataprep is None:
            raise ValueError("dataprep must be set for automatic plots.")
        if self.W is None:
            raise ValueError("No weight matrix available; fit data first.")

        Z0, Z1 = self.dataprep.make_outcome_mats(time_period=time_period)
        ts_synthetic = (Z0 * self.W).sum(axis=1)
        ts_gap = Z1 - ts_synthetic

        plt.plot(ts_gap, color="black", linewidth=1)
        plt.ylabel(self.dataprep.dependent)
        plt.hlines(
            y=0,
            xmin=min(ts_gap.index),
            xmax=max(ts_gap.index),
            color="black",
            linestyle="dashed",
        )
        if treatment_time:
            plt.axvline(x=treatment_time, ymin=0.05, ymax=0.95, linestyle="dashed")
        plt.grid(grid)
        plt.show()

    def weights(self, threshold: float = 0.0, round: int = 3) -> pd.Series:
        if self.W is None:
            raise ValueError("No weight matrix available; fit data first.")
        if self.dataprep is None:
            weights_ser = pd.Series(self.W, name="weights")
        else:
            weights_ser = pd.Series(
                self.W, index=list(self.dataprep.controls_identifier), name="weights"
            )
        return weights_ser[weights_ser >= threshold].round(round)

    def summary(self, round: int = 3) -> pd.DataFrame:
        if self.dataprep is None:
            raise ValueError("dataprep must be set for summary.")
        if self.W is None:
            raise ValueError("No weight matrix available: fit data first.")
        if self.V is None:
            raise ValueError("No V matrix available; fit data first.")
        X0, X1 = self.dataprep.make_covariate_mats()

        V = pd.Series(self.V, index=X1.index, name="V")
        treated = X1.rename("treated")
        synthetic = (X0 * self.W).sum(axis=1).rename("synthetic")
        sample_mean = X0.mean(axis=1).rename("sample mean")

        return pd.concat([V, treated, synthetic, sample_mean], axis=1).round(round)
