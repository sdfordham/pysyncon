from __future__ import annotations
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .dataprep import Dataprep, IsinArg_t
from .synth import WeightOptimizerMixin
from .utils import HoldoutSplitter, CrossValidationResult


class AugSynth(WeightOptimizerMixin):
    def __init__(self) -> None:
        self.dataprep: Optional[Dataprep] = None
        self.lambda_: Optional[float] = None
        self.W: Optional[np.ndarray] = None
        self.cv_result: Optional[CrossValidationResult] = None

    def fit(
        self, dataprep: Optional[Dataprep], lambda_: Optional[float] = None
    ) -> None:
        self.dataprep = dataprep
        Z0, Z1 = self.dataprep.make_covariate_mats()
        X0, X1 = self.dataprep.make_outcome_mats()

        X0_demean, X1_demean, Z0_normal, Z1_normal = self._normalize(X0, X1, Z0, Z1)
        X0_stacked = pd.concat([X0_demean, Z0_normal], axis=0)
        X1_stacked = pd.concat([X1_demean, Z1_normal], axis=0)

        if lambda_ is None:
            lambdas = self.generate_lambdas(X0)
            self.cv_result = self.cross_validate(X0, X1, lambdas)
            self.lambda_ = self.cv_result.best_lambda()
        else:
            self.lambda_ = lambda_

        V_mat = np.diag([1.0 / X0.shape[0]] * X0.shape[0])
        W, _, _ = self.w_optimize(
            V_mat=V_mat,
            X0=X0.to_numpy(),
            X1=X1.to_numpy(),
            Z0=X0.to_numpy(),
            Z1=X1.to_numpy(),
        )

        W_ridge = self.solve_ridge(
            X1_stacked.to_numpy(), X0_stacked.to_numpy(), W, self.lambda_
        )
        self.W = W + W_ridge

    @staticmethod
    def solve_ridge(
        A: np.ndarray, B: np.ndarray, W: np.ndarray, lambda_: float
    ) -> np.ndarray:
        M = A - B @ W
        N = np.linalg.inv(B @ B.T + lambda_ * np.identity(B.shape[0]))
        return M @ N @ B

    def _normalize(
        self, X0: pd.DataFrame, X1: pd.Series, Z0: pd.DataFrame, Z1: pd.Series
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        X0_demean = X0.subtract(X0.mean(axis=1), axis=0)
        X1_demean = X1.subtract(X0.mean(axis=1), axis=0)

        Z0_demean = Z0.subtract(Z0.mean(axis=1), axis=0)
        Z1_demean = Z1.subtract(Z0.mean(axis=1), axis=0)

        Z0_std = Z0_demean.std(axis=1)
        X0_std = X0_demean.to_numpy().std(ddof=1).item()

        Z0_normal = Z0_demean.divide(Z0_std, axis=0) * X0_std
        Z1_normal = Z1_demean.divide(Z0_std, axis=0) * X0_std
        return X0_demean, X1_demean, Z0_normal, Z1_normal

    def cross_validate(
        self, X0: np.ndarray, X1: np.ndarray, lambdas: np.ndarray, holdout_len: int = 1
    ) -> CrossValidationResult:
        V = np.identity(X0.shape[0] - holdout_len)
        res = list()
        for X0_t, X0_v, X1_t, X1_v in HoldoutSplitter(X0, X1, holdout_len=holdout_len):
            W, _, _ = self.w_optimize(
                V_mat=V,
                X0=X0_t.to_numpy(),
                X1=X1_t.to_numpy(),
                Z0=X0_t.to_numpy(),
                Z1=X1_t.to_numpy(),
            )
            this_res = list()
            for l in lambdas:
                ridge_weights = self.solve_ridge(A=X1_t, B=X0_t, W=W, lambda_=l)
                W_aug = W + ridge_weights
                err = (X1_v - X0_v @ W_aug).pow(2).sum()
                this_res.append(err.item())
            res.append(this_res)
        means = np.array(res).mean(axis=0)
        ses = np.array(res).std(axis=0) / np.sqrt(len(lambdas))
        return CrossValidationResult(lambdas, means, ses)

    def generate_lambdas(
        self, X: pd.DataFrame, lambda_min_ratio: float = 1e-8, n_lambda: int = 20
    ) -> np.ndarray:
        _, sing, _ = np.linalg.svd(X.T)
        lambda_max = sing[0].item() ** 2.0
        scaler = lambda_min_ratio ** (1 / n_lambda)
        return lambda_max * (scaler ** np.array(range(n_lambda)))

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
        ts_synthetic = (Z0 * self.W).sum(axis=1).rename("Synthetic")
        Z1.plot(ylabel=self.dataprep.dependent, color="black", linewidth=1)
        ts_synthetic.plot(
            ylabel=self.dataprep.dependent,
            color="black",
            linewidth=1,
            linestyle="dashed",
        )

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
        ts_gap.plot(ylabel=self.dataprep.dependent, color="black", linewidth=1)

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

    def l2_imbalance(self) -> tuple[float, float]:
        if self.dataprep is None:
            raise ValueError("dataprep must be set for L2 imbalance.")
        if self.W is None:
            raise ValueError("No weight matrix available; fit data first.")

        Z0, Z1 = self.dataprep.make_covariate_mats()
        W_eq = np.array([1 / Z0.shape[1]] * Z0.shape[1])

        l2_imbalance = np.sqrt((Z0 @ self.W - Z1).pow(2).sum()).item()
        l2_imbalance_eq = np.sqrt((Z0 @ W_eq - Z1).pow(2).sum()).item()
        l2_imbalance_scaled = l2_imbalance / l2_imbalance_eq
        return l2_imbalance, l2_imbalance_scaled
