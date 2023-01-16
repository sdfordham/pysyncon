from typing import Union, Optional, Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds, LinearConstraint

from .dataprep import Dataprep, IsinArg_t


OptimizerMethod_t = Literal[
    "Nelder-Mead", "Powell", "CG", "BFGS", "L-BFGS-B", "TNC", "COBYLA", "trust-constr"
]


class WeightOptimizerMixin:
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


class Synth(WeightOptimizerMixin):
    def __init__(self) -> None:
        self.dataprep: Optional[Dataprep] = None
        self.W: Optional[np.ndarray] = None
        self.loss_W: Optional[float] = None
        self.V: Optional[np.ndarray] = None
        self.loss_V: Optional[float] = None

    def fit(
        self,
        dataprep: Optional[Dataprep] = None,
        X0: Optional[pd.DataFrame] = None,
        X1: Optional[Union[pd.Series, pd.DataFrame]] = None,
        Z0: Optional[pd.DataFrame] = None,
        Z1: Optional[Union[pd.Series, pd.DataFrame]] = None,
        custom_V: Optional[np.ndarray] = None,
        optim_method: OptimizerMethod_t = "Nelder-Mead",
        optim_initial: Literal["equal", "ols"] = "equal",
        optim_options: dict = {"maxiter": 1000},
    ) -> None:
        if dataprep:
            self.dataprep = dataprep
            X0, X1 = dataprep.make_covariate_mats()
            Z0, Z1 = dataprep.make_outcome_mats()
        else:
            if X0 is None or X1 is None or Z0 is None or Z1 is None:
                raise ValueError(
                    "dataprep must be set or (X0, X1, Z0, Z1) must all be set."
                )

        X = pd.concat([X0, X1], axis=1)
        X_scaled = X.divide(X.std(axis=1), axis=0)
        X0_scaled, X1_scaled = X_scaled.drop(columns=X1.name), X_scaled[X1.name]

        X0_arr = X0_scaled.to_numpy()
        X1_arr = X1_scaled.to_numpy()
        Z0_arr = Z0.to_numpy()
        Z1_arr = Z1.to_numpy()

        if custom_V is not None:
            V_mat = np.diag(custom_V)
            W, loss_W, loss_V = self.w_optimize(
                V_mat=V_mat, X0=X0_arr, X1=X1_arr, Z0=Z0_arr, Z1=Z1_arr
            )
            self.W, self.loss_W, self.V, self.loss_V = W, loss_W, custom_V, loss_V
            return

        n_r, _ = X0_arr.shape

        if optim_initial == "equal":
            x0 = [1 / n_r] * n_r
        elif optim_initial == "ols":
            X_arr = np.hstack([X0_arr, X1_arr.reshape(-1, 1)])
            X_arr = np.hstack([np.array([1] * X_arr.shape[1], ndmin=2).T, X_arr.T])
            Z_arr = np.hstack([Z0_arr, Z1_arr.reshape(-1, 1)])

            try:
                beta = np.linalg.inv(X_arr.T @ X_arr) @ X_arr.T @ Z_arr.T
            except np.linalg.LinAlgError:
                raise ValueError(
                    'Could not invert X^T.X required for `optim_initial="ols"`, '
                    "probably there is collinearity in your data."
                )

            beta = beta[1:,]  # fmt: skip
            x0 = np.diag(beta @ beta.T)
            x0 = x0 / sum(x0)
        else:
            raise ValueError("Unknown option for `optim_initial`.")

        def fun(x):
            V_mat = np.diag(np.abs(x)) / np.sum(np.abs(x))
            _, _, loss_V = self.w_optimize(
                V_mat=V_mat, X0=X0_arr, X1=X1_arr, Z0=Z0_arr, Z1=Z1_arr
            )
            return loss_V

        res = minimize(fun=fun, x0=x0, method=optim_method, options=optim_options)
        V_mat = np.diag(np.abs(res["x"])) / np.sum(np.abs(res["x"]))
        W, loss_W, loss_V = self.w_optimize(
            V_mat=V_mat, X0=X0_arr, X1=X1_arr, Z0=Z0_arr, Z1=Z1_arr
        )
        self.W, self.loss_W, self.V, self.loss_V = W, loss_W, V_mat.diagonal(), loss_V

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

        plt.plot(Z1, color="black", linewidth=1)
        plt.plot(ts_synthetic, color="black", linewidth=1, linestyle="dashed")
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
