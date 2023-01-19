from __future__ import annotations
from typing import Union, Optional, Literal

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .dataprep import Dataprep
from .base import BaseSynth


OptimizerMethod_t = Literal[
    "Nelder-Mead", "Powell", "CG", "BFGS", "L-BFGS-B", "TNC", "COBYLA", "trust-constr"
]


class Synth(BaseSynth):
    def __init__(self) -> None:
        super().__init__()
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
