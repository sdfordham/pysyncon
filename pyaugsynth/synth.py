from typing import Union, Optional, Literal
import numpy as np
import pandas as pd
from scipy.optimize import minimize, Bounds, LinearConstraint


class Synth:
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs

    def fit(
        self,
        X0: pd.DataFrame,
        X1: Union[pd.Series, pd.DataFrame],
        Z0: pd.DataFrame,
        Z1: Union[pd.Series, pd.DataFrame],
        custom_V: Optional[np.ndarray] = None,
        optim_method: Literal["Nelder-Mead", "BFGS"] = "Nelder-Mead",
        optim_initial: Literal["equal", "ols"] = "equal",
        optim_options: dict = {"maxiter": 1000},
    ):
        X = pd.concat([X0, X1], axis=1)
        X_scaled = X.divide(X.var(axis=1).pow(0.5), axis=0)
        X0_scaled, X1_scaled = X_scaled.drop(columns=X1.name), X_scaled[X1.name]

        X0_arr = X0_scaled.to_numpy()
        X1_arr = X1_scaled.to_numpy()
        Z0_arr = Z0.to_numpy()
        Z1_arr = Z1.to_numpy()

        if custom_V is not None:
            W, loss_W, loss_V = self.optimize_W(
                V=custom_V, X0=X0_arr, X1=X1_arr, Z0=Z0_arr, Z1=Z1_arr
            )
            return W, loss_W, custom_V, loss_V

        n_r, _ = X0_arr.shape

        if optim_initial == "equal":
            x0 = [1 / n_r] * n_r
        elif optim_initial == "ols":
            X_arr = np.hstack([X0_arr, X1_arr.reshape(-1, 1)])
            X_arr = np.hstack([np.array([1] * X_arr.shape[1], ndmin=2).T, X_arr.T])
            Z_arr = np.hstack([Z0_arr, Z1_arr.reshape(-1, 1)])
            beta = np.linalg.inv(X_arr.T @ X_arr) @ X_arr.T @ Z_arr.T
            beta = beta[1:,]

            x0 = np.diag(beta @ beta.T)
            x0 = x0 / sum(x0)
        else:
            raise ValueError("Unknown option for `optim_initial`.")
        
        def fun(x):
            V = np.diag(np.abs(x)) / np.sum(np.abs(x))
            _, _, loss_V = self.optimize_W(
                V=V, X0=X0_arr, X1=X1_arr, Z0=Z0_arr, Z1=Z1_arr
            )
            return loss_V
        
        res = minimize(fun=fun, x0=x0, method=optim_method, options=optim_options)
        V = np.diag(np.abs(res["x"])) / np.sum(np.abs(res["x"]))
        W, loss_W, loss_V = self.optimize_W(
            V=V, X0=X0_arr, X1=X1_arr, Z0=Z0_arr, Z1=Z1_arr
        )
        return W, loss_W, V, loss_V

    @staticmethod
    def optimize_W(
        V: np.ndarray,
        X0: np.ndarray,
        X1: np.ndarray,
        Z0: np.ndarray,
        Z1: np.ndarray,
        qp_method: Literal["SLSQP"] = "SLSQP",
        qp_options: dict = {"maxiter": 1000},
    ):
        _, n_c = X0.shape

        P = X0.T @ V @ X0
        q = -1.0 * X1.T @ V @ X0

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
        return W, loss_W, loss_V
