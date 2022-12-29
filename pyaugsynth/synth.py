import numpy as np
import pandas as pd
from scipy.optimize import minimize, Bounds, LinearConstraint


def optim_W(
    V: np.ndarray, X0: np.ndarray, X1: np.ndarray, Z0: np.ndarray, Z1: np.ndarray
):
    _, n_c = X0.shape

    P = X0.T @ V @ X0
    q = -1.0 * X1.T @ V @ X0

    def fun(x):
        return q.T @ x + 0.5 * x.T @ P @ x

    bound_tol = 0.0
    bounds = Bounds(
        lb=np.array([bound_tol] * n_c).T, ub=np.array([1.0 - bound_tol] * n_c).T
    )
    constraints = LinearConstraint(
        A=np.array([1.0] * n_c), lb=np.array([1.0]), ub=np.array([1.0])
    )

    x0 = np.array([1 / n_c] * n_c)
    res = minimize(
        fun=fun,
        x0=x0,
        bounds=bounds,
        constraints=constraints,
        method="SLSQP",
        options={"maxiter": 1000},
    )
    W = res["x"]
    loss_W = res["fun"]
    loss_V = (Z1 - Z0 @ W).T @ (Z1 - Z0 @ W) / len(Z0)
    return W, loss_W, loss_V


def synth(
    X0: pd.DataFrame,
    X1: pd.DataFrame,
    Z0: pd.DataFrame,
    Z1: pd.DataFrame,
    custom_V=None,
    optim_method="Nelder-Mead",
    optim_options={"maxiter": 1000},
):
    X = pd.concat([X0, X1], axis=1)
    X_scaled = X.divide(X.var(axis=1).pow(0.5), axis=0)
    X0_scaled, X1_scaled = X_scaled.drop(columns=X1.name), X_scaled[X1.name]

    X0_arr = X0_scaled.to_numpy()
    X1_arr = X1_scaled.to_numpy()
    Z0_arr = Z0.to_numpy()
    Z1_arr = Z1.to_numpy()

    if custom_V is not None:
        W, loss_W, loss_V = optim_W(
            V=custom_V, X0=X0_arr, X1=X1_arr, Z0=Z0_arr, Z1=Z1_arr
        )
        return W, loss_W, custom_V, loss_V

    def fun(x):
        V = np.diag(np.abs(x)) / np.sum(np.abs(x))
        _, _, loss_V = optim_W(V=V, X0=X0_arr, X1=X1_arr, Z0=Z0_arr, Z1=Z1_arr)
        return loss_V

    # First run: start with equal weights
    n_r, _ = X0_arr.shape
    x0 = [1 / n_r] * n_r

    res = minimize(fun=fun, x0=x0, method=optim_method, options=optim_options)
    V_eq = np.diag(np.abs(res["x"])) / np.sum(np.abs(res["x"]))
    W_eq, loss_W_eq, loss_V_eq = optim_W(
        V=V_eq, X0=X0_arr, X1=X1_arr, Z0=Z0_arr, Z1=Z1_arr
    )

    # Second run: starting weights are OLS coefficients
    X_arr = np.hstack([X0_arr, X1_arr.reshape(-1, 1)])
    X_arr = np.hstack([np.array([1] * X_arr.shape[1], ndmin=2).T, X_arr.T])
    Z_arr = np.hstack([Z0_arr, Z1_arr.reshape(-1, 1)])
    beta = np.linalg.inv(X_arr.T @ X_arr) @ X_arr.T @ Z_arr.T
    beta = beta[
        1:,
    ]

    x0 = np.diag(beta @ beta.T)
    x0 = x0 / sum(x0)

    res = minimize(fun=fun, x0=x0, method=optim_method, options=optim_options)
    V_ols = np.diag(np.abs(res["x"])) / np.sum(np.abs(res["x"]))
    W_ols, loss_W_ols, loss_V_ols = optim_W(
        V=V_ols, X0=X0_arr, X1=X1_arr, Z0=Z0_arr, Z1=Z1_arr
    )

    if loss_V_eq < loss_V_ols:
        return W_eq, loss_W_eq, V_eq, loss_V_eq
    else:
        return W_ols, loss_W_ols, V_ols, loss_V_ols
