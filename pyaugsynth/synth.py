import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint


def compute_V_loss(
    V: np.ndarray, X0: np.ndarray, X1: np.ndarray, Z0: np.ndarray, Z1: np.ndarray
):
    _, n_c = X0.shape

    P = X0.T @ V @ X0
    q = -1.0 * X1.T @ V @ X0

    def fun(x):
        return q.T @ x + 0.5 * x.T @ P @ x

    bounds = Bounds(lb=np.array([0.0] * n_c).T, ub=np.array([1.0] * n_c).T)
    constraints = LinearConstraint(
        A=np.array([1.0] * n_c), lb=np.array([1.0]), ub=np.array([1.0])
    )

    x0 = np.array([1 / n_c] * n_c)
    res = minimize(
        fun=fun, x0=x0, constraints=constraints, bounds=bounds, method="SLSQP"
    )
    W = res["x"]
    loss_V = (Z1 - Z0 @ W).T @ (Z1 - Z0 @ W)
    return loss_V / len(Z0)
