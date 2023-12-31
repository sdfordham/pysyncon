from __future__ import annotations
from typing import Optional

import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint

from .dataprep import Dataprep
from .base import BaseSynth


class PenalizedOptimMixin:
    @staticmethod
    def w_optimize(
        X0: np.ndarray,
        X1: np.ndarray,
        lambda_: float,
        initial: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, float]:
        """Solves the weight optimisation problem in the penalized setting.

        Parameters
        ----------
        X0 : numpy.ndarray, shape (m, c)
            Matrix with each column corresponding to a control unit and each
            row is covariates.
        X1 : numpy.ndarray, shape (m,)
            Column vector giving the covariate values for the treated unit.
        lambda_ : float,
            Regularization parameter.
        initial: numpy.ndarray, shape(m,), optional
            Initial point to use in the optimisation problem.

        Returns
        -------
        tuple[np.ndarray, float]
            tuple of the optimal weights and the loss

        :meta private:
        """
        _, n_c = X0.shape
        diff = np.subtract(X0, X1.reshape(-1, 1))
        r = np.linalg.norm(diff, axis=0)

        def fun(x):
            return (X1 - X0 @ x).T @ (X1 - X0 @ x) + lambda_ * (r.T @ x)

        bounds = Bounds(lb=np.full(n_c, 0.0), ub=np.full(n_c, 1.0))
        constraints = LinearConstraint(A=np.full(n_c, 1.0), lb=1.0, ub=1.0)

        if initial:
            x0 = initial
        else:
            x0 = np.full(n_c, 1 / n_c)

        res = minimize(fun=fun, x0=x0, bounds=bounds, constraints=constraints)
        W, loss_W = res["x"], res["fun"]
        return W, loss_W.item()


class PenalizedSynth(BaseSynth, PenalizedOptimMixin):
    """Implementation of the penalized synthetic control method due to
    Abadie & L'Hourblack.
    """

    def __init__(self) -> None:
        super().__init__()
        self.W: Optional[np.ndarray] = None
        self.lambda_: Optional[float] = None

    def fit(self, dataprep: Dataprep, lambda_: float) -> None:
        """Fit the model/calculate the weights.

        Parameters
        ----------
        dataprep : Dataprep
            :class:`Dataprep` object containing data to model.
        lambda_ : float
            Ridge parameter to use.
        """
        self.dataprep = dataprep
        self.lambda_ = lambda_

        X0_df, X1_df = dataprep.make_outcome_mats()
        X0, X1 = X0_df.to_numpy(), X1_df.to_numpy()

        W, _ = self.w_optimize(X0=X0, X1=X1, lambda_=lambda_)
        self.W = W
