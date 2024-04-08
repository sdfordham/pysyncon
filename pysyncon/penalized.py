from __future__ import annotations
from typing import Optional, Literal, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize, Bounds, LinearConstraint

from .dataprep import Dataprep
from .base import BaseSynth


class PenalizedOptimMixin:
    @staticmethod
    def w_optimize(
        V_mat: np.ndarray,
        X0: np.ndarray,
        X1: np.ndarray,
        lambda_: float,
        qp_method: Literal["SLSQP"] = "SLSQP",
        qp_options: dict = {"maxiter": 1000},
    ) -> tuple[np.ndarray, float]:
        """Solves the weight optimisation problem in the penalized setting,
        see Abadie & L'Hour :cite:`penalized2021`.

        Parameters
        ----------
        V_mat : numpy.ndarray, shape (c, c)
            The V matrix (using the notation of the Abadie, Diamond &
            Hainmueller paper, this matrix is denoted by Γ in the Abadie and
            L'Hour paper).
        X0 : numpy.ndarray, shape (c, m)
            Matrix with each column corresponding to a control unit and each
            row is covariates.
        X1 : numpy.ndarray, shape (c,)
            Column vector giving the covariate values for the treated unit.
        lambda_ : float,
            Regularization parameter.
        qp_method : str, optional
            Minimization routine to use in scipy minimize to solve the problem
            , by default "SLSQP"
        qp_options : dict, optional
            Options for scipy minimize, by default {"maxiter": 1000}

        Returns
        -------
        tuple[np.ndarray, float]
            tuple of the optimal weights and the loss

        :meta private:
        """
        _, n_c = X0.shape

        diff = np.subtract(X0, X1.reshape(-1, 1))
        r = np.diag(diff.T @ V_mat @ diff)

        P = X0.T @ V_mat @ X0
        q = -1.0 * X1.T @ V_mat @ X0 + (lambda_ / 2.0) * r.T

        def fun(x):
            return q.T @ x + 0.5 * x.T @ P @ x

        bounds = Bounds(lb=np.full(n_c, 0.0), ub=np.full(n_c, 1.0))
        constraints = LinearConstraint(A=np.full(n_c, 1.0), lb=1.0, ub=1.0)

        x0 = np.full(n_c, 1 / n_c)
        res = minimize(
            fun=fun,
            x0=x0,
            bounds=bounds,
            constraints=constraints,
            method=qp_method,
            options=qp_options,
        )
        W, loss_W = res["x"], res["fun"]
        return W, loss_W.item()


class PenalizedSynth(BaseSynth, PenalizedOptimMixin):
    """Implementation of the penalized synthetic control method due to
    Abadie & L'Hour :cite:`penalized2021`.
    """

    def __init__(self) -> None:
        super().__init__()
        self.loss_W: Optional[float] = None
        self.lambda_: Optional[float] = None

    def fit(
        self,
        dataprep: Optional[Dataprep] = None,
        X0: Optional[pd.DataFrame] = None,
        X1: Optional[Union[pd.Series, pd.DataFrame]] = None,
        lambda_: Optional[float] = 0.01,
        custom_V: Optional[np.ndarray] = None,
    ) -> None:
        """Fit the model/calculate the weights.

        Parameters
        ----------
        dataprep : Dataprep, optional
            :class:`Dataprep` object containing data to model, by default None.
        X0 : pd.DataFrame, shape (c, m), optional
            Matrix with each column corresponding to a control unit and each
            row is a covariate value, by default None.
        X1 : pandas.Series, shape (c, 1), optional
            Column vector giving the covariate values for the treated unit, by
            default None.
        lambda_ : float, optional
            Ridge parameter to use, default 0.01
        custom_V : numpy.ndarray, shape (c, c), optional
            Provide a V matrix (using the notation of the Abadie, Diamond &
            Hainmueller paper, this matrix is denoted by Γ in the Abadie and
            L'Hour paper), if not provided then the identity matrix is used
            (equal importance to all covariates).

        Returns
        -------
        NoneType
            None

        Raises
        ------
        ValueError
            if neither a Dataprep object nor all of (X0, X1) are
            supplied.
        """
        if dataprep:
            if (
                isinstance(dataprep.treatment_identifier, (list, tuple))
                and len(dataprep.treatment_identifier) > 1
            ):
                raise ValueError("PenalizedSynth requires exactly one treated unit.")
            self.dataprep = dataprep
            X0, X1 = dataprep.make_covariate_mats()
        else:
            if X0 is None or X1 is None:
                raise ValueError("dataprep must be set or (X0, X1) must all be set.")
            if not isinstance(X1, pd.Series):
                raise TypeError("X1 must be of type `pandas.Series`.")
        self.lambda_ = lambda_

        X = pd.concat([X0, X1], axis=1)
        X_scaled = X.divide(X.std(axis=1), axis=0)
        X0_scaled, X1_scaled = X_scaled.drop(columns=X1.name), X_scaled[X1.name]

        X0_arr = X0_scaled.to_numpy()
        X1_arr = X1_scaled.to_numpy()

        if custom_V is None:
            V_mat = np.identity(X0_arr.shape[0])
        else:
            V_mat = np.diag(custom_V)

        W, loss_W = self.w_optimize(V_mat=V_mat, X0=X0_arr, X1=X1_arr, lambda_=lambda_)
        self.W, self.loss_W = W, loss_W
