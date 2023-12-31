from __future__ import annotations
from typing import Union, Optional, Literal

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .dataprep import Dataprep
from .base import BaseSynth, VanillaOptimMixin


OptimizerMethod_t = Literal[
    "Nelder-Mead", "Powell", "CG", "BFGS", "L-BFGS-B", "TNC", "COBYLA", "trust-constr"
]


class Synth(BaseSynth, VanillaOptimMixin):
    """Implementation of the synthetic control method due to
    `Abadie, Diamond & Hainmueller <http://dx.doi.org/10.1198/jasa.2009.ap08746>`_.
    """

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
        """Fit the model/calculate the weights. Either a :class:`Dataprep` object
        should be provided or otherwise matrices (X0, X1, Z0, Z1)
        should be provided (using the same notation as the
        `Abadie, Diamond & Hainmueller <http://dx.doi.org/10.1198/jasa.2009.ap08746>`_
        paper).

        Parameters
        ----------
        dataprep : Dataprep, optional
            :class:`Dataprep` object containing data to model, by default None.
        X0 : pd.DataFrame, shape (m, c), optional
            Matrix with each column corresponding to a control unit and each
            row is covariates, by default None.
        X1 : pandas.Series | pandas.DataFrame, shape (m, 1), optional
            Column vector giving the covariate values for the treated unit, by
            default None.
        Z0 : pandas.DataFrame, shape (n, c), optional
            A matrix of the time series of the outcome variable with each
            column corresponding to a control unit and the rows are the time
            steps; the columns correspond with the columns of X0, by default
            None.
        Z1 : pandas.Series | pandas.DataFrame, shape (n, 1), optional
            Column vector giving the outcome variable values over time for the
            treated unit, by default None.
        custom_V : numpy.ndarray, shape (c, c), optional
            Provide a V matrix (using the notation of the Abadie, Diamond &
            Hainmueller paper), the optimisation problem will only then be
            solved for the weight matrix W, by default None.
        optim_method : str, optional
            Optimisation method to use for the outer optimisation, can be
            any of the valid options for scipy minimize that do not require a
            jacobian matrix, namely

                - 'Nelder-Mead'
                - 'Powell'
                - 'CG'
                - 'BFGS'
                - 'L-BFGS-B'
                - 'TNC'
                - 'COBYLA'
                - 'trust-constr'

            By default 'Nelder-Mead'.
        optim_initial : str, optional
            Starting value for the outer optimisation, possible starting
            values are

                - 'equal', where the weights are all equal,
                - 'ols', which uses a starting value obtained for fitting a
                  regression.

            By default 'equal'.
        optim_options : dict, optional
            options to provide to the outer part of the optimisation, value
            options are any option that can be provided to scipy minimize for
            the given optimisation method, by default {'maxiter': 1000}.

        Returns
        -------
        NoneType
            None

        Raises
        ------
        ValueError
            if neither a Dataprep object nor all of (X0, X1, Z0, Z1) are
            supplied.
        ValueError
            if `optim_initial=ols` there is collinearity in the data.
        ValueError
            if `optim_initial` is not one of `'equal'` or `'ols'`.
        """
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
            W, loss_W = self.w_optimize(V_mat=V_mat, X0=X0_arr, X1=X1_arr)
            loss_V = self.calc_loss_V(W=W, Z0=Z0_arr, Z1=Z1_arr)
            self.W, self.loss_W, self.V, self.loss_V = W, loss_W, custom_V, loss_V
            return

        n_r, _ = X0_arr.shape

        if optim_initial == "equal":
            x0 = [1 / n_r] * n_r
        elif optim_initial == "ols":
            X_arr = np.hstack([X0_arr, X1_arr.reshape(-1, 1)])
            X_arr = np.hstack([np.full((X_arr.shape[1], 1), 1), X_arr.T])
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
            W, _ = self.w_optimize(V_mat=V_mat, X0=X0_arr, X1=X1_arr)
            loss_V = self.calc_loss_V(W=W, Z0=Z0_arr, Z1=Z1_arr)
            return loss_V

        res = minimize(fun=fun, x0=x0, method=optim_method, options=optim_options)
        V_mat = np.diag(np.abs(res["x"])) / np.sum(np.abs(res["x"]))
        W, loss_W = self.w_optimize(V_mat=V_mat, X0=X0_arr, X1=X1_arr)
        loss_V = self.calc_loss_V(W=W, Z0=Z0_arr, Z1=Z1_arr)

        self.W, self.loss_W, self.V, self.loss_V = W, loss_W, V_mat.diagonal(), loss_V

    @staticmethod
    def calc_loss_V(W: np.ndarray, Z0: np.ndarray, Z1: np.ndarray) -> float:
        """Calculates the V loss.

        Parameters
        ----------
        W : numpy.ndarray, shape (n,)
            Vector of the control weights
        Z0 : numpy.ndarray, shape (m, n)
            Matrix of the time series of the outcome variable with each
            column corresponding to a control unit and the rows are the time
            steps.
        Z1 : numpy.ndarray, shape (m,)
            Column vector giving the outcome variable values over time for the
            treated unit

        Returns
        -------
        float
            V loss.

        :meta private:
        """
        loss_V = (Z1 - Z0 @ W).T @ (Z1 - Z0 @ W) / len(Z0)
        return loss_V.item()

    def summary(self, round: int = 3) -> pd.DataFrame:
        """Generates a ``pandas.DataFrame`` with summary data. In particular,
        it will show the values of the V matrix for each predictor, then the
        next column will show the mean value of each predictor over the time
        period ``time_predictors_prior`` for the treated unit and the synthetic
        unit and finally there will be a column 'sample mean' that shows the
        mean value of each predictor over the time period
        ``time_predictors_prior`` across all the control units, i.e. this will
        be the same as a synthetic control where all the weights are equal.

        Parameters
        ----------
        round : int, optional
            Round the numbers to given number of places, by default 3

        Returns
        -------
        pandas.DataFrame
            Summary data.

        Raises
        ------
        ValueError
            If there is no :class:`Dataprep` object set
        ValueError
            If there is no weight matrix available
        ValueError
            If there is no V matrix available
        """
        if self.V is None:
            raise ValueError("No V matrix available; fit data first.")
        summary_ser = super().summary(round=round)

        V = pd.Series(self.V, index=summary_ser.index, name="V")
        return pd.concat([V, summary_ser], axis=1).round(round)
