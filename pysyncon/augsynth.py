from __future__ import annotations
from typing import Optional

import numpy as np
import pandas as pd

from .dataprep import Dataprep
from .base import BaseSynth, VanillaOptimMixin
from .utils import HoldoutSplitter, CrossValidationResult


class AugSynth(BaseSynth, VanillaOptimMixin):
    """Implementation of the augmented synthetic control method due to Ben-
    Michael, Feller & Rothstein :cite:`augsynth2021`.

    The implementation follows the augsynth R package with the option
    `progfunc="Ridge"`.
    """

    def __init__(self) -> None:
        super().__init__()
        self.lambda_: Optional[float] = None
        self.cv_result: Optional[CrossValidationResult] = None

    def fit(self, dataprep: Dataprep, lambda_: Optional[float] = None) -> None:
        """Fit the model/calculate the weights.

        Parameters
        ----------
        dataprep : Dataprep, optional
            :class:`Dataprep` object containing data to model.
        lambda_ : float, optional
            Ridge parameter to use. If not supplied, then it is obtained by
            cross-validation, by default None
        """
        if (
            isinstance(dataprep.treatment_identifier, (list, tuple))
            and len(dataprep.treatment_identifier) > 1
        ):
            raise ValueError("AugSynth requires exactly one treated unit.")
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

        n_r, _ = X0.shape
        V_mat = np.diag(np.full(n_r, 1 / n_r))
        W, _ = self.w_optimize(V_mat=V_mat, X0=X0.to_numpy(), X1=X1.to_numpy())

        W_ridge = self.solve_ridge(
            X1_stacked.to_numpy(), X0_stacked.to_numpy(), W, self.lambda_
        )
        self.W = W + W_ridge

    @staticmethod
    def solve_ridge(
        A: np.ndarray, B: np.ndarray, W: np.ndarray, lambda_: float
    ) -> np.ndarray:
        """Calculate the ridge adjustment to the weights.

        :meta private:
        """
        M = A - B @ W
        N = np.linalg.inv(B @ B.T + lambda_ * np.identity(B.shape[0]))
        return M @ N @ B

    def _normalize(
        self, X0: pd.DataFrame, X1: pd.Series, Z0: pd.DataFrame, Z1: pd.Series
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Normalise the data before the weight calculation.

        :meta private:
        """
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
        """Method that calculates the mean error and standard error to the mean
        error using a cross-validation procedure for the given ridge parameter
        values.

        :meta private:
        """
        V = np.identity(X0.shape[0] - holdout_len)
        res = list()
        for X0_t, X0_v, X1_t, X1_v in HoldoutSplitter(X0, X1, holdout_len=holdout_len):
            W, _ = self.w_optimize(V_mat=V, X0=X0_t.to_numpy(), X1=X1_t.to_numpy())
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
        """Generate a suitable set of lambdas to run the cross-validation
        procedure on.

        :meta private:
        """
        _, sing, _ = np.linalg.svd(X.T)
        lambda_max = sing[0].item() ** 2.0
        scaler = lambda_min_ratio ** (1 / n_lambda)
        return lambda_max * (scaler ** np.array(range(n_lambda)))
