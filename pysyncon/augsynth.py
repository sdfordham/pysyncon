from __future__ import annotations
from typing import Any, Optional

import numpy as np
import pandas as pd

from .dataprep import Dataprep
from .base import BaseSynth, VanillaOptimMixin
from .utils import CrossValidationResult


class AugSynth(BaseSynth, VanillaOptimMixin):
    """Implementation of the augmented synthetic control method due to Ben-
    Michael, Feller & Rothstein :cite:`augsynth2021`.

    The implementation follows the augsynth R package with the option
    `progfunc="Ridge"`.

    In addition to the weights, the fit computes the `ridge_mhat` outcome
    model (a ridge regression of the post-treatment control outcomes on the
    design matrix), which is used for the `bias_est`/`avg_bias` diagnostics.
    """

    def __init__(self) -> None:
        super().__init__()
        self.lambda_: Optional[float] = None
        self.cv_result: Optional[CrossValidationResult] = None
        self.pre_periods: Optional[list] = None
        self.post_periods: Optional[list] = None
        self.t_int: Optional[Any] = None
        self.synw: Optional[np.ndarray] = None
        self.beta: Optional[np.ndarray] = None
        self.X_cent: Optional[pd.DataFrame] = None
        self.ridge_mhat: Optional[pd.DataFrame] = None
        self.bias_est: Optional[pd.Series] = None
        self.avg_bias: Optional[float] = None

    def fit(
        self,
        dataprep: Dataprep,
        lambda_: Optional[float] = None,
        use_covariates: bool = True,
    ) -> None:
        """Fit the model/calculate the weights.

        Parameters
        ----------
        dataprep : Dataprep, optional
            :class:`Dataprep` object containing data to model.
        lambda_ : float, optional
            Ridge parameter to use. If not supplied, then it is obtained by
            cross-validation, by default None
        use_covariates : bool, optional
            Whether or not to include the covariates from the :class:`Dataprep`
            object in the design matrix (concatenated with the pre-treatment
            outcomes, so that the covariates are balanced jointly with the
            outcomes), by default True

        Raises
        ------
        ValueError
            if `dataprep.time_optimize_ssr` is not exactly the pre-treatment
            time periods.
        """
        if (
            isinstance(dataprep.treatment_identifier, (list, tuple))
            and len(dataprep.treatment_identifier) > 1
        ):
            raise ValueError("AugSynth requires exactly one treated unit.")
        self.dataprep = dataprep

        # The pre-treatment periods are exactly the periods before the first
        # post-treatment period (the treatment time `t_int`), following the
        # convention of the augsynth R package.
        times = sorted(dataprep.foo[dataprep.time_variable].unique())
        is_pre = dataprep.foo[dataprep.time_variable].isin(dataprep.time_optimize_ssr)
        pre_periods = sorted(set(dataprep.foo.loc[is_pre, dataprep.time_variable]))
        post_periods = sorted(set(dataprep.foo.loc[~is_pre, dataprep.time_variable]))
        if not post_periods:
            raise ValueError(
                "No post-treatment time periods found in "
                f"`dataprep.foo[{dataprep.time_variable}]`."
            )
        t_int = min(post_periods)
        if pre_periods != [t for t in times if t < t_int]:
            raise ValueError(
                "`time_optimize_ssr` must be exactly the pre-treatment time "
                "periods for AugSynth (all time-periods less than the "
                f"treatment time {t_int})."
            )
        self.pre_periods = pre_periods
        self.post_periods = post_periods
        self.t_int = t_int

        X0, X1 = self.dataprep.make_outcome_mats(time_period=self.pre_periods)

        if use_covariates:
            Z0, Z1 = self.dataprep.make_covariate_mats()
            X0_demean, X1_demean, Z0_normal, Z1_normal = self._normalize(X0, X1, Z0, Z1)
            X0_stacked = pd.concat([X0_demean, Z0_normal], axis=0)
            X1_stacked = pd.concat([X1_demean, Z1_normal], axis=0)
        else:
            # Center the outcomes at the control unit mean (per time period)
            X0_stacked = X0.subtract(X0.mean(axis=1), axis=0)
            X1_stacked = X1.subtract(X0.mean(axis=1), axis=0).rename(X1.name)

        if lambda_ is None:
            lambdas = self.generate_lambdas(X0_stacked)
            t0 = X0.shape[0]
            self.cv_result = self.cross_validate(X0_stacked, X1_stacked, lambdas, t0)
            self.lambda_ = self.cv_result.best_lambda()
        else:
            self.lambda_ = lambda_

        n_r, _ = X0_stacked.shape
        V_mat = np.eye(n_r)
        W, _ = self.w_optimize(
            V_mat=V_mat,
            X0=X0_stacked.to_numpy(),
            X1=X1_stacked.to_numpy(),
            qp_options={"maxiter": 2000, "ftol": 1e-12},
        )
        self.synw = W

        W_ridge = self.solve_ridge(
            X1_stacked.to_numpy(), X0_stacked.to_numpy(), W, self.lambda_
        )
        self.W = W + W_ridge

        # Outcome model (`ridge_mhat`): ridge regression of the post-treatment
        # control outcomes on the design matrix. It is used to estimate the
        # bias of the augmented weights (see the augsynth R package,
        # `fit_ridgeaug_formatted`).
        Y0, Y1 = self.dataprep.make_outcome_mats(time_period=self.post_periods)
        # center at the control unit mean per time period (n_c x T_post)
        y_c = Y0.subtract(Y0.mean(axis=1), axis=0).T.to_numpy()

        if use_covariates:
            X_all_c = pd.concat([X0_demean, X1_demean], axis=1)
            Z0_c = Z0.subtract(Z0.mean(axis=1), axis=0)
            Z1_c = Z1.subtract(Z0.mean(axis=1), axis=0).rename(Z1.name)
            # NOTE: replicate the R package exactly - the covariate block of
            # the evaluation matrix uses the UNscaled centered covariates
            # (the design used for the weights and beta uses the scaled
            # covariates).
            F_all = pd.concat([X_all_c, pd.concat([Z0_c, Z1_c], axis=1)], axis=0)
        else:
            X_all_c = pd.concat([X0_stacked, X1_stacked], axis=1)
            F_all = X_all_c

        design = X0_stacked.to_numpy()  # m x n_c
        N = np.linalg.inv(design @ design.T + self.lambda_ * np.eye(design.shape[0]))
        self.beta = N @ (design @ y_c)  # m x T_post
        self.X_cent = X_all_c
        self.ridge_mhat = pd.DataFrame(
            F_all.T.to_numpy() @ self.beta,
            index=F_all.columns,
            columns=self.post_periods,
        )
        m1 = self.ridge_mhat.loc[X1.name]
        m0 = self.ridge_mhat.loc[list(X0.columns)]
        self.bias_est = m1 - self.synw @ m0
        self.avg_bias = self.bias_est.mean().item()

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
        X1_demean = X1.subtract(X0.mean(axis=1), axis=0).rename(X1.name)

        Z0_demean = Z0.subtract(Z0.mean(axis=1), axis=0)
        Z1_demean = Z1.subtract(Z0.mean(axis=1), axis=0)

        Z0_std = Z0_demean.std(axis=1)
        X0_std = X0_demean.to_numpy().std(ddof=1).item()

        Z0_normal = Z0_demean.divide(Z0_std, axis=0) * X0_std
        Z1_normal = Z1_demean.divide(Z0_std, axis=0) * X0_std
        return X0_demean, X1_demean, Z0_normal, Z1_normal

    def cross_validate(
        self,
        X0: pd.DataFrame,
        X1: pd.Series,
        lambdas: np.ndarray,
        t0: int,
        holdout_len: int = 1,
    ) -> CrossValidationResult:
        """Method that calculates the mean error and standard error to the mean
        error using a cross-validation procedure for the given ridge parameter
        values.

        In each fold a block of `holdout_len` pre-treatment time periods (the
        first `t0` rows of `X0`/`X1`) is held out, the synthetic control is
        re-fit on the remaining pre-treatment time periods and the error
        measures how well the augmented weights predict the treated unit's
        held-out pre-treatment outcomes.

        :meta private:
        """
        if holdout_len < 1 or t0 < 2 or holdout_len >= t0:
            raise ValueError(
                "`holdout_len` must be at least 1 and less than the number of "
                f"pre-treatment time periods (got `t0`={t0}, "
                f"`holdout_len`={holdout_len})."
            )
        res = list()
        for i in range(t0 - holdout_len):
            holdout = slice(i, i + holdout_len)
            X0_t = X0.drop(index=X0.index[holdout])
            X0_v = X0.iloc[holdout]
            X1_t = X1.drop(index=X1.index[holdout])
            X1_v = X1.iloc[holdout]

            W, _ = self.w_optimize(
                V_mat=np.identity(X0_t.shape[0]),
                X0=X0_t.to_numpy(),
                X1=X1_t.to_numpy(),
                qp_options={"maxiter": 2000, "ftol": 1e-12},
            )
            this_res = list()
            for l in lambdas:
                ridge_weights = self.solve_ridge(A=X1_t, B=X0_t, W=W, lambda_=l)
                W_aug = W + ridge_weights
                err = (X1_v - X0_v @ W_aug).pow(2).sum()
                this_res.append(err.item())
            res.append(this_res)
        means = np.array(res).mean(axis=0)
        ses = np.array(res).std(axis=0, ddof=1) / np.sqrt(t0 - holdout_len)
        return CrossValidationResult(lambdas, means, ses)

    def generate_lambdas(
        self, X: pd.DataFrame, lambda_min_ratio: float = 1e-8, n_lambda: int = 20
    ) -> np.ndarray:
        """Generate a suitable set of lambdas to run the cross-validation
        procedure on.

        :meta private:
        """
        sing = np.linalg.svd(X, compute_uv=False)
        lambda_max = sing[0] ** 2.0
        scaler = lambda_min_ratio ** (1 / n_lambda)
        return lambda_max * (scaler ** (np.arange(0, n_lambda + 1) - 1))
