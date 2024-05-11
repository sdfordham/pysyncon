from __future__ import annotations
from typing import Union, Optional, Literal

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .dataprep import Dataprep
from .base import BaseSynth, VanillaOptimMixin
from .inference import ConformalInference


OptimizerMethod_t = Literal[
    "Nelder-Mead", "Powell", "CG", "BFGS", "L-BFGS-B", "TNC", "COBYLA", "trust-constr"
]


class Synth(BaseSynth, VanillaOptimMixin):
    """Implementation of the synthetic control method due to
    Abadie & Gardeazabal :cite:`basque2003`.
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
        X1: Optional[pd.Series] = None,
        Z0: Optional[pd.DataFrame] = None,
        Z1: Optional[pd.Series] = None,
        custom_V: Optional[np.ndarray] = None,
        optim_method: OptimizerMethod_t = "Nelder-Mead",
        optim_initial: Literal["equal", "ols"] = "equal",
        optim_options: dict = {"maxiter": 1000},
    ) -> None:
        """Fit the model/calculate the weights. Either a :class:`Dataprep` object
        should be provided or otherwise matrices (:math:`X_0`, :math:`X_1`, :math:`Z_0`,
        :math:`Z_1`) should be provided (using the notation of Abadie &
        Gardeazabal :cite:`basque2003`).

        Parameters
        ----------
        dataprep : Dataprep, optional
            :class:`Dataprep` object containing data to model, by default None.
        X0 : pd.DataFrame, shape (m, c), optional
            Matrix with each column corresponding to a control unit and each
            row is covariates, by default None.
        X1 : pandas.Series, shape (m, 1), optional
            Column vector giving the covariate values for the treated unit, by
            default None.
        Z0 : pandas.DataFrame, shape (n, c), optional
            A matrix of the time series of the outcome variable with each
            column corresponding to a control unit and the rows are the time
            steps; the columns correspond with the columns of X0, by default
            None.
        Z1 : pandas.Series, shape (n, 1), optional
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
            the given optimisation method, by default `{'maxiter': 1000}`.

        Returns
        -------
        NoneType
            None

        Raises
        ------
        ValueError
            if neither a `Dataprep` object nor all of (:math:`X_0`, :math:`X_1`,
            :math:`Z_0`, :math:`Z_1`) are supplied.
        TypeError
            if (:math:`X1`, :math:`Z1`) are not of type `pandas.Series`.
        ValueError
            if `optim_initial=ols` and there is collinearity in the data.
        ValueError
            if `optim_initial` is not one of `'equal'` or `'ols'`.
        """
        if dataprep:
            if (
                isinstance(dataprep.treatment_identifier, (list, tuple))
                and len(dataprep.treatment_identifier) > 1
            ):
                raise ValueError("Synth requires exactly one treated unit.")
            self.dataprep = dataprep
            X0, X1 = dataprep.make_covariate_mats()
            Z0, Z1 = dataprep.make_outcome_mats()
        else:
            if X0 is None or X1 is None or Z0 is None or Z1 is None:
                raise ValueError(
                    "dataprep must be set or (X0, X1, Z0, Z1) must all be set."
                )
            if not isinstance(X1, pd.Series) or not isinstance(Z1, pd.Series):
                raise TypeError("X1 and Z1 must be of type `pandas.Series`.")

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
        self.W_names = Z0.columns

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

    def summary(
        self,
        round: int = 3,
        X0: Optional[pd.DataFrame] = None,
        X1: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
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
        X0 : pd.DataFrame, shape (n_cov, n_controls), optional
            Matrix with each column corresponding to a control unit and each
            row is a covariate. If no dataprep is set, then this must be
            supplied along with X1, by default None.
        X1 : pandas.Series, shape (n_cov, 1), optional
            Column vector giving the covariate values for the treated unit.
            If no dataprep is set, then this must be supplied along with Z1,
            by default None.

        Returns
        -------
        pandas.DataFrame
            Summary data.

        Raises
        ------
        ValueError
            If there is no V matrix available
        ValueError
            If there is no :class:`Dataprep` object set or (Z0, Z1) is not supplied
        ValueError
            If there is no weight matrix available
        """
        if self.V is None:
            raise ValueError("No V matrix available; fit data first.")
        summary_ser = super().summary(round=round, X0=X0, X1=X1)

        V = pd.Series(self.V, index=summary_ser.index, name="V")
        return pd.concat([V, summary_ser], axis=1).round(round)

    def confidence_interval(
        self,
        alpha: float,
        time_periods: list,
        tol: float,
        pre_periods: Optional[list] = None,
        dataprep: Optional[Dataprep] = None,
        X0: Optional[pd.DataFrame] = None,
        X1: Optional[pd.Series] = None,
        Z0: Optional[pd.DataFrame] = None,
        Z1: Optional[pd.Series] = None,
        custom_V: Optional[np.ndarray] = None,
        optim_method: OptimizerMethod_t = None,
        optim_initial: Literal["equal", "ols"] = None,
        optim_options: dict = None,
        method: Literal["conformal"] = "conformal",
        max_iter: int = 50,
        step_sz: Optional[float] = None,
        step_sz_div: float = 20.0,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Confidence intervals obtained from test-inversion, where
        the p-values are obtained by adjusted refits of the data
        following Chernozhukov et al. :cite:`inference2021`.

        Parameters
        ----------
        alpha : float
            The required significance level, e.g. alpha = 0.05 will
            yield a confidence level of 100 * (1 - alpha) = 95%.
        time_periods : list
            The time-periods to calculate confidence intervals for.
        tol : float
            The required tolerance (accuracy) required when calculating the
            lower/upper cut-off point of the confidence interval. The search
            will try to obtain this tolerance level but will not exceed `max_iter`
            iterations trying to achieve that.
        pre_periods : Optional[list], optional
            The time-periods to use for the optimization when refitting the
            data with the adjusted outcomes, optional.
        dataprep : Optional[Dataprep], optional
            Dataprep object defining the study data, if this is not supplied
            then either self.dataprep must be set or else (X0, X1, Z0, Z1) must
            all be supplied, by default None.
        X0 : pd.DataFrame, shape (m, c), optional
            Matrix with each column corresponding to a control unit and each
            row is covariates, if this is not supplied then either `dataprep` must
             be supplied or `self.dataprep` must be set by default None.
        X1 : pandas.Series, shape (m, 1), optional
            Column vector giving the covariate values for the treated unit, if
            this is not supplied then either `dataprep` must
             be supplied or `self.dataprep` must be set by default None.
        Z0 : pandas.DataFrame, shape (n, c), optional
            A matrix of the time series of the outcome variable with each
            column corresponding to a control unit and the rows are the time
            steps; the columns correspond with the columns of X0, if this
            is not supplied then either `dataprep` must be supplied or
            `self.dataprep` must be set by default None.
        Z1 : pandas.Series, shape (n, 1), optional
            Column vector giving the outcome variable values over time for the
            treated unit, if this is not supplied then either `dataprep` must
             be supplied or `self.dataprep` must be set by default None.
        custom_V : numpy.ndarray, shape (c, c), optional
            Provide a V matrix (using the notation of the Abadie, Diamond &
            Hainmueller paper), the optimisation problem will only then be
            solved for the weight matrix W. This is the same argument
            as in the `fit` method, by default None.
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

            This is the same argument as in the `fit` method, by default
            'Nelder-Mead'.
        optim_initial : str, optional
            Starting value for the outer optimisation, possible starting
            values are

                - 'equal', where the weights are all equal,
                - 'ols', which uses a starting value obtained for fitting a
                  regression.

            This is the same argument as in the `fit` method, by default
            'equal'.
        optim_options : dict, optional
            options to provide to the outer part of the optimisation, value
            options are any option that can be provided to scipy minimize for
            the given optimisation method. This is the same argument as in
             the `fit` method, by default `{'maxiter': 1000}`.
        method : str, optional
            The type of method to use when computing the confidence intervals,
            currently only conformal inference (`conformal`) is implemented,
            by default "conformal".
        max_iter : int, optional
            Maximum number of times to re-fit the data when trying to locate
            the lower/upper cut-off point and when binary searching for the
            cut-off point, by default 20.
        step_sz : Optional[float], optional
            Step size to use when searching for an interval that contains the
            lower or upper cut-off point of the confidence interval, by default None.
        step_sz_div : float, optional
            Alternative way to define step size: it is the fraction that defines
            step-size in terms of the standard deviation of the att, i.e. if
            `step_sz_div=20.0` then the step size used will be (att +/- 2.5 * std(att)) / 20.0,
            by default 20.0.
        verbose : bool, optional
            Print output, by default True.

        Returns
        -------
        pd.DataFrame
            A pandas.DataFrame indexed by `post_periods`, with 3 columns: `value` that
            gives the calculated treatment effect, `lower_ci` that gives the value
            defining the lower-end of the confidence interval, `upper_ci` that gives
            the value defining the upper-end of the confidence interval.

        Raises
        ------
        ValueError
            If there is no :class:`Dataprep` object set or (X0, X1, Z0, Z1) is not supplied or
            `self.dataprep` is not set.
        TypeError
            if (:math:`X1`, :math:`Z1`) are not of type `pandas.Series`.
        ValueError
            if `dataprep` is not set and `pre-periods` is not set.
        ValueError
            if an invalid option for `method` is given, currently only `conformal` is supported.
        """
        if method == "conformal":
            if dataprep is not None:
                X0, X1 = dataprep.make_covariate_mats()
                if pre_periods is None:
                    pre_periods = list(dataprep.time_optimize_ssr)
                if 1.0 / len(pre_periods) > alpha:
                    raise ValueError(
                        "Too few pre-intervention time-periods available for "
                        f"significance level `alpha`={alpha}, either increase `alpha` "
                        "or use more pre-intervention time-periods."
                    )
                all_time_periods = time_periods + list(pre_periods)
                Z0, Z1 = dataprep.make_outcome_mats(time_period=all_time_periods)
            elif self.dataprep is not None:
                X0, X1 = self.dataprep.make_covariate_mats()
                if pre_periods is None:
                    pre_periods = list(self.dataprep.time_optimize_ssr)
                if 1.0 / len(pre_periods) > alpha:
                    raise ValueError(
                        "Too few pre-intervention time-periods available for "
                        f"significance level `alpha`={alpha}, either increase `alpha` "
                        "or use more pre-intervention time-periods."
                    )
                all_time_periods = time_periods + list(pre_periods)
                Z0, Z1 = self.dataprep.make_outcome_mats(time_period=all_time_periods)
            else:
                if X0 is None or X1 is None or Z0 is None or Z1 is None:
                    raise ValueError(
                        "dataprep must be set or (X0, X1, Z0, Z1) must all be set."
                    )
                if not isinstance(X1, pd.Series) or not isinstance(Z1, pd.Series):
                    raise TypeError("X1 and Z1 must be of type `pandas.Series`.")
                if pre_periods is None:
                    raise ValueError("`pre_periods` must be set if not using dataprep.")
                if 1.0 / len(pre_periods) > alpha:
                    raise ValueError(
                        "Too few pre-intervention time-periods available for "
                        f"significance level `alpha`={alpha}, either increase `alpha` "
                        "or use more pre-intervention time-periods."
                    )

            scm_fit_args = {"X0": X0, "X1": X1}
            if custom_V is not None:
                scm_fit_args["custom_V"] = custom_V
            if optim_method:
                scm_fit_args["optim_method"] = optim_method
            if optim_initial:
                scm_fit_args["optim_initial"] = optim_initial
            if optim_options:
                scm_fit_args["optim_options"] = optim_options

            conformal_inf = ConformalInference()
            df_cis = conformal_inf.confidence_intervals(
                alpha=alpha,
                scm=self,
                Z0=Z0,
                Z1=Z1,
                pre_periods=pre_periods,
                post_periods=time_periods,
                scm_fit_args=scm_fit_args,
                max_iter=max_iter,
                tol=tol,
                step_sz=step_sz,
                step_sz_div=step_sz_div,
                verbose=verbose,
            )
            return df_cis
        else:
            raise ValueError("Invalid option for `method`.")
