from __future__ import annotations
from typing import Optional, Callable, Literal

import numpy as np
import pandas as pd

from pysyncon.base import BaseSynth


class ConformalInference:
    """Implementation of the conformal inference based confidence intervals
    following Chernozhukov et al. :cite:`inference2021`
    """

    def __init__(self) -> None:
        pass

    def confidence_intervals(
        self,
        alpha: float,
        scm: BaseSynth,
        Z0: pd.DataFrame,
        Z1: pd.Series,
        pre_periods: list,
        post_periods: list,
        tol: float = 0.1,
        max_iter: int = 50,
        step_sz: Optional[float] = None,
        step_sz_div: float = 20.0,
        verbose: bool = True,
        scm_fit_args: dict = {},
    ) -> pd.DataFrame:
        """Confidence intervals obtained from test-inversion, where
        the p-values are obtained by adjusted re-fits of the data
        following Chernozhukov et al. :cite:`inference2021`.

        Parameters
        ----------
        alpha : float
            The required significance level, e.g. alpha = 0.05 will
            yield a confidence level of 100 * (1 - alpha) = 95%.
        scm : BaseSynth
            The synth object to calculate the confidence interval for.
        Z0 : pandas.DataFrame, shape (n, c)
            A matrix of the time series of the outcome variable with each
            column corresponding to a control unit and the rows are the time
            steps.
        Z1 : pd.Series
            Column vector giving the outcome variable values over time for the
            treated unit.
        tol : float
            The required tolerance (accuracy) required when calculating the
            lower/upper cut-off point of the confidence interval. The search
            will try to obtain this tolerance level but will not exceed `max_iter`
            iterations trying to achieve that.
        pre_periods : list
            The time-periods to use for the optimization when refitting the
            data with the adjusted outcomes.
        post_periods : list
            The time-periods to calculate confidence intervals for.
        max_iter : int, optional
            Maximum number of times to re-fit the data when trying to locate
            the lower/upper cut-off point, by default 50
        step_sz : Optional[float], optional
            Step size to use when searching for an interval that contains the
            lower or upper cut-off point of the confidence interval, by default None
        step_sz_div : float, optional
            Alternative way to define step size: it is the fraction that defines
            step-size in terms of the standard deviation of the att, i.e. if
            `step_sz_div=20.0` then the step size used will be (att +/- 2.5 * std(att)) / 20.0,
            by default 20.0
        verbose : bool, optional
            Print output, by default True
        scm_fit_args : dict, optional
            A dictionary defining anything extra that should be provided to the
            synthetic control object `fit` method when doing the refits, by default {}

        Returns
        -------
        pd.DataFrame
            A pandas.DataFrame indexed by `post_periods`, with 3 columns: `value` that
            gives the calculated treatment effect, `lower_ci` that gives the value
            defining the lower-end of the confidence interval, `upper_ci` that gives
            the value defining the upper-end of the confidence interval.

        Raises
        ------
        TypeError
            if `alpha` is not a float
        ValueError
            if `alpha` is not in the open interval (0, 1).
        TypeError
            if `max_iter` is not an integer
        ValueError
            if `max_iter` is not at least 1
        TypeError
            if `tol` is not a float
        ValueError
            if `tol` is less than 0.0
        TypeError
            if `step_sz` is not a float
        ValueError
            if `step_sz` is not greater than 0.0
        TypeError
            if `step_sz_div` is not a float
        ValueError
            if `step_sz_div` is not greater than 0.0
        """
        if not isinstance(alpha, float):
            raise TypeError("`alpha` must be a float")
        elif not 0.0 < alpha < 1.0:
            raise ValueError("`alpha` must be greater than 0.0 and less than 1.0")
        if not isinstance(max_iter, int):
            raise TypeError("`max_iter` must be an integer")
        elif max_iter < 1:
            raise ValueError("`max_iter` must be at least 1")
        if not isinstance(tol, float):
            raise TypeError("`tol` must be a float")
        elif tol <= 0.0:
            raise ValueError("`tol` must be greater than 0.0")
        if step_sz != None:
            if not isinstance(step_sz, float):
                raise TypeError("`step_sz` should be a float")
            elif step_sz <= 0.0:
                raise ValueError("`step_sz` should be greater than 0.0")
            elif step_sz <= tol:
                raise ValueError("`step_sz` must be greater than `tol`.")
        if not isinstance(step_sz_div, float):
            raise TypeError("`step_sz_div` must be a float")
        elif step_sz_div <= 0.0:
            raise ValueError("`step_sz_div` must be greater than 0.0")
        if scm.W is None:
            raise ValueError("No weight matrix available; fit data first.")

        gaps = scm._gaps(Z0=Z0, Z1=Z1)
        if step_sz is None:
            # Try to guess a step-size
            if len(post_periods) > 1:
                factor = np.std(gaps.loc[post_periods])
            else:
                factor = gaps.loc[post_periods].item() / 2.0
            step_sz = 2.5 * factor / step_sz_div
            if step_sz <= tol:
                # Failed to guess a sensible step-size :(
                step_sz = 1.1 * tol

        conf_interval = dict()
        n_periods = len(post_periods)
        for idx, post_period in enumerate(post_periods, 1):
            if verbose:
                print(
                    f"({idx}/{n_periods}) Calculating confidence interval "
                    f"for time-period t={post_period}..."
                )
            new_time_range = pre_periods + [post_period]
            Z0_new, Z1_new = Z0.loc[new_time_range], Z1.loc[new_time_range]
            Z1_post_orig = Z1_new.loc[post_period].item()

            def _compute_p_value(g):
                Z1_new.loc[post_period] = Z1_post_orig - g
                scm.fit(Z0=Z0_new, Z1=Z1_new, **scm_fit_args)
                _gaps = scm._gaps(Z0=Z0_new, Z1=Z1_new)

                u_hat = _gaps.loc[new_time_range]
                u_hat_post = u_hat.loc[post_period]
                return np.mean(abs(u_hat) >= abs(u_hat_post))

            lower_ci = self._root_search(
                fn=lambda x: _compute_p_value(x) - alpha,
                x0=gaps.loc[post_period],
                direction=-1.0,
                tol=tol,
                step_sz=step_sz,
                max_iter=max_iter,
            )

            upper_ci = self._root_search(
                fn=lambda x: _compute_p_value(x) - alpha,
                x0=gaps.loc[post_period],
                direction=1.0,
                tol=tol,
                step_sz=step_sz,
                max_iter=max_iter,
            )

            conf_interval[post_period] = (lower_ci, upper_ci)
            if verbose:
                print(
                    f"\t{100 * (1 - alpha)}% CI: [{round(lower_ci, 3)}, {round(upper_ci, 3)}]"
                )

        df_ci = pd.DataFrame.from_dict(
            conf_interval, orient="index", columns=["lower_ci", "upper_ci"]
        )
        df_ci = pd.concat([gaps.loc[post_periods].rename("value"), df_ci], axis=1)
        df_ci.index.name = "time"
        return df_ci

    def _root_search(
        self,
        fn: Callable,
        x0: float,
        direction: Literal[+1, -1],
        tol: float,
        step_sz: float,
        max_iter: int,
        theta: float = 0.75,
        phi: float = 1.3,
    ) -> float:
        """Search for a root

        Parameters
        ----------
        fn : callable
            Function to find a root of
        x0 : float
            Starting point
        direction : int
            Direction, either -1.0 or +1.0.
        tol : float
            Tolerance
        step_sz : float
            Step size in the search
        max_iter : int
            Maximum number of iterations
        theta : float, optional
            Step size reduction factor, should be positive and < 1.0, by default 0.75
        phi : float, optional
            Step size increase factor, should be positive and > 1.0, by default 1.3

        Returns
        -------
        float
            Root of the function

        Raises
        ------
        Exception
            if `max_iter` iterations exceeded before satisfying tolerance condition.

        :meta private:
        """
        x, gamma = x0, step_sz
        for _ in range(max_iter):
            if gamma <= tol:
                return x
            y = fn(x + gamma * direction)
            if y > 0.0:
                x = x + gamma * direction
                gamma = phi * gamma
            else:
                gamma = theta * gamma
        raise Exception(
            "Exceeded `max_iter` iterations without satisfying tolerance requirement."
        )
