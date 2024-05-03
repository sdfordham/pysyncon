from __future__ import annotations
from typing import Optional, Any

import numpy as np
import pandas as pd

from pysyncon.base import BaseSynth


class ConformalInference:
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
        max_iter: int = 20,
        tol: float = 0.1,
        q: int = 2,
        step_sz: Optional[float] = None,
        step_sz_div: float = 20.0,
        verbose: bool = True,
        scm_fit_args: dict = {},
    ) -> pd.DataFrame:
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
        if not isinstance(q, int):
            raise TypeError("`q` must be an integer")
        elif q <= 1:
            raise ValueError("`q` must be at least 1")
        if step_sz:
            if not isinstance(step_sz, float):
                raise TypeError("`step_sz` should be a float")
            elif step_sz <= 0.0:
                raise ValueError("`step_sz` should be greater than 0.0")
        if not isinstance(step_sz_div, float):
            raise TypeError("`step_sz_div` must be a float")
        elif step_sz_div <= 0.0:
            raise ValueError("`step_sz_div` must be greater than 0.0")

        gaps = scm._gaps(Z0=Z0, Z1=Z1)
        att = np.mean(gaps.loc[post_periods])

        if len(post_periods) > 1:
            att_std = np.std(gaps.loc[post_periods])
        else:
            att_std = gaps.loc[post_periods].item() / 2.0

        if step_sz is None:
            step_sz = 2.5 * att_std / step_sz_div

        conf_interval = dict()
        for post_period in post_periods:
            new_time_range = pre_periods + [post_period]
            Z0_new, Z1_new = Z0.loc[new_time_range], Z1.loc[new_time_range]
            Z1_post_orig = Z1_new.loc[post_period].item()

            def _compute_p_value(g):
                Z1_new.loc[post_period] = Z1_post_orig - g
                scm.fit(Z0=Z0_new, Z1=Z1_new, **scm_fit_args)
                _gaps = scm._gaps(Z0=Z0_new, Z1=Z1_new)

                u_hat = _gaps.loc[new_time_range]
                u_hat_post = u_hat.loc[post_period]
                return np.mean(abs(u_hat).pow(q) >= pow(abs(u_hat_post), q))

            #################
            ## LOWER CI VALUE
            #################

            # Find an interval containing the cut-off
            x0 = att - 2.5 * att_std
            left = right = x0
            p_value = _compute_p_value(x0)
            if p_value < alpha:
                it = 0
                right_p_value = p_value
                while right_p_value < alpha and it < max_iter:
                    left = right
                    right = right + step_sz
                    it = it + 1
                    right_p_value = _compute_p_value(right)
                if right_p_value < alpha and it == max_iter:
                    raise Exception(
                        "Exceeded `max_iter` iterations without locating lower CI cut-off."
                    )
            else:
                it = 0
                left_p_value = p_value
                while left_p_value >= alpha and it < max_iter:
                    left = left - step_sz
                    right = left
                    it = it + 1
                    left_p_value = _compute_p_value(left)
                if left_p_value >= alpha and it == max_iter:
                    raise Exception(
                        "Exceeded `max_iter` iterations without locating lower CI cut-off."
                    )

            # Binary search the interval [left, right]
            it = 0
            while right - left > tol and it < max_iter:
                mid = (left + right) / 2.0
                p_value_mid = _compute_p_value(mid)
                if p_value_mid > alpha:
                    right = mid
                else:
                    left = mid
                it = it + 1
            lower_ci = (left + right) / 2.0

            #################
            ## UPPER CI VALUE
            #################

            # Find an interval containing the cut-off
            x0 = att + 2.5 * att_std
            left = right = x0
            p_value = _compute_p_value(x0)
            if p_value > alpha:
                it = 0
                right_p_value = p_value
                while right_p_value > alpha and it < max_iter:
                    left = right
                    right = right + step_sz
                    it = it + 1
                    right_p_value = _compute_p_value(right)
                if right_p_value > alpha and it == max_iter:
                    raise Exception(
                        "Exceeded `max_iter` iterations without locating upper CI cut-off."
                    )
            else:
                it = 0
                left_p_value = p_value
                while left_p_value <= alpha and it < max_iter:
                    left = left - step_sz
                    right = left
                    it = it + 1
                    left_p_value = _compute_p_value(left)
                if left_p_value <= alpha and it == max_iter:
                    raise Exception(
                        "Exceeded `max_iter` iterations without locating upper CI cut-off."
                    )

            # Binary search the interval [left, right]
            it = 0
            while right - left > tol and it < max_iter:
                mid = (right + left) / 2.0
                p_value_mid = _compute_p_value(mid)
                if p_value_mid < alpha:
                    right = mid
                else:
                    left = mid
                it = it + 1
            upper_ci = (left + right) / 2.0
            conf_interval[post_period] = (lower_ci, upper_ci)
            if verbose:
                print(
                    f"Time-period: {post_period}, "
                    f"{100 * (1 - alpha)}% CI: [{round(lower_ci, 3)}, {round(upper_ci, 3)}]"
                )

        df_ci = pd.DataFrame.from_dict(
            conf_interval, orient="index", columns=["lower_ci", "upper_ci"]
        )
        df_ci = pd.concat([gaps.loc[post_periods].rename("value"), df_ci], axis=1)
        df_ci.index.name = "time"
        return df_ci
