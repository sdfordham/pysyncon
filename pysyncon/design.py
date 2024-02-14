from __future__ import annotations
from itertools import combinations
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize, Bounds, LinearConstraint
from scipy.linalg import block_diag
import matplotlib.pyplot as plt

from .dataprep import Dataprep, IsinArg_t


class ExperimentalDesign:
    """Implementation of the Abadie & Zhao's Experimental
    Controls for Experimental Design"""

    def __init__(self) -> None:
        super().__init__()
        self.res: Optional[dict[str, Union[float, np.ndarray]]] = None
        self.loss: Optional[float] = None

    def fit(self, dataprep: Optional[Dataprep] = None) -> None:
        """Fit the model/calculate the weights.

        Parameters
        ----------
        dataprep : Dataprep, optional
            :class:`Dataprep` object containing data to model, by default None.

        Returns
        -------
        NoneType
            None
        """
        self.dataprep = dataprep

        X = pd.concat(dataprep.make_covariate_mats(), axis=1)
        X = X.divide(X.std(axis=1), axis=0)
        f = np.full(X.shape[1], 1.0)
        X_bar = X @ f

        results = list()
        for sz_v in range(2, len(dataprep.treatment_identifier) + 1):
            x0_v = np.full(sz_v, 1 / sz_v)

            for combination_v in combinations(dataprep.treatment_identifier, sz_v):
                for sz_w in range(2, len(dataprep.controls_identifier) + 1):
                    e_1 = np.hstack([np.full(sz_w, 1), np.full(sz_v, 0)])
                    e_2 = np.hstack([np.full(sz_w, 0), np.full(sz_v, 1)])

                    constraints = [
                        LinearConstraint(A=e_1.T, lb=1.0, ub=1.0),
                        LinearConstraint(A=e_2.T, lb=1.0, ub=1.0),
                    ]
                    bounds = Bounds(
                        lb=np.full(sz_w + sz_v, 0.0), ub=np.full(sz_w + sz_v, 1.0)
                    )

                    x0_w = np.full(sz_w, 1 / sz_w)
                    x0 = np.hstack([x0_w, x0_v])

                    for combination_w in combinations(
                        dataprep.controls_identifier, sz_w
                    ):
                        X_w = X.loc[:, combination_w]
                        X_v = X.loc[:, combination_v]

                        P = block_diag(X_w.T @ X_w, X_v.T @ X_v)
                        q = np.hstack([X_bar.T @ X_w, X_bar @ X_v])

                        def fun(x):
                            return x.T @ P @ x - 2.0 * q.T @ x

                        res = minimize(
                            fun=fun,
                            x0=x0,
                            bounds=bounds,
                            constraints=constraints,
                            method="SLSQP",
                        )
                        results.append(
                            {
                                "fun": res["fun"],
                                "w_units": combination_w,
                                "w_weights": res["x"][:sz_w].round(3),
                                "v_units": combination_v,
                                "v_weights": res["x"][sz_w:].round(3),
                            }
                        )

        min_, min_res = None, None
        for r in results:
            treated_unit = X.loc[:, r["v_units"]] @ r["v_weights"]
            control_unit = X.loc[:, r["w_units"]] @ r["w_weights"]
            ss = np.sqrt((treated_unit - control_unit).pow(2).sum())
            if min_ is None or ss < min_:
                min_, min_res = ss, r

        self.W = min_res["w_weights"]
        self.res, self.loss = min_res, min_

    @property
    def synthetic_treated(self) -> pd.Series:
        """Returns the weights associated with the units that
        make up the synthetic treated unit. 

        Returns
        -------
        pd.Series
            A pandas Series with the weights associated with
            the units that make up the synthetic treated unit

        Raises
        ------
        ValueError
            If there is no weights available.
        """
        if not self.res:
            raise ValueError("No results available; fit data first.")
        return pd.Series(data=self.res["v_weights"], index=self.res["v_units"]).rename(
            "Synthetic treated"
        )

    @property
    def synthetic_control(self) -> pd.Series:
        """Returns the weights associated with the units that
        make up the synthetic control unit. 

        Returns
        -------
        pd.Series
            A pandas Series with the weights associated with
            the units that make up the synthetic control unit

        Raises
        ------
        ValueError
            If there is no weights available.
        """
        if not self.res:
            raise ValueError("No results available; fit data first.")
        return pd.Series(data=self.res["w_weights"], index=self.res["w_units"]).rename(
            "Synthetic control"
        )

    def summary(self) -> pd.DataFrame:
        """Returns a pandas DataFrame showing the covariate values
        over the minimization period for the synthetic treated unit,
        the synthetic control unit and the sample mean of all the
        units.

        Returns
        -------
        pd.DataFrame
        """
        X = pd.concat(self.dataprep.make_covariate_mats(), axis=1)
        return pd.concat(
            [
                X.loc[:, self.res["v_units"]] @ self.res["v_weights"],
                X.loc[:, self.res["w_units"]] @ self.res["w_weights"],
                X.mean(axis=1),
            ],
            axis=1,
        ).rename(
            columns={
                0: "synthetic treated",
                1: "synthetic control",
                2: "sample mean",
            }
        )

    def path_plot(
        self,
        time_period: Optional[IsinArg_t] = None,
        treatment_time: Optional[int] = None,
        grid: bool = True,
    ) -> None:
        """Plot the outcome variable over time for the treated unit and the
        synthetic control.

        Parameters
        ----------
        time_period : Iterable | pandas.Series | dict, optional
            Time range to plot, if none is supplied then the time range used
            is the time period over which the optimisation happens, by default
            None
        treatment_time : int, optional
            If supplied, plot a vertical line at the time period that the
            treatment time occurred, by default None
        grid : bool, optional
            Whether or not to plot a grid, by default True
        """
        Z = pd.concat(self.dataprep.make_outcome_mats(time_period=time_period), axis=1)

        treated_unit = Z.loc[:, self.res["v_units"]] @ self.res["v_weights"]
        control_unit = Z.loc[:, self.res["w_units"]] @ self.res["w_weights"]

        plt.plot(treated_unit, color="black", linewidth=1, label="Synthetic treated")
        plt.plot(
            control_unit,
            color="black",
            linewidth=1,
            linestyle="dashed",
            label="Synthetic control",
        )
        plt.ylabel(self.dataprep.dependent)
        if treatment_time:
            plt.axvline(x=treatment_time, ymin=0.05, ymax=0.95, linestyle="dashed")
        plt.legend()
        plt.grid(grid)
        plt.show()
