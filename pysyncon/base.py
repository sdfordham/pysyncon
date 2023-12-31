from __future__ import annotations
from typing import Optional, Literal
from abc import ABCMeta, abstractmethod

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds, LinearConstraint

from .dataprep import Dataprep, IsinArg_t


class BaseSynth(metaclass=ABCMeta):
    """Metaclass that defines methods common to synthetic control methods."""

    def __init__(self) -> None:
        self.dataprep: Optional[Dataprep] = None
        self.W: Optional[np.ndarray] = None

    @abstractmethod
    def fit(*args, **kwargs) -> None:
        raise NotImplemented

    def _synthetic(self, time_period: Optional[IsinArg_t] = None) -> pd.Series:
        """Assemble the synthetic unit using the calculated weight matrix.

        Parameters
        ----------
        time_period : Iterable | pandas.Series | dict, optional
            Time range to plot, if none is supplied then the time range used
            is the time period over which the optimisation happens, by default
            None

        Returns
        -------
        pd.Series
            Time series of the synthetic unit.
        """
        Z0, _ = self.dataprep.make_outcome_mats(time_period=time_period)
        ts_synthetic = (Z0 * self.W).sum(axis=1)
        return ts_synthetic

    def path_plot(
        self,
        time_period: Optional[IsinArg_t] = None,
        treatment_time: Optional[int] = None,
        grid: bool = True,
    ) -> None:
        """Plot the outcome variable over time for the treated unit and the
        synthetic control. The fit method needs to be run with a :class:`Dataprep`
        object for this method to be available.

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

        Raises
        ------
        ValueError
            If there is no weight matrix available
        ValueError
            If there is no :class:`Dataprep` object set
        """
        if self.dataprep is None:
            raise ValueError("dataprep must be set for automatic plots.")
        if self.W is None:
            raise ValueError("No weight matrix available; fit data first.")

        Z0, Z1 = self.dataprep.make_outcome_mats(time_period=time_period)
        ts_synthetic = self._synthetic(time_period=time_period)

        plt.plot(Z1, color="black", linewidth=1, label=Z1.name)
        plt.plot(
            ts_synthetic,
            color="black",
            linewidth=1,
            linestyle="dashed",
            label="Synthetic",
        )
        plt.ylabel(self.dataprep.dependent)
        if treatment_time:
            plt.axvline(x=treatment_time, ymin=0.05, ymax=0.95, linestyle="dashed")
        plt.legend()
        plt.grid(grid)
        plt.show()

    def _gaps(self, time_period: Optional[IsinArg_t] = None) -> pd.Series:
        """Calculate the gaps (difference between factual
        and estimated counterfactual)

        Parameters
        ----------
        time_period : Iterable | pandas.Series | dict, optional
            Time range to plot, if none is supplied then the time range used
            is the time period over which the optimisation happens, default
            None

        Returns
        -------
        pd.Series
            Series containing the gaps

        :meta private:
        """
        _, Z1 = self.dataprep.make_outcome_mats(time_period=time_period)
        ts_synthetic = self._synthetic(time_period=time_period)
        ts_gap = Z1 - ts_synthetic
        return ts_gap

    def gaps_plot(
        self,
        time_period: Optional[IsinArg_t] = None,
        treatment_time: Optional[int] = None,
        grid: bool = True,
    ) -> None:
        """Plots the gap between the treated unit and the synthetic unit over
        time. The fit method needs to be run with a Dataprep
        object for this method to be available.

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

        Raises
        ------
        ValueError
            If there is no :class:`Dataprep` object set
        ValueError
            If there is no weight matrix available
        """
        if self.dataprep is None:
            raise ValueError("dataprep must be set for automatic plots.")
        if self.W is None:
            raise ValueError("No weight matrix available; fit data first.")

        ts_gap = self._gaps(time_period=time_period)
        plt.plot(ts_gap, color="black", linewidth=1)
        plt.ylabel(self.dataprep.dependent)
        plt.hlines(
            y=0,
            xmin=min(ts_gap.index),
            xmax=max(ts_gap.index),
            color="black",
            linestyle="dashed",
        )
        if treatment_time:
            plt.axvline(x=treatment_time, ymin=0.05, ymax=0.95, linestyle="dashed")
        plt.grid(grid)
        plt.show()

    def weights(self, round: int = 3, threshold: Optional[float] = None) -> pd.Series:
        """Return a ``pandas.Series`` of the weights for each control unit.

        Parameters
        ----------
        round : int, optional
            Round the weights to given number of places, by default 3
        threshold : float, optional
            If supplied, will only show weights above this value, by default
            None

        Returns
        -------
        pandas.Series
            The weights computed

        Raises
        ------
        ValueError
            If there is no weight matrix available
        """
        if self.W is None:
            raise ValueError("No weight matrix available; fit data first.")
        if self.dataprep is None:
            weights_ser = pd.Series(self.W, name="weights")
        else:
            weights_ser = pd.Series(
                self.W, index=list(self.dataprep.controls_identifier), name="weights"
            )
        weights_ser = (
            weights_ser[weights_ser >= threshold] if threshold else weights_ser
        )
        return weights_ser.round(round)

    def summary(self, round: int = 3) -> pd.DataFrame:
        """Generates a ``pandas.DataFrame`` with summary data. The
        first column will show the mean value of each predictor over the time
        period ``time_predictors_prior`` for the treated unit and the second
        column the case of the synthetic unit and finally there will be a
        column 'sample mean' that shows the mean value of each predictor
        over the time period ``time_predictors_prior`` across all the control
        units, i.e. this will be the same as a synthetic control where all
        the weights are equal.

        Parameters
        ----------
        round : int, optional
            Round the table values to the given number of places, by
            default 3

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
        """
        if self.dataprep is None:
            raise ValueError("dataprep must be set for summary.")
        if self.W is None:
            raise ValueError("No weight matrix available: fit data first.")
        X0, X1 = self.dataprep.make_covariate_mats()

        treated = X1.rename("treated")
        synthetic = (X0 * self.W).sum(axis=1).rename("synthetic")
        sample_mean = X0.mean(axis=1).rename("sample mean")

        return pd.concat([treated, synthetic, sample_mean], axis=1).round(round)

    def att(self, time_period: IsinArg_t) -> dict[str, float]:
        """Computes the average treatment effect on the treated unit (ATT) and
        the standard error to the value over the chosen time-period.

        Parameters
        ----------
        time_period : Iterable | pandas.Series | dict, optional
            Time period to compute the ATT over.

        Returns
        -------
        dict
            A dictionary with the ATT value and the standard error to the ATT.

        Raises
        ------
        ValueError
            If there is no weight matrix available
        """
        if self.W is None:
            raise ValueError("No weight matrix available; fit data first.")
        gaps = self._gaps(time_period=time_period)

        att = np.mean(gaps)
        se = np.std(gaps, ddof=1) / np.sqrt(len(time_period))

        return {"att": att.item(), "se": se.item()}


class VanillaOptimMixin:
    @staticmethod
    def w_optimize(
        V_mat: np.ndarray,
        X0: np.ndarray,
        X1: np.ndarray,
        qp_method: Literal["SLSQP"] = "SLSQP",
        qp_options: dict = {"maxiter": 1000},
    ) -> tuple[np.ndarray, float]:
        """Solves the inner part of the quadratic minimization problem for a
        given V matrix.

        Parameters
        ----------
        V_mat : numpy.ndarray, shape (c, c)
            V matrix using the notation of the Abadie, Diamond & Hainmueller
            paper defining.
        X0 : numpy.ndarray, shape (m, c)
            Matrix with each column corresponding to a control unit and each
            row is covariates.
        X1 : numpy.ndarray, shape (m,)
            Column vector giving the covariate values for the treated unit.
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

        P = X0.T @ V_mat @ X0
        q = -1.0 * X1.T @ V_mat @ X0

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
