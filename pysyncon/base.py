from __future__ import annotations
from typing import Optional, Literal, Sequence
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
        self.W_names: Optional[Sequence] = None

    @abstractmethod
    def fit(*args, **kwargs) -> None:
        raise NotImplementedError

    def _synthetic(self, Z0: pd.DataFrame) -> pd.Series:
        """Assemble the synthetic unit using the calculated weight matrix.

        Parameters
        ----------
        Z0 : pandas.DataFrame, shape (n, c)
            A matrix of the time series of the outcome variable with each
            column corresponding to a control unit and the rows are the time
            steps

        Returns
        -------
        pd.Series
            Time series of the synthetic unit.
        """
        ts_synthetic = (Z0 * self.W).sum(axis=1)
        return ts_synthetic

    def path_plot(
        self,
        time_period: Optional[IsinArg_t] = None,
        treatment_time: Optional[int] = None,
        grid: bool = True,
        Z0: Optional[pd.DataFrame] = None,
        Z1: Optional[pd.Series] = None,
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
        Z0 : pandas.DataFrame, shape (n, c), optional
            The matrix of the time series of the outcome variable for the control units.
            If no dataprep is set, then this must be supplied along with Z1, by default None.
        Z1 : pandas.Series, shape (n, 1), optional
            The matrix of the time series of the outcome variable for the treated unit.
            If no dataprep is set, then this must be supplied along with Z0, by default None.

        Raises
        ------
        ValueError
            If there is no weight matrix available
        ValueError
            If there is no :class:`Dataprep` object set or (Z0, Z1) is not supplied
        """
        if self.dataprep is not None:
            Z0, Z1 = self.dataprep.make_outcome_mats(time_period=time_period)
        elif Z0 is None or Z1 is None:
            raise ValueError("dataprep must be set or (Z0, Z1) must be set for plots.")
        if self.W is None:
            raise ValueError("No weight matrix available; fit data first.")

        ts_synthetic = self._synthetic(Z0=Z0)
        plt.plot(Z1, color="black", linewidth=1, label=Z1.name)
        plt.plot(
            ts_synthetic,
            color="black",
            linewidth=1,
            linestyle="dashed",
            label="Synthetic",
        )
        if self.dataprep is not None:
            plt.ylabel(self.dataprep.dependent)
        if treatment_time:
            plt.axvline(x=treatment_time, ymin=0.05, ymax=0.95, linestyle="dashed")
        plt.legend()
        plt.grid(grid)
        plt.show()

    def _gaps(self, Z0: pd.DataFrame, Z1: pd.Series) -> pd.Series:
        """Calculate the gaps (difference between factual
        and estimated counterfactual)

        Parameters
        ----------
        Z0 : pandas.DataFrame, shape (n, c)
            A matrix of the time series of the outcome variable with each
            column corresponding to a control unit and the rows are the time
            steps
        Z1 : pandas.DataFrame, shape (n, 1)
            A matrix of the time series of the outcome variable for the treated
            unit and the rows are the time steps

        Returns
        -------
        pd.Series
            Series containing the gaps

        :meta private:
        """
        ts_synthetic = self._synthetic(Z0=Z0)
        ts_gap = Z1 - ts_synthetic
        return ts_gap

    def gaps_plot(
        self,
        time_period: Optional[IsinArg_t] = None,
        treatment_time: Optional[int] = None,
        grid: bool = True,
        Z0: Optional[pd.DataFrame] = None,
        Z1: Optional[pd.Series] = None,
    ) -> None:
        """Plots the gap between the treated unit and the synthetic unit over
        time.

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
        Z0 : pandas.DataFrame, shape (n, c), optional
            The matrix of the time series of the outcome variable for the control units.
            If no dataprep is set, then this must be supplied along with Z1, by default None.
        Z1 : pandas.Series, shape (n, 1), optional
            The matrix of the time series of the outcome variable for the treated unit.
            If no dataprep is set, then this must be supplied along with Z0, by default None.

        Raises
        ------
        ValueError
            If there is no weight matrix available
        ValueError
            If there is no :class:`Dataprep` object set or (Z0, Z1) is not supplied
        """
        if self.dataprep is not None:
            Z0, Z1 = self.dataprep.make_outcome_mats(time_period=time_period)
        elif Z0 is None or Z1 is None:
            raise ValueError("dataprep must be set or (Z0, Z1) must be set for plots.")
        if self.W is None:
            raise ValueError("No weight matrix available; fit data first.")

        ts_gap = self._gaps(Z0=Z0, Z1=Z1)
        plt.plot(ts_gap, color="black", linewidth=1)
        if self.dataprep is not None:
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
            weights_ser = pd.Series(self.W, index=self.W_names, name="weights")
        else:
            weights_ser = pd.Series(
                self.W, index=list(self.dataprep.controls_identifier), name="weights"
            )
        weights_ser = (
            weights_ser[weights_ser >= threshold] if threshold else weights_ser
        )
        return weights_ser.round(round)

    def summary(
        self,
        round: int = 3,
        X0: Optional[pd.DataFrame] = None,
        X1: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
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
            If there is no weight matrix available
        ValueError
            If there is no :class:`Dataprep` object set or (Z0, Z1) is not supplied
        """
        if self.W is None:
            raise ValueError("No weight matrix available: fit data first.")
        if self.dataprep is not None:
            X0, X1 = self.dataprep.make_covariate_mats()
        elif X0 is None or X1 is None:
            raise ValueError(
                "dataprep must be set or (X0, X1) must be set for summary."
            )

        treated = X1.rename("treated")
        synthetic = (X0 * self.W).sum(axis=1).rename("synthetic")
        sample_mean = X0.mean(axis=1).rename("sample mean")

        return pd.concat([treated, synthetic, sample_mean], axis=1).round(round)

    def att(
        self,
        time_period: IsinArg_t,
        Z0: Optional[pd.DataFrame] = None,
        Z1: Optional[pd.Series] = None,
    ) -> dict[str, float]:
        """Computes the average treatment effect on the treated unit (ATT) and
        the standard error to the value over the chosen time-period.

        Parameters
        ----------
        time_period : Iterable | pandas.Series | dict, optional
            Time period to compute the ATT over.
        Z0 : pandas.DataFrame, shape (n, c), optional
            The matrix of the time series of the outcome variable for the control units.
            If no dataprep is set, then this must be supplied along with Z1, by default None.
        Z1 : pandas.Series, shape (n, 1), optional
            The matrix of the time series of the outcome variable for the treated unit.
            If no dataprep is set, then this must be supplied along with Z0, by default None.

        Returns
        -------
        dict
            A dictionary with the ATT value and the standard error to the ATT.

        Raises
        ------
        ValueError
            If there is no weight matrix available
        ValueError
            If there is no :class:`Dataprep` object set or (Z0, Z1) is not supplied
        """
        if self.W is None:
            raise ValueError("No weight matrix available; fit data first.")
        if self.dataprep is not None:
            Z0, Z1 = self.dataprep.make_outcome_mats(time_period=time_period)
            gaps = self._gaps(Z0=Z0, Z1=Z1)
        if Z0 is not None and Z1 is not None:
            gaps = self._gaps(Z0=Z0.loc[time_period, :], Z1=Z1.loc[time_period])
        else:
            raise ValueError("dataprep must be set or (Z0, Z1) must be set for att.")
        att = np.mean(gaps)
        se = np.std(gaps, ddof=1) / np.sqrt(len(time_period))

        return {"att": att.item(), "se": se.item()}

    def mspe(
        self, Z0: Optional[pd.DataFrame] = None, Z1: Optional[pd.Series] = None
    ) -> float:
        """Returns the mean square prediction error in the fit of
        the synthetic control versus the treated unit over the
        optimization time-period.

        Parameters
        ----------
        Z0 : pandas.DataFrame, shape (n, c), optional
            The matrix of the time series of the outcome variable for the control units.
            If no dataprep is set, then this must be supplied along with Z1, by default None.
        Z1 : pandas.Series, shape (n, 1), optional
            The matrix of the time series of the outcome variable for the treated unit.
            If no dataprep is set, then this must be supplied along with Z0, by default None.

        Returns
        -------
        float
            Mean square prediction Error

        Raises
        ------
        ValueError
            If the fit method has not been run (no weights available.)
        ValueError
            If there is no :class:`Dataprep` object set or (Z0, Z1) is not supplied
        """
        if self.W is None:
            raise ValueError("No weight matrix available; fit data first.")
        if self.dataprep is not None:
            Z0, Z1 = self.dataprep.make_outcome_mats(
                time_period=self.dataprep.time_optimize_ssr
            )
        if Z0 is None or Z1 is None:
            raise ValueError("dataprep must be set or (Z0, Z1) must be set for plots.")
        ts_synthetic = self._synthetic(Z0=Z0)

        n = len(ts_synthetic)
        return (1 / n) * (Z1 - ts_synthetic).pow(2).sum().item()

    def mape(
        self, Z0: Optional[pd.DataFrame] = None, Z1: Optional[pd.Series] = None
    ) -> float:
        """Returns the mean absolute percentage error in the fit of
        the synthetic control versus the treated unit over the
        optimization time-period.

        Parameters
        ----------
        Z0 : pandas.DataFrame, shape (n, c), optional
            The matrix of the time series of the outcome variable for the control units.
            If no dataprep is set, then this must be supplied along with Z1, by default None.
        Z1 : pandas.Series, shape (n, 1), optional
            The matrix of the time series of the outcome variable for the treated unit.
            If no dataprep is set, then this must be supplied along with Z0, by default None.

        Returns
        -------
        float
            Mean absolute percentage error

        Raises
        ------
        ValueError
            If the fit method has not been run (no weights available.)
        ValueError
            If there is no :class:`Dataprep` object set or (Z0, Z1) is not supplied
        """
        if self.W is None:
            raise ValueError("No weight matrix available; fit data first.")
        if self.dataprep is not None:
            Z0, Z1 = self.dataprep.make_outcome_mats(
                time_period=self.dataprep.time_optimize_ssr
            )
        if Z0 is None or Z1 is None:
            raise ValueError("dataprep must be set or (Z0, Z1) must be set for plots.")
        ts_synthetic = self._synthetic(Z0=Z0)

        n = len(ts_synthetic)
        return (1 / n) * ((Z1 - ts_synthetic) / Z1).abs().sum().item()

    def mae(
        self, Z0: Optional[pd.DataFrame] = None, Z1: Optional[pd.Series] = None
    ) -> float:
        """Returns the mean absolute error in the fit of
        the synthetic control versus the treated unit over the
        optimization time-period.

        Parameters
        ----------
        Z0 : pandas.DataFrame, shape (n, c), optional
            The matrix of the time series of the outcome variable for the control units.
            If no dataprep is set, then this must be supplied along with Z1, by default None.
        Z1 : pandas.Series, shape (n, 1), optional
            The matrix of the time series of the outcome variable for the treated unit.
            If no dataprep is set, then this must be supplied along with Z0, by default None.

        Returns
        -------
        float
            Mean absolute error

        Raises
        ------
        ValueError
            If the fit method has not been run (no weights available.)
        ValueError
            If there is no :class:`Dataprep` object set or (Z0, Z1) is not supplied
        """
        if self.W is None:
            raise ValueError("No weight matrix available; fit data first.")
        if self.dataprep is not None:
            Z0, Z1 = self.dataprep.make_outcome_mats(
                time_period=self.dataprep.time_optimize_ssr
            )
        if Z0 is None or Z1 is None:
            raise ValueError("dataprep must be set or (Z0, Z1) must be set for plots.")
        ts_synthetic = self._synthetic(Z0=Z0)

        n = len(ts_synthetic)
        return (1 / n) * (Z1 - ts_synthetic).abs().sum().item()


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
        given V matrix (see Abadie and Gardeazabal :cite:`basque2003`).

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
        q = X1.T @ V_mat @ X0

        def fun(x):
            return 0.5 * x.T @ P @ x - q.T @ x

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
