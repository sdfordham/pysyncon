from __future__ import annotations
from typing import Optional, Union
from concurrent import futures
import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .dataprep import Dataprep, IsinArg_t
from .base import BaseSynth


class HoldoutSplitter:
    """Iterator that prepares the time series for cross-validation by
    progressively removing blocks of length `holdout_len`.
    """

    def __init__(self, df: pd.DataFrame, ser: pd.Series, holdout_len: int = 1):
        """Iterator that prepares the time series for cross-validation by
        progressively removing blocks of length `holdout_len`.

        Parameters
        ----------
        df : pandas.DataFrame, shape (r, c)
            Dataframe that will be split for the cross-validation.
        ser : pandas.Series, shape (r, 1)
            Series that will split for the cross-validation.
        holdout_len : int, optional
            Number of days to remove in each iteration, by default 1.

        Raises
        ------
        ValueError
            if df and ser do not have the same number of rows.
        ValueError
            if `holdout_len` is not >= 1.
        ValueError
            if `holdout_len` is larger than the number of rows of df.
        """
        if df.shape[0] != ser.shape[0]:
            raise ValueError("`df` and `ser` must have the same number of rows.")
        if holdout_len < 1:
            raise ValueError("`holdout_len` must be at least 1.")
        if holdout_len >= df.shape[0]:
            raise ValueError("`holdout_len` must be less than df.shape[0]")
        self.df = df
        self.ser = ser
        self.holdout_len = holdout_len
        self.idx = 0

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        if (self.idx + self.holdout_len) > self.df.shape[0]:
            raise StopIteration
        holdout = slice(self.idx, self.idx + self.holdout_len)

        df_holdout = self.df.iloc[holdout,]  # fmt: skip
        ser_holdout = self.ser.iloc[holdout]

        df = self.df.drop(index=self.df.index[holdout])
        ser = self.ser.drop(index=self.ser.index[holdout])

        self.idx += 1
        return df, df_holdout, ser, ser_holdout


@dataclass
class CrossValidationResult:
    """Convenience class for holding the results of the cross-validation
    procedure from the AugSynth.
    """

    lambdas: np.ndarray
    errors_mean: np.ndarray
    errors_se: np.ndarray

    def best_lambda(self, min_1se: bool = True) -> float:
        """Return the best lambda.

        Parameters
        ----------
        min_1se : bool, optional
            return the largest lambda within 1 standard error of the minimum
            , by default True

        Returns
        -------
        float
        """
        if min_1se:
            return (
                self.lambdas[
                    self.errors_mean
                    <= self.errors_mean.min()
                    + self.errors_se[self.errors_mean.argmin()]
                ]
                .max()
                .item()
            )
        return self.lambdas[self.errors_mean.argmin()].item()

    def plot(self) -> None:
        """Plots the mean errors against the lambda values with the standard
        errors as error bars.
        """
        plt.errorbar(
            x=self.lambdas,
            y=self.errors_mean,
            yerr=self.errors_se,
            ecolor="black",
            capsize=2,
        )
        plt.xlabel("Lambda")
        plt.ylabel("Mean error")
        plt.xscale("log")
        plt.yscale("log")
        plt.title("Cross validation result")
        plt.grid()
        plt.show()


class PlaceboTest:
    """Class that carries out placebo tests by running a synthetic control
    study using each possible control unit as the treated unit and the
    remaining control units as controls. See :cite:`germany2015` for more details.
    """

    def __init__(self) -> None:
        self.paths: Optional[pd.DataFrame] = None
        self.treated_path: Optional[pd.DataFrame] = None
        self.gaps: Optional[pd.DataFrame] = None
        self.treated_gap: Optional[pd.DataFrame] = None
        self.time_optimize_ssr: Optional[IsinArg_t] = None

    def fit(
        self,
        dataprep: Dataprep,
        scm: BaseSynth,
        scm_options: dict = {},
        max_workers: Optional[int] = None,
        verbose: bool = True,
    ):
        """Run the placebo tests. This method is multi-process and by default
        will use all available processors. Use the `max_workers` option to change
        this behaviour.

        Parameters
        ----------
        dataprep : Dataprep
            :class:`Dataprep` object containing data to model, by default None.
        scm : Synth | AugSynth
            Synthetic control study to use
        scm_options : dict, optional
            Options to provide to the fit method of the synthetic control
            study, valid options are any valid option that the `scm_type`
            takes, by default {}
        max_workers : Optional[int], optional
            Maximum number of processes to use, if not provided then will use
            all available, by default None
        verbose : bool, optional
            Whether or not to output progress, by default True
        """
        paths, gaps = list(), list()
        n_tests = len(dataprep.controls_identifier)
        with futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            to_do = list()
            for treated, controls in self.placebo_iter(dataprep.controls_identifier):
                _dataprep = copy.copy(dataprep)
                _dataprep.treatment_identifier = treated
                _dataprep.controls_identifier = controls
                to_do.append(
                    executor.submit(
                        self._single_placebo,
                        dataprep=_dataprep,
                        scm=scm,
                        scm_options=scm_options,
                    )
                )
            for idx, future in enumerate(futures.as_completed(to_do), 1):
                path, gap = future.result()
                if verbose:
                    print(f"({idx}/{n_tests}) Completed placebo test for {path.name}.")
                paths.append(path)
                gaps.append(gap)

        self.paths = pd.concat(paths, axis=1)
        self.gaps = pd.concat(gaps, axis=1)
        self.time_optimize_ssr = dataprep.time_optimize_ssr

        print(f"Calculating treated unit gaps.")
        self.treated_path, self.treated_gap = self._single_placebo(
            dataprep=dataprep, scm=scm, scm_options=scm_options
        )
        print("Done.")

    @staticmethod
    def placebo_iter(controls: list[str]) -> tuple[str, list[str]]:
        """Generates combinations of (treated unit, control units) for the
        placebo tests.

        Parameters
        ----------
        controls : list[str]
            List of unit labels to use

        Yields
        ------
        tuple[str, list[str]]
            Tuple of (treated unit label, control unit labels)

        :meta private:
        """
        for control in controls:
            yield (control, [c for c in controls if c != control])

    @staticmethod
    def _single_placebo(
        dataprep: Dataprep, scm: BaseSynth, scm_options: dict = {}
    ) -> tuple[pd.Series, pd.Series]:
        """Run a single placebo test.

        Parameters
        ----------
        dataprep : Dataprep
            :class:`Dataprep` object containing data to model
        scm : Synth | AugSynth
            Type of synthetic control study to use
        scm_options : dict, optional
            Options to provide to the fit method of the synthetic control
            study, valid options are any valid option that `scm` takes, by
            default {}

        Returns
        -------
        tuple[pandas.Series, pandas.Series]
            A time-series of the path of the synthetic control and a
            time-series of the gap between the treated unit and the synthetic
            control.

        :meta private:
        """
        scm.fit(dataprep=dataprep, **scm_options)

        Z0, Z1 = dataprep.make_outcome_mats(
            time_period=dataprep.foo[dataprep.time_variable]
        )
        synthetic = scm._synthetic(Z0=Z0)
        gaps = scm._gaps(Z0=Z0, Z1=Z1)
        return synthetic.rename(dataprep.treatment_identifier), gaps.rename(
            dataprep.treatment_identifier
        )

    def gaps_plot(
        self,
        time_period: Optional[IsinArg_t] = None,
        grid: bool = True,
        treatment_time: Optional[int] = None,
        mspe_threshold: Optional[float] = None,
        exclude_units: Optional[list] = None,
    ):
        """Plot the gaps between the treated unit and the synthetic control
        for each placebo test.

        Parameters
        ----------
        time_period : Iterable | pandas.Series | dict, optional
            Time range to plot, if none is supplied then the time range used
            is the time period over which the optimisation happens, by default
            None
        grid : bool, optional
            Whether or not to plot a grid, by default True
        treatment_time : int, optional
            If supplied, plot a vertical line at the time period that the
            treatment time occurred, by default None
        mspe_threshold : float, optional
            Remove any non-treated units whose MSPE pre-treatment is :math:`>`
            mspe_threshold :math:`\\times` the MSPE of the treated unit pre-treatment.
            This serves to exclude any non-treated units whose synthetic control
            had a poor pre-treatment match to the actual relative to how the
            actual treated unit matched pre-treatment.

        Raises
        ------
        ValueError
            if no placebo test has been run yet
        ValueError
            if `mspe_threshold` is supplied but `treatment_year` is not.
        """
        if self.gaps is None:
            raise ValueError("No gaps available; run a placebo test first.")
        time_period = time_period if time_period is not None else self.time_optimize_ssr

        gaps = self.gaps.drop(columns=exclude_units) if exclude_units else self.gaps

        if mspe_threshold:
            if not treatment_time:
                raise ValueError("Need `treatment_time` to use `mspe_threshold`.")
            pre_mspe = gaps.loc[:treatment_time].pow(2).sum(axis=0)
            pre_mspe_treated = self.treated_gap.loc[:treatment_time].pow(2).sum(axis=0)
            keep = pre_mspe[pre_mspe < mspe_threshold * pre_mspe_treated].index
            placebo_gaps = gaps[gaps.index.isin(time_period)][keep]
        else:
            placebo_gaps = gaps[gaps.index.isin(time_period)]

        plt.plot(placebo_gaps, color="black", alpha=0.1)
        plt.plot(self.treated_gap, color="black", alpha=1.0)
        if treatment_time:
            plt.axvline(x=treatment_time, ymin=0.05, ymax=0.95, linestyle="dashed")
        plt.grid(grid)
        plt.show()

    def pvalue(self, treatment_time: int) -> float:
        """Calculate p-value of Abadie et al's version of Fisher's
        exact hypothesis test for no effect of treatment null, see also
        section 2.2. of :cite:`fp2018`.

        Parameters
        ----------
        treatment_time : int
            The time period that the treatment time occurred

        Returns
        -------
        float
            p-value for null hypothesis of no effect of treatment

        Raises
        ------
        ValueError
            if no placebo test has been run yet
        """
        if self.gaps is None or self.treated_gap is None:
            raise ValueError("Run a placebo test first.")

        all_ = pd.concat([self.gaps, self.treated_gap], axis=1)

        denom = all_.loc[:treatment_time].pow(2).sum(axis=0)
        num = all_.loc[treatment_time:].pow(2).sum(axis=0)

        t, _ = self.gaps.shape
        t0, _ = self.gaps.loc[:treatment_time].shape

        rmspe = (num / (t - t0)) / (denom / t0)
        return sum(
            rmspe.drop(index=self.treated_gap.name) >= rmspe.loc[self.treated_gap.name]
        ) / len(rmspe)
