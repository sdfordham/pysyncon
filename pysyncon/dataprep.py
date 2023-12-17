from __future__ import annotations
from typing import Any, Iterable, Union, Optional, Literal, Sequence, Mapping, Tuple

import pandas as pd
from pandas._typing import Axes


PredictorsOp_t = Literal["mean", "std", "median"]
IsinArg_t = Union[Iterable, pd.Series, dict]
SpecialPredictor_t = Tuple[
    Any, Union[pd.Series, pd.DataFrame, Sequence, Mapping], PredictorsOp_t
]


class Dataprep:
    """Helper class that takes in the panel data and all necessary information
    needed to describe the study setup. It is used to automatically generate
    the matrices needed for the optimisation methods, plots of the results etc.

    Parameters
    ----------
    foo : pandas.DataFrame
        A pandas DataFrame containing the panel data where the columns are
        predictor/outcome variables and each row is a time-step for some unit
    predictors : Axes
        The columns of ``foo`` to use as predictors
    predictors_op : "mean" | "std" | "median"
        The statistical operation to use on the predictors - the time range that
        the operation is applied to is ``time_predictors_prior``
    dependent : Any
        The column of ``foo`` to use as the dependent variable
    unit_variable : Any
        The column of ``foo`` that contains the unit labels
    time_variable : Any
        The column of ``foo`` that contains the time period
    treatment_identifier : Any
        The unit label that denotes the treated unit
    controls_identifier : Iterable
        The unit labels denoting the control units
    time_predictors_prior : Iterable
        The time range over which to apply the statistical operation to the
        predictors (see ``predictors_op`` argument)
    time_optimize_ssr : Iterable
        The time range over which the loss function should be minimised
    special_predictors : Iterable[SpecialPredictor_t], optional
        An iterable of special predictors which are additional predictors
        that should be averaged over a custom time period and an indicated
        statistical operator. In particular, a special predictor
        consists of a triple of:

            - ``column``: the column of ``foo`` containing the predictor to use,
            - ``time-range``: the time range to apply ``operator`` over - it should
              have the same type as ``time_predictors_prior`` or ``time_optimize_ssr``
            - ``operator``: the statistical operator to apply to ``column`` - it should
              have the same type as ``predictors_op``

        by default None

    Raises
    ------
    TypeError
        if ``foo`` is not of type ``pandas.DataFrame``
    ValueError
        if ``predictor`` is not a column of ``foo``
    ValueError
        if ``predictor_op`` is not one of "mean", "std", "median"
    ValueError
        if ``dependent`` is not a column of ``foo``
    ValueError
        if ``unit_variable`` is not a column of ``foo``
    ValueError
        if ``time_variable`` is not a column of ``foo``
    ValueError
        if ``treatment_identifier`` is not present in ``foo['unit_variable']``
    TypeError
        if ``controls_identifier`` is not of type ``Iterable``
    ValueError
        if ``treatment_identifier`` is in the list of controls
    ValueError
        if any of the controls is not in ``foo['unit_variable']``
    ValueError
        if any element of ``special_predictors`` is not an Iterable of length
        3
    ValueError
        if a predictor in an element of ``special_predictors`` is not a column
        of foo
    ValueError
        if one of the operators in an element of ``special_predictors`` is not
        one of "mean", "std", "median"
    """

    def __init__(
        self,
        foo: pd.DataFrame,
        predictors: Axes,
        predictors_op: PredictorsOp_t,
        dependent: Any,
        unit_variable: Any,
        time_variable: Any,
        treatment_identifier: Any,
        controls_identifier: Iterable,
        time_predictors_prior: IsinArg_t,
        time_optimize_ssr: IsinArg_t,
        special_predictors: Optional[Iterable[SpecialPredictor_t]] = None,
    ) -> None:
        if not isinstance(foo, pd.DataFrame):
            raise TypeError("foo must be pandas.DataFrame.")
        self.foo = foo

        for predictor in predictors:
            if predictor not in foo.columns:
                raise ValueError(f"predictor {predictor} not in foo columns.")
        self.predictors = predictors

        if predictors_op not in ("mean", "std", "median"):
            raise ValueError("predictors_op must be one of mean, std, median.")
        self.predictors_op = predictors_op

        if dependent not in foo.columns:
            raise ValueError(f"dependent {dependent} not in foo columns.")
        self.dependent = dependent

        if unit_variable not in foo.columns:
            raise ValueError(f"unit_variable {unit_variable} not in foo columns.")
        self.unit_variable = unit_variable

        if time_variable not in foo.columns:
            raise ValueError(f"time_variable {time_variable} not in foo columns.")
        self.time_variable = time_variable

        uniq_ident = foo[unit_variable].unique()
        if treatment_identifier not in uniq_ident:
            raise ValueError(
                f'treatment_identifier {treatment_identifier} not found in foo["{unit_variable}"].'
            )
        self.treatment_identifier = treatment_identifier

        if not isinstance(controls_identifier, Iterable):
            raise TypeError("controls_identifier should be an Iterable")
        for control in controls_identifier:
            if control == treatment_identifier:
                raise ValueError("treatment_identifier in controls_identifier.")
            if control not in uniq_ident:
                raise ValueError(
                    f'controls_identifier {control} not found in foo["{unit_variable}"].'
                )
        self.controls_identifier = controls_identifier

        self.time_predictors_prior = time_predictors_prior
        self.time_optimize_ssr = time_optimize_ssr

        if special_predictors:
            for el in special_predictors:
                if not isinstance(el, tuple) or len(el) != 3:
                    raise ValueError(
                        "Elements of special_predictors should be tuples of length 3."
                    )
                predictor, _, op = el
                if predictor not in foo.columns:
                    raise ValueError(
                        f"{predictor} in special_predictors not in foo columns."
                    )
                if op not in ("mean", "std", "median"):
                    raise ValueError(
                        f"{op} in special_predictors must be one of mean, std, median."
                    )
        self.special_predictors = special_predictors

    def make_covariate_mats(self) -> tuple[pd.DataFrame, pd.Series]:
        """Generate the covariate matrices to use as input to the fit method
        of the synthetic control computation.

        Returns
        -------
        tuple[pandas.DataFrame, pandas.Series]
            Returns the matrices X0, X1 (using the notation of the Abadie,
            Diamond & Hainmueller paper).

        Raises
        ------
        ValueError
            if predictors_op is not one of "mean", "std", "median"

        :meta private:
        """
        X_nonspecial = (
            self.foo[self.foo[self.time_variable].isin(self.time_predictors_prior)]
            .groupby(self.unit_variable)[self.predictors]
            .agg(self.predictors_op)
            .T
        )
        X1_nonspecial = X_nonspecial[self.treatment_identifier]
        X0_nonspecial = X_nonspecial[list(self.controls_identifier)]

        if self.special_predictors is None:
            return X0_nonspecial, X1_nonspecial

        X0_special = list()
        for control in self.controls_identifier:
            this_control = list()
            for predictor, time_period, op in self.special_predictors:
                mask = (self.foo[self.unit_variable] == control) & (
                    self.foo[self.time_variable].isin(time_period)
                )
                if op == "mean":
                    this_control.append(self.foo[mask][predictor].mean())
                elif op == "std":
                    this_control.append(self.foo[mask][predictor].std())
                elif op == "median":
                    this_control.append(self.foo[mask][predictor].median())
                else:
                    raise ValueError(f"Invalid predictors_op: {self.predictors_op}")
            X0_special.append(this_control)

        X0_special_columns = list()
        for idx, (predictor, _, _) in enumerate(self.special_predictors, 1):
            X0_special_columns.append(f"special.{idx}.{predictor}")

        X0_special = pd.DataFrame(
            X0_special, columns=X0_special_columns, index=self.controls_identifier
        ).T
        X0 = pd.concat([X0_nonspecial, X0_special], axis=0)

        X1_special = list()
        for predictor, time_period, op in self.special_predictors:
            mask = (self.foo[self.unit_variable] == self.treatment_identifier) & (
                self.foo[self.time_variable].isin(time_period)
            )
            if op == "mean":
                X1_special.append(self.foo[mask][predictor].mean())
            elif op == "std":
                X1_special.append(self.foo[mask][predictor].std())
            elif op == "median":
                X1_special.append(self.foo[mask][predictor].median())
            else:
                raise ValueError(f"Invalid predictors_op: {self.predictors_op}")

        X1_special = pd.Series(X1_special, index=X0_special_columns).rename(
            self.treatment_identifier
        )
        X1 = pd.concat([X1_nonspecial, X1_special], axis=0)
        return X0, X1

    def make_outcome_mats(
        self, time_period: Optional[IsinArg_t] = None
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Generates the time-series matrices to use as input to the fit
        method of the synthetic control computation.

        Parameters
        ----------
        time_period : Iterable | pandas.Series | dict, optional
            Time period to use when generating the matrices, defaults to
            time_optimize_ssr set when initialising the class, by default None

        Returns
        -------
        tuple[pd.DataFrame, pd.Series]
            Returns the matrices Z0, Z1 (using the notation of the Abadie,
            Diamond & Hainmueller paper).

        :meta private:
        """
        time_period = time_period if time_period is not None else self.time_optimize_ssr

        Z = self.foo[self.foo[self.time_variable].isin(time_period)].pivot(
            index=self.time_variable, columns=self.unit_variable, values=self.dependent
        )
        Z0, Z1 = Z[list(self.controls_identifier)], Z[self.treatment_identifier]
        return Z0, Z1

    def __str__(self) -> str:
        str_rep = (
            "Dataprep\n"
            f"Treated unit: {self.treatment_identifier}\n"
            f"Dependent variable: {self.dependent}\n"
            f"Control units: {', '.join([str(c) for c in self.controls_identifier])}\n"
            f"Time range in data: {min(self.foo[self.time_variable])}"
            f" - {max(self.foo[self.time_variable])}\n"
            f"Time range for loss minimization: {self.time_optimize_ssr}\n"
            f"Time range for predictors: {self.time_predictors_prior}\n"
            f"Predictors: {', '.join([str(p) for p in self.predictors])}\n"
        )

        if self.special_predictors:
            str_special_pred = ""
            for predictor, time_range, op in self.special_predictors:
                rep = f"    `{predictor}` over `{time_range}` using `{op}`\n"
                str_special_pred = str_special_pred + rep
            str_rep = str_rep + f"Special predictors:\n" + str_special_pred
        return str_rep
