from typing import Any, Iterable, Union, Optional, Literal, Sequence, Mapping

import pandas as pd
from pandas._typing import Axes


PredictorsOp_t = Literal["mean", "std", "median"]
IsinArg_t = Union[Iterable, pd.Series, dict]
SpecialPredictor_t = tuple[
    Any, Union[pd.Series, pd.DataFrame, Sequence, Mapping], PredictorsOp_t
]


class Dataprep:
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
        time_period = time_period if time_period is not None else self.time_optimize_ssr

        Z = self.foo[self.foo[self.time_variable].isin(time_period)].pivot(
            index=self.time_variable, columns=self.unit_variable, values=self.dependent
        )
        Z0, Z1 = Z[list(self.controls_identifier)], Z[self.treatment_identifier]
        return Z0, Z1
