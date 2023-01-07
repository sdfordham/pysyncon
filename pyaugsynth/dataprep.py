import pandas as pd
from typing import Iterable, Union, Optional

from ._types import TimeRange_t, PredictorsOp_t, SpecialPredictor_t


class Dataprep:
    def __init__(
        self,
        foo: pd.DataFrame,
        predictors: list,
        predictors_op: PredictorsOp_t,
        time_predictors_prior: TimeRange_t,
        dependent: Union[int, str],
        unit_variable: Union[int, str],
        time_variable: Union[int, str],
        treatment_identifier: Union[int, str],
        controls_identifier: list,
        time_optimize_ssr: TimeRange_t,
        special_predictors: Optional[list[SpecialPredictor_t]] = None,
    ) -> None:
        if not isinstance(foo, pd.DataFrame):
            raise TypeError("foo must be pandas.DataFrame.")
        self.foo: pd.DataFrame = foo

        if not isinstance(predictors, list):
            raise TypeError(f"predictors must be a list.{type(predictors)}")
        for predictor in predictors:
            if predictor not in foo.columns:
                raise ValueError(f"predictor {predictor} not in foo columns.")
        self.predictors: Iterable = predictors

        if predictors_op not in ("mean", "std", "median"):
            raise ValueError("predictors_op must be one of mean, std, median.")
        self.predictors_op: PredictorsOp_t = predictors_op

        if not isinstance(time_predictors_prior, (list, range)):
            raise TypeError("time_predictors_prior must be of type list or range.")
        self.time_predictors_prior: TimeRange_t = time_predictors_prior

        if dependent not in foo.columns:
            raise ValueError(f"dependent {dependent} not in foo columns.")
        self.dependent: Union[int, str] = dependent

        if unit_variable not in foo.columns:
            raise ValueError(f"unit_variable {unit_variable} not in foo columns.")
        self.unit_variable: Union[int, str] = unit_variable

        if time_variable not in foo.columns:
            raise ValueError(f"time_variable {time_variable} not in foo columns.")
        self.time_variable: Union[int, str] = time_variable

        uniq_ident = foo[unit_variable].unique()
        if treatment_identifier not in uniq_ident:
            raise ValueError(
                f'treatment_identifier {treatment_identifier} not found in foo["{unit_variable}"].'
            )
        self.treatment_identifier: Union[int, str] = treatment_identifier

        if not isinstance(controls_identifier, list):
            raise TypeError("controls_identifier must be a list.")
        for control in controls_identifier:
            if control == treatment_identifier:
                raise ValueError("treatment_identifier in controls_identifier.")
            if control not in uniq_ident:
                raise ValueError(
                    f'controls_identifier {control} not found in foo["{unit_variable}"].'
                )
        self.controls_identifier: Union[list, tuple] = controls_identifier

        if not isinstance(time_optimize_ssr, (list, range)):
            raise TypeError("time_optimize_ssr must be of type list or range.")
        self.time_optimize_ssr: TimeRange_t = time_optimize_ssr

        if special_predictors:
            if not isinstance(special_predictors, list):
                raise TypeError("special_predictors must be a list.")
            for el in special_predictors:
                if not isinstance(el, tuple) or len(el) != 3:
                    raise ValueError(
                        "Elements of special_predictors should be tuples of length 3."
                    )
                pred, rng, op = el
                if pred not in foo.columns:
                    raise ValueError(
                        f"{pred} in special_predictors not in foo columns."
                    )
                if not isinstance(rng, (list, range)):
                    raise TypeError(
                        f"{rng} in special_predictors must be of type list or range."
                    )
                if op not in ("mean", "std", "median"):
                    raise ValueError(
                        f"{op} in special_predictors must be one of mean, std, median."
                    )
        self.special_predictors: Optional[
            Iterable[SpecialPredictor_t]
        ] = special_predictors

    def make_covariate_mats(self) -> tuple[pd.DataFrame, pd.Series]:
        X_nonspecial = (
            self.foo[self.foo[self.time_variable].isin(self.time_predictors_prior)]
            .groupby(self.unit_variable)[list(self.predictors)]
            .agg(self.predictors_op)
            .T
        )
        X1_nonspecial = X_nonspecial[self.treatment_identifier]
        X0_nonspecial = X_nonspecial[list(self.controls_identifier)]

        if self.special_predictors is None:
            return X0_nonspecial, X1_nonspecial

        X0_special = list()
        for ci in self.controls_identifier:
            this_control = list()
            for sp_pred, sp_period, sp_op in self.special_predictors:
                if not isinstance(sp_pred, str):
                    raise TypeError("Elements of predictors must be str.")
                if sp_pred not in self.foo.columns:
                    raise ValueError(f"{sp_pred} not in dataframe column names")
                mask = (self.foo[self.unit_variable] == ci) & (
                    self.foo[self.time_variable].isin(sp_period)
                )
                if sp_op == "mean":
                    this_control.append(self.foo[mask][sp_pred].mean())
                elif sp_op == "std":
                    this_control.append(self.foo[mask][sp_pred].std())
                elif sp_op == "median":
                    this_control.append(self.foo[mask][sp_pred].median())
                else:
                    raise ValueError(f"Invalid predictors_op: {self.predictors_op}")
            X0_special.append(this_control)

        X0_special_columns = list()
        for i, (sp_pred, _, _) in enumerate(self.special_predictors, 1):
            X0_special_columns.append(f"special.{i}.{sp_pred}")

        X0_special = pd.DataFrame(
            X0_special, columns=X0_special_columns, index=self.controls_identifier
        ).T
        X0 = pd.concat([X0_nonspecial, X0_special], axis=0)

        X1_special = list()
        for sp_pred, sp_period, sp_op in self.special_predictors:
            mask = (self.foo[self.unit_variable] == self.treatment_identifier) & (
                self.foo[self.time_variable].isin(sp_period)
            )
            if sp_op == "mean":
                X1_special.append(self.foo[mask][sp_pred].mean())
            elif sp_op == "std":
                X1_special.append(self.foo[mask][sp_pred].std())
            elif sp_op == "median":
                X1_special.append(self.foo[mask][sp_pred].median())
            else:
                raise ValueError(f"Invalid predictors_op: {self.predictors_op}")

        X1_special = pd.Series(X1_special, index=X0_special_columns).rename(
            self.treatment_identifier
        )
        X1 = pd.concat([X1_nonspecial, X1_special], axis=0)
        return X0, X1

    def make_outcome_mats(self) -> tuple[pd.DataFrame, pd.Series]:
        Z = self.foo[self.foo[self.time_variable].isin(self.time_optimize_ssr)].pivot(
            index=self.time_variable, columns=self.unit_variable, values=self.dependent
        )
        Z0, Z1 = Z[list(self.controls_identifier)], Z[self.treatment_identifier]
        return Z0, Z1
