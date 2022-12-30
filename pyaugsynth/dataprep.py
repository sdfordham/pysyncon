import pandas as pd
from typing import Iterable, Union, Literal


class Dataprep:
    def __init__(
        self,
        foo: pd.DataFrame,
        predictors: Iterable[str],
        predictors_op: Literal["mean", "std", "median"],
        time_predictors_prior: Union[list, tuple, range],
        special_predictors: Iterable,
        dependent: str,
        unit_variable: str,
        time_variable: str,
        treatment_identifier: Union[int, str],
        controls_identifier: Iterable[Union[int, str]],
        time_optimize_ssr: Iterable[Union[int, str]],
    ) -> None:
        self.foo: pd.DataFrame = foo
        self.predictors: Iterable[str] = predictors
        self.predictors_op: Literal["mean", "std", "median"] = predictors_op
        self.time_predictors_prior = time_predictors_prior
        self.special_predictors = special_predictors
        self.dependent = dependent
        self.unit_variable = unit_variable
        self.time_variable = time_variable
        self.treatment_identifier = treatment_identifier
        self.controls_identifier = controls_identifier
        self.time_optimize_ssr = time_optimize_ssr

    def compute_X0_X1(self):
        X0_nonspecial = (
            self.foo[self.foo[self.time_variable].isin(self.time_predictors_prior)]
            .groupby(self.unit_variable)[list(self.predictors)]
            .agg(self.predictors_op)
            .T
        )
        X1_nonspecial = X0_nonspecial[self.treatment_identifier]
        X0_nonspecial = X0_nonspecial[list(self.controls_identifier)]

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

    def compute_Z0_Z1(self):
        Z = self.foo[self.foo[self.time_variable].isin(self.time_optimize_ssr)].pivot(
            index=self.time_variable, columns=self.unit_variable, values=self.dependent
        )
        Z0, Z1 = Z[list(self.controls_identifier)], Z[self.treatment_identifier]
        return Z0, Z1
