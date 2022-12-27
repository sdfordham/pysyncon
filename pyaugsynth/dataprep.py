import pandas as pd
from typing import Iterable, Union, Literal


def dataprep(
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
    time_plot: Iterable[Union[int, str]],
):
    X0 = list()
    for ci in controls_identifier:
        this_control = list()
        if not isinstance(ci, int) and not isinstance(ci, str):
            raise TypeError("Elements of controls_identifier must be int or str.")
        for pred in predictors:
            if not isinstance(pred, str):
                raise TypeError("Elements of predictors must be str.")
            if pred not in foo.columns:
                raise ValueError(f"{pred} not in dataframe column names")
            mask = (foo[unit_variable] == ci) & (
                foo[time_variable].isin(time_predictors_prior)
            )
            if predictors_op == "mean":
                this_control.append(foo[mask][pred].mean())
            elif predictors_op == "std":
                this_control.append(foo[mask][pred].std())
            elif predictors_op == "median":
                this_control.append(foo[mask][pred].median())
            else:
                raise ValueError(f"Invalid predictors_op: {predictors_op}")
        for sp_pred, sp_period, sp_op in special_predictors:
            if not isinstance(sp_pred, str):
                raise TypeError("Elements of predictors must be str.")
            if sp_pred not in foo.columns:
                raise ValueError(f"{sp_pred} not in dataframe column names")
            mask = (foo[unit_variable] == ci) & (foo[time_variable].isin(sp_period))
            if sp_op == "mean":
                this_control.append(foo[mask][sp_pred].mean())
            elif sp_op == "std":
                this_control.append(foo[mask][sp_pred].std())
            elif sp_op == "median":
                this_control.append(foo[mask][sp_pred].median())
            else:
                raise ValueError(f"Invalid predictors_op: {predictors_op}")
        X0.append(this_control)

    X0_columns = list(predictors)
    for i, (sp_pred, _, _) in enumerate(special_predictors, 1):
        X0_columns.append(f"special.{i}.{sp_pred}")

    X0 = pd.DataFrame(X0, columns=X0_columns, index=controls_identifier).T

    X1 = list()
    for pred in predictors:
        mask = (foo[unit_variable] == treatment_identifier) & (
            foo[time_variable].isin(time_predictors_prior)
        )
        if predictors_op == "mean":
            X1.append(foo[mask][pred].mean())
        elif predictors_op == "std":
            X1.append(foo[mask][pred].std())
        elif predictors_op == "median":
            X1.append(foo[mask][pred].median())
        else:
            raise ValueError(f"Invalid predictors_op: {predictors_op}")
    for sp_pred, sp_period, sp_op in special_predictors:
        mask = (foo[unit_variable] == treatment_identifier) & (
            foo[time_variable].isin(sp_period)
        )
        if sp_op == "mean":
            X1.append(foo[mask][sp_pred].mean())
        elif sp_op == "std":
            X1.append(foo[mask][sp_pred].std())
        elif sp_op == "median":
            X1.append(foo[mask][sp_pred].median())
        else:
            raise ValueError(f"Invalid predictors_op: {predictors_op}")

    X1 = pd.Series(X1, index=X0_columns).rename(treatment_identifier)

    Z = foo[foo[time_variable].isin(time_optimize_ssr)].pivot(
        index=time_variable, columns=unit_variable, values=dependent
    )
    Z0, Z1 = Z[list(controls_identifier)], Z[treatment_identifier]

    return X0, X1, Z0, Z1
