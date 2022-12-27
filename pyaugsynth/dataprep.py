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

    X0_nonspecial = (
        foo[foo[time_variable].isin(time_predictors_prior)]
        .groupby(unit_variable)[list(predictors)]
        .agg(predictors_op)
        .T
    )
    X1_nonspecial = X0_nonspecial[treatment_identifier]
    X0_nonspecial = X0_nonspecial[list(controls_identifier)]

    X0_special = list()
    for ci in controls_identifier:
        this_control = list()
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
        X0_special.append(this_control)

    X0_special_columns = list()
    for i, (sp_pred, _, _) in enumerate(special_predictors, 1):
        X0_special_columns.append(f"special.{i}.{sp_pred}")

    X0_special = pd.DataFrame(
        X0_special, columns=X0_special_columns, index=controls_identifier
    ).T
    X0 = pd.concat([X0_nonspecial, X0_special], axis=0)

    X1_special = list()
    for sp_pred, sp_period, sp_op in special_predictors:
        mask = (foo[unit_variable] == treatment_identifier) & (
            foo[time_variable].isin(sp_period)
        )
        if sp_op == "mean":
            X1_special.append(foo[mask][sp_pred].mean())
        elif sp_op == "std":
            X1_special.append(foo[mask][sp_pred].std())
        elif sp_op == "median":
            X1_special.append(foo[mask][sp_pred].median())
        else:
            raise ValueError(f"Invalid predictors_op: {predictors_op}")

    X1_special = pd.Series(X1_special, index=X0_special_columns).rename(
        treatment_identifier
    )
    X1 = pd.concat([X1_nonspecial, X1_special], axis=0)

    Z = foo[foo[time_variable].isin(time_optimize_ssr)].pivot(
        index=time_variable, columns=unit_variable, values=dependent
    )
    Z0, Z1 = Z[list(controls_identifier)], Z[treatment_identifier]

    return X0, X1, Z0, Z1
