"""Descriptive sensitivity to a prespecified registry of research designs."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from inspect import signature
from importlib.metadata import PackageNotFoundError, version
from statistics import mean, median_high, median_low
from typing import Any, Iterable, Optional
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .dataprep import Dataprep
from .synth import Synth


_DESIGN_FIELDS = {
    "controls_identifier",
    "predictors",
    "predictors_op",
    "special_predictors",
    "time_predictors_prior",
    "time_optimize_ssr",
}


def _package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _rms(values):
    """Compute RMS without squaring values in their original units."""
    values = np.asarray(values, dtype=float)
    scale = np.max(np.abs(values))
    return float(scale * np.sqrt(np.mean((values / scale) ** 2))) if scale else 0.0


def _median(values):
    """Average the central observations without overflowing their sum."""
    return (
        mean([float(median_low(values)), float(median_high(values))])
        if len(values)
        else np.nan
    )


def _std(values):
    """Return sample SD, or NaN if undefined or outside floating-point range."""
    if len(values) < 2:
        return np.nan
    values = np.asarray(values, dtype=float)
    scale = float(np.max(np.abs(values)))
    with np.errstate(over="ignore"):
        value = float(np.std(values / scale, ddof=1) * scale) if scale else 0.0
    return value if np.isfinite(value) else np.nan


@dataclass
class SensitivityResult:
    """All attempted designs, with unrounded estimates and no fit filtering.

    ``summary`` has one row per attempt. ``paths``, ``weights`` and
    ``predictor_weights`` retain finite fitted results, including fits with
    warnings. ``failures`` records exceptions; ``warnings`` records estimator
    warnings, including convergence problems. ``registry`` preserves merged
    configurations before validation, while ``baseline`` and ``metadata``
    preserve the original design and common settings. ``fits`` contains each
    fitted estimator, including its outer ``optimization_result``.
    """

    summary: pd.DataFrame
    paths: pd.DataFrame
    weights: pd.DataFrame
    failures: pd.DataFrame
    predictor_weights: pd.DataFrame
    warnings: pd.DataFrame
    registry: dict
    baseline: dict
    metadata: dict
    fits: dict

    def dispersion(self) -> pd.Series:
        """Describe finite estimates; sign shares are not causal probabilities.

        Estimates from warning-bearing fits are included. Failure and warning
        counts accompany the denominator used for all descriptive shares.
        Standard deviation is NaN when fewer than two estimates are finite or
        its value exceeds the floating-point range.
        """
        effects = self.summary.loc[np.isfinite(self.summary["att"]), "att"]
        return pd.Series(
            {
                "attempted": len(self.summary),
                "estimated": len(effects),
                "failed": (self.summary["status"] == "failed").sum(),
                "with_warnings": (self.summary["status"] == "warning").sum(),
                "min": effects.min(),
                "median": _median(effects),
                "max": effects.max(),
                "std": _std(effects),
                "positive_share": (effects > 0).mean() if len(effects) else np.nan,
                "negative_share": (effects < 0).mean() if len(effects) else np.nan,
                "zero_share": (effects == 0).mean() if len(effects) else np.nan,
            }
        )

    def plot(self, ax=None):
        """Return a specification-curve Axes showing every attempted design.

        Finite estimates are ordered by ATT. Failed designs occupy labelled
        positions at the end, with text in axes coordinates and no effect value.
        """
        owns_axes = ax is None
        if owns_axes:
            _, ax = plt.subplots()
        ordered = self.summary.sort_values("att", kind="stable", na_position="last")
        for position, row in enumerate(ordered.itertuples()):
            if row.status == "failed":
                ax.text(
                    position,
                    0.02,
                    "failed",
                    rotation=90,
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="bottom",
                )
            else:
                ax.scatter(
                    position, row.att, marker="x" if row.status == "warning" else "o"
                )
        ax.axhline(0, color="grey", linewidth=0.8)
        ax.set_xticks(range(len(ordered)))
        ax.set_xticklabels(ordered.spec_id, rotation=90)
        ax.set_ylabel("ATT")
        ax.set_xlabel("Specification (x: fit with warnings; failed: no estimate)")
        if owns_axes:
            ax.figure.tight_layout()
        return ax


def _periods(values, available, name, treatment_time=None):
    values = list(values)
    if not values or len(set(values)) != len(values):
        raise ValueError(f"{name} must contain nonempty, unique periods.")
    if not set(values).issubset(available):
        raise ValueError(f"{name} contains periods absent from the panel.")
    if treatment_time is not None and any(t >= treatment_time for t in values):
        raise ValueError(f"{name} must use strictly pre-treatment information.")
    return values


def run_sensitivity(
    data: pd.DataFrame,
    base_dataprep: dict,
    specifications: Iterable[dict],
    treatment_time: Any,
    att_periods: Iterable,
    estimator=Synth,
    fit_options: Optional[dict] = None,
    excluded_donors: Iterable = (),
) -> SensitivityResult:
    """Refit classical Synth over explicitly declared design overrides.

    Parameters
    ----------
    data : pandas.DataFrame
        Original panel. It is copied before fitting.
    base_dataprep : dict
        Arguments to :class:`Dataprep`, excluding ``foo``.
    specifications : iterable of dict
        Each entry has a unique ``name`` and overrides for donor pool,
        predictors, aggregation operations/windows or fitting periods. All
        other overrides fail that design. An unchanged ``baseline`` is added
        first when absent. The name ``baseline`` is reserved for that design.
    treatment_time : scalar
        First treated period. All fitting and predictor periods must precede it.
    att_periods : iterable
        Fixed, nonempty, unique observed periods on or after treatment.
    estimator : class
        Only :class:`Synth` is supported.
    fit_options : dict, optional
        Fixed ``Synth.fit`` optimizer options or ``custom_V``. Matrix inputs
        cannot be supplied through this argument. With ``custom_V``, overrides
        of predictors or special_predictors fail to prevent positional weights
        from being reassigned to different matching variables.
    excluded_donors : iterable
        Collection of known unit IDs, not a scalar string. Unknown IDs raise
        before fitting. Any pool containing an excluded unit fails.
        Eligibility beyond this declared list is the researcher's responsibility.

    Returns
    -------
    SensitivityResult
        Complete attempts, configurations and diagnostics. ``pre_rmse`` uses
        all observed pre-treatment periods, common across specifications;
        ``fit_rmse`` uses each design's optimization periods.

    Notes
    -----
    Declare designs before inspecting post-treatment estimates. Dispersion is
    descriptive and does not form a confidence interval. Precomputed predictor
    columns must themselves have been constructed without future information.
    Missing or nonfinite outcomes anywhere on the common trajectory fail a
    design; they never shorten its ATT period. Poor fit is never filtered out.
    """
    if estimator is not Synth:
        raise ValueError("Only classical Synth is supported.")
    baseline = deepcopy(base_dataprep)
    if "foo" in baseline:
        raise ValueError("Pass the panel as data, not base_dataprep['foo'].")
    options = {
        k: deepcopy(p.default)
        for k, p in signature(Synth.fit).parameters.items()
        if k in {"custom_V", "optim_method", "optim_initial", "optim_options"}
    }
    if set(fit_options or {}) - set(options):
        raise ValueError(
            "fit_options may only contain fixed optimizer settings/custom_V."
        )
    options.update(deepcopy(fit_options or {}))
    data = data.copy(deep=True)
    unit, time = baseline["unit_variable"], baseline["time_variable"]
    treated = baseline["treatment_identifier"]
    treated = treated if isinstance(treated, (list, tuple)) else [treated]
    available = sorted(data.loc[data[unit].isin(treated), time].dropna().unique())
    if pd.isna(treatment_time):
        raise ValueError("treatment_time must not be missing.")
    post = _periods(att_periods, available, "att_periods")
    if any(t < treatment_time for t in post):
        raise ValueError("att_periods must be on or after treatment_time.")
    pre = [t for t in available if t < treatment_time]
    if not pre:
        raise ValueError("At least one pre-treatment period is required.")
    if isinstance(excluded_donors, (str, bytes)):
        raise ValueError("excluded_donors must be a collection of known unit IDs.")
    excluded = tuple(excluded_donors)
    if not set(excluded).issubset(data[unit].unique()):
        raise ValueError("excluded_donors contains unit IDs absent from the panel.")
    specs = deepcopy(list(specifications))
    names = [spec["name"] for spec in specs]
    if any(not isinstance(n, str) or not n for n in names) or len(set(names)) != len(
        names
    ):
        raise ValueError("Specification names must be unique nonempty strings.")
    if "baseline" not in names:
        specs.insert(0, {"name": "baseline"})
    registry = {
        spec["name"]: {
            **deepcopy(baseline),
            **{k: v for k, v in spec.items() if k != "name"},
        }
        for spec in specs
    }
    summaries, paths, weights, failures, predictor_weights, warning_rows = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    fits = {}
    for spec in specs:
        name = spec["name"]
        cfg = deepcopy(registry[name])
        row = dict(
            spec_id=name,
            att=np.nan,
            pre_rmse=np.nan,
            fit_rmse=np.nan,
            n_donors=np.nan,
            status="failed",
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                if set(spec) - _DESIGN_FIELDS - {"name"}:
                    raise ValueError(
                        "Only design fields may vary; study identity and solver settings are fixed."
                    )
                if name == "baseline" and len(spec) != 1:
                    raise ValueError(
                        "The baseline specification cannot contain overrides."
                    )
                if options.get("custom_V") is not None and set(spec).intersection(
                    {"predictors", "special_predictors"}
                ):
                    raise ValueError(
                        "custom_V requires fixed predictors and special_predictors."
                    )
                donors = list(cfg["controls_identifier"])
                row["n_donors"] = len(donors)
                if not donors or len(set(donors)) != len(donors):
                    raise ValueError("Donor pool must be nonempty and unique.")
                if set(donors).intersection(excluded):
                    raise ValueError(
                        "Donor pool contains a declared treated or contaminated donor."
                    )
                # Canonical ordering makes numerical fits invariant to donor-list order.
                cfg["controls_identifier"] = sorted(donors)
                for field in ("time_predictors_prior", "time_optimize_ssr"):
                    cfg[field] = _periods(cfg[field], available, field, treatment_time)
                specials = []
                for predictor, periods, operation in (
                    cfg.get("special_predictors") or []
                ):
                    specials.append(
                        (
                            predictor,
                            _periods(
                                periods, available, "special_predictors", treatment_time
                            ),
                            operation,
                        )
                    )
                cfg["special_predictors"] = specials or None
                dp = Dataprep(foo=data.copy(deep=True), **cfg)
                X0, X1 = dp.make_covariate_mats()
                if X0.empty or not np.isfinite(np.column_stack([X0, X1])).all():
                    raise ValueError("Matching predictors must be nonempty and finite.")
                if (pd.concat([X0, X1], axis=1).std(axis=1) == 0).any():
                    raise ValueError("Matching predictors must vary across units.")
                Z0, Z1 = dp.make_outcome_mats(time_period=available)
                Z0, Z1 = Z0.reindex(available), Z1.reindex(available)
                if not np.isfinite(np.column_stack([Z0, Z1])).all():
                    raise ValueError(
                        "All common trajectory periods require finite observed outcomes."
                    )
                model = estimator()
                model.fit(dataprep=dp, **deepcopy(options))
                w = np.asarray(model.W, dtype=float)
                if not np.isfinite(w).all() or not np.isfinite(model.V).all():
                    raise ValueError("Estimator returned nonfinite weights.")
                synthetic = Z0 @ w
                gap = Z1 - synthetic
                if not np.isfinite(gap).all():
                    raise ValueError("Estimator returned a nonfinite trajectory.")
                statistics = dict(
                    att=mean(map(float, gap.loc[post])),
                    pre_rmse=_rms(gap.loc[pre]),
                    fit_rmse=_rms(gap.loc[cfg["time_optimize_ssr"]]),
                )
                if not all(np.isfinite(value) for value in statistics.values()):
                    raise ValueError("Estimator returned nonfinite summary statistics.")
                row.update(
                    **statistics,
                    status="warning" if caught else "ok",
                )
                fits[name] = model
                paths.extend(
                    dict(
                        spec_id=name,
                        time=t,
                        observed=Z1.loc[t],
                        synthetic=synthetic.loc[t],
                        gap=gap.loc[t],
                    )
                    for t in available
                )
                weights.extend(
                    dict(spec_id=name, donor=d, weight=v) for d, v in zip(Z0.columns, w)
                )
                predictor_weights.extend(
                    dict(spec_id=name, predictor=p, weight=v)
                    for p, v in zip(X0.index, model.V)
                )
            except Exception as exc:
                failures.append(
                    dict(
                        spec_id=name,
                        exception_type=type(exc).__name__,
                        message=str(exc),
                    )
                )
            warning_rows.extend(
                dict(
                    spec_id=name,
                    warning_type=w.category.__name__,
                    message=str(w.message),
                )
                for w in caught
            )
        summaries.append(row)
    return SensitivityResult(
        summary=pd.DataFrame(summaries),
        paths=pd.DataFrame(
            paths, columns=["spec_id", "time", "observed", "synthetic", "gap"]
        ),
        weights=pd.DataFrame(weights, columns=["spec_id", "donor", "weight"]),
        failures=pd.DataFrame(
            failures, columns=["spec_id", "exception_type", "message"]
        ),
        predictor_weights=pd.DataFrame(
            predictor_weights, columns=["spec_id", "predictor", "weight"]
        ),
        warnings=pd.DataFrame(
            warning_rows, columns=["spec_id", "warning_type", "message"]
        ),
        registry=registry,
        baseline=baseline,
        metadata=dict(
            treatment_time=treatment_time,
            att_periods=post,
            pre_periods=pre,
            fit_options=options,
            excluded_donors=excluded,
            estimator="pysyncon.Synth",
            versions={
                name: _package_version(name)
                for name in ("pysyncon", "numpy", "pandas", "scipy")
            },
        ),
        fits=fits,
    )
