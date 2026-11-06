"""Descriptive sensitivity to a prespecified registry of research designs."""

from __future__ import annotations

from contextlib import contextmanager
from copy import copy, deepcopy
from dataclasses import dataclass, field
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
    # Remove the common level before scaling to preserve small differences.
    with np.errstate(over="ignore"):
        shifted = values - values[0]
    if np.isfinite(shifted).all():
        values = shifted
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

    ``validation`` contains one diagnostic row per design and held-out window,
    including exception type/message for failed folds. ``validation_paths``,
    ``validation_weights`` and ``validation_warnings`` carry the same spec_id
    and fold_id keys. ``validation_registry`` and ``validation_fits`` are nested
    by specification and fold. These are forecasting exercises, not policy fits.
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

    validation: pd.DataFrame = field(default_factory=pd.DataFrame)
    validation_paths: pd.DataFrame = field(default_factory=pd.DataFrame)
    validation_weights: pd.DataFrame = field(default_factory=pd.DataFrame)
    validation_warnings: pd.DataFrame = field(default_factory=pd.DataFrame)
    validation_registry: dict = field(default_factory=dict)
    validation_fits: dict = field(default_factory=dict)

    def dispersion(self, validation_only=False) -> pd.Series:
        """Describe finite estimates; sign shares are not causal probabilities.

        Estimates from warning-bearing fits are included. Failure and warning
        counts accompany the denominator used for all descriptive shares.
        Standard deviation is NaN when fewer than two estimates are finite or
        its value exceeds the floating-point range.
        Set ``validation_only=True`` to summarize the subset passing the
        validation_max_rmse declared before fitting. Complete results remain
        in the original tables.
        """
        table = self.summary
        if validation_only:
            if self.metadata.get("validation_max_rmse") is None:
                raise ValueError(
                    "Declare validation_max_rmse before fitting to summarize its subset."
                )
            table = table.loc[table.validation_pass]
        effects = table.loc[np.isfinite(table["att"]), "att"]
        return pd.Series(
            {
                "attempted": len(table),
                "estimated": len(effects),
                "failed": (table["status"] == "failed").sum(),
                "with_warnings": (table["status"] == "warning").sum(),
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

    def plot_validation(self, ax=None):
        """Plot ATT against pooled held-out RMSE, labelling every design.

        Attempts without both quantities are named in the plot margin. A
        declared validation_max_rmse is drawn without removing other designs.
        """
        if not self.metadata.get("validation_windows"):
            raise ValueError("Declare validation_windows before fitting.")
        owns_axes = ax is None
        if owns_axes:
            _, ax = plt.subplots()
        table = self.summary
        finite = np.isfinite(table.att) & np.isfinite(table.validation_rmse)
        shown = table.loc[finite]
        for position, row in enumerate(shown.itertuples()):
            warning = row.status == "warning" or row.validation_status == "warning"
            ax.scatter(row.validation_rmse, row.att, marker="x" if warning else "o")
            ax.annotate(
                row.spec_id,
                (row.validation_rmse, row.att),
                xytext=(5, 8 if position % 2 == 0 else 22),
                textcoords="offset points",
            )
        missing = table.loc[~finite, "spec_id"].tolist()
        if missing:
            ax.text(
                0,
                1.02,
                "ATT or validation unavailable: " + ", ".join(missing),
                transform=ax.transAxes,
                va="bottom",
                wrap=True,
            )
        limit = self.metadata.get("validation_max_rmse")
        if limit is not None:
            ax.axvline(limit, color="grey", linestyle="--")
        ax.axhline(0, color="grey", linewidth=0.8)
        ax.set_xlabel("Pre-treatment validation RMSE")
        ax.set_ylabel("ATT")
        if owns_axes:
            ax.figure.tight_layout()
        return ax


@contextmanager
def _distinct_warnings():
    """Collect each distinct warning once.

    A solver can repeat one warning at every objective evaluation, so keeping
    every instance grows with the length of the search.
    """
    seen = {}

    def record(message, category, *args, **kwargs):
        seen[(category.__name__, str(message))] = None

    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.showwarning = record
        yield seen


def _periods(values, available, name, treatment_time=None):
    values = list(values)
    if not values or len(set(values)) != len(values):
        raise ValueError(f"{name} must contain nonempty, unique periods.")
    if not set(values).issubset(available):
        raise ValueError(f"{name} contains periods absent from the panel.")
    if treatment_time is not None and any(t >= treatment_time for t in values):
        raise ValueError(f"{name} must use strictly pre-treatment information.")
    return values


def _validation_windows(windows, pre, limit):
    if limit is not None and (not np.isfinite(limit) or limit < 0):
        raise ValueError("validation_max_rmse must be finite and nonnegative.")
    if windows is None:
        if limit is not None:
            raise ValueError("validation_max_rmse requires validation_windows.")
        return []
    windows = [sorted(_periods(w, pre, "validation_windows")) for w in windows]
    if not windows:
        raise ValueError("validation_windows must contain at least one window.")
    seen = set()
    for window in windows:
        if min(window) <= min(pre) or seen.intersection(window):
            raise ValueError(
                "validation_windows must be disjoint and have earlier training periods."
            )
        seen.update(window)
    return sorted(windows, key=lambda w: w[0])


def _prepare_design(spec, cfg, available, treatment_time, excluded, options):
    cfg = deepcopy(cfg)
    if set(spec) - _DESIGN_FIELDS - {"name"}:
        raise ValueError(
            "Only design fields may vary; study identity and solver settings are fixed."
        )
    if spec["name"] == "baseline" and len(spec) != 1:
        raise ValueError("The baseline specification cannot contain overrides.")
    if options.get("custom_V") is not None and set(spec).intersection(
        {"predictors", "special_predictors"}
    ):
        raise ValueError("custom_V requires fixed predictors and special_predictors.")
    donors = list(cfg["controls_identifier"])
    if not donors or len(set(donors)) != len(donors):
        raise ValueError("Donor pool must be nonempty and unique.")
    if set(donors).intersection(excluded):
        raise ValueError(
            "Donor pool contains a declared treated or contaminated donor."
        )
    cfg["controls_identifier"] = sorted(donors)
    for field in ("time_predictors_prior", "time_optimize_ssr"):
        cfg[field] = _periods(cfg[field], available, field, treatment_time)
    cfg["special_predictors"] = [
        (
            predictor,
            _periods(periods, available, "special_predictors", treatment_time),
            op,
        )
        for predictor, periods, op in (cfg.get("special_predictors") or [])
    ] or None
    return cfg


def _fit_design(data, cfg, options):
    dp = Dataprep(foo=data.copy(deep=True), **cfg)
    _outcomes(dp, data, cfg["time_optimize_ssr"])
    X0, X1 = dp.make_covariate_mats()
    if X0.empty or not np.isfinite(np.column_stack([X0, X1])).all():
        raise ValueError("Matching predictors must be nonempty and finite.")
    if (pd.concat([X0, X1], axis=1).std(axis=1) == 0).any():
        raise ValueError("Matching predictors must vary across units.")
    model = Synth()
    model.fit(dataprep=dp, **deepcopy(options))
    if not np.isfinite(model.W).all() or not np.isfinite(model.V).all():
        raise ValueError("Estimator returned nonfinite weights.")
    return model


def _outcomes(dataprep, data, periods):
    dp = copy(dataprep)
    dp.foo = data
    Z0, Z1 = dp.make_outcome_mats(time_period=periods)
    Z0, Z1 = Z0.reindex(periods), Z1.reindex(periods)
    if not np.isfinite(np.column_stack([Z0, Z1])).all():
        raise ValueError(
            "All common trajectory periods require finite observed outcomes."
        )
    return Z0, Z1


def _trajectory(model, data, periods):
    Z0, observed = _outcomes(model.dataprep, data, periods)
    synthetic = Z0 @ model.W
    gap = observed - synthetic
    if not np.isfinite(gap).all():
        raise ValueError("Estimator returned a nonfinite trajectory.")
    return observed, synthetic, gap


def _run_validation(
    data, registry, specs, options, available, treatment_time, excluded, windows
):
    rows, paths, weights, warning_rows = [], [], [], []
    configurations, fits = {}, {}
    for spec in specs:
        name = spec["name"]
        configurations[name], fits[name] = {}, {}
        for number, window in enumerate(windows, 1):
            fold = f"fold_{number}"
            train = [t for t in available if t < window[0]]
            row = dict(
                spec_id=name,
                fold_id=fold,
                rmse=np.nan,
                n_train=len(train),
                n_validation=len(window),
                status="failed",
                exception_type=None,
                message=None,
            )
            cfg = deepcopy(registry[name])
            with _distinct_warnings() as caught:
                try:
                    cfg = _prepare_design(
                        spec, cfg, available, treatment_time, excluded, options
                    )
                    for key in ("time_predictors_prior", "time_optimize_ssr"):
                        cfg[key] = [t for t in cfg[key] if t < window[0]]
                    cfg["special_predictors"] = [
                        (p, [t for t in periods if t < window[0]], op)
                        for p, periods, op in (cfg["special_predictors"] or [])
                    ] or None
                    effective = _prepare_design(
                        {"name": name}, cfg, train, window[0], excluded, options
                    )
                    training = data.loc[data[cfg["time_variable"]].isin(train)]
                    model = _fit_design(training, effective, options)
                    fits[name][fold] = model
                    weights.extend(
                        dict(spec_id=name, fold_id=fold, donor=d, weight=w)
                        for d, w in zip(effective["controls_identifier"], model.W)
                    )
                    Z1, synthetic, gap = _trajectory(model, data, train + window)
                    row.update(
                        rmse=_rms(gap.loc[window]),
                        status="warning" if caught else "ok",
                    )
                    paths.extend(
                        dict(
                            spec_id=name,
                            fold_id=fold,
                            time=t,
                            observed=Z1.loc[t],
                            synthetic=synthetic.loc[t],
                            gap=gap.loc[t],
                        )
                        for t in window
                    )
                except Exception as exc:
                    row.update(exception_type=type(exc).__name__, message=str(exc))
                configurations[name][fold] = deepcopy(cfg)
                warning_rows.extend(
                    dict(spec_id=name, fold_id=fold, warning_type=kind, message=message)
                    for kind, message in caught
                )
            rows.append(row)
    return dict(
        validation=pd.DataFrame(
            rows,
            columns=[
                "spec_id",
                "fold_id",
                "rmse",
                "n_train",
                "n_validation",
                "status",
                "exception_type",
                "message",
            ],
        ),
        validation_paths=pd.DataFrame(
            paths,
            columns=["spec_id", "fold_id", "time", "observed", "synthetic", "gap"],
        ),
        validation_weights=pd.DataFrame(
            weights, columns=["spec_id", "fold_id", "donor", "weight"]
        ),
        validation_warnings=pd.DataFrame(
            warning_rows, columns=["spec_id", "fold_id", "warning_type", "message"]
        ),
        validation_registry=configurations,
        validation_fits=fits,
    )


def run_sensitivity(
    data: pd.DataFrame,
    base_dataprep: dict,
    specifications: Iterable[dict],
    treatment_time: Any,
    att_periods: Iterable,
    estimator=Synth,
    fit_options: Optional[dict] = None,
    excluded_donors: Iterable = (),
    validation_windows: Optional[Iterable[Iterable]] = None,
    validation_max_rmse: Optional[float] = None,
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
    validation_windows : iterable of iterables, optional
        Disjoint observed pre-treatment periods to predict. Each fold trains
        strictly before its first period, intersecting every matching window
        with that training history. Empty required windows fail the fold.
        Windows are ordered chronologically and labelled fold_1, fold_2, etc.
    validation_max_rmse : float, optional
        Finite nonnegative cutoff declared before fitting. Adds validation_pass
        to summary; it never removes designs or changes final fits.

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
    Validation RMSE pools all held-out squared gaps, giving each declared
    period equal weight. Every fold must succeed for an aggregate score.
    The final ATT is refitted on the original full pre-treatment design.
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
    windows = _validation_windows(validation_windows, pre, validation_max_rmse)
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
    validation = _run_validation(
        data, registry, specs, options, available, treatment_time, excluded, windows
    )
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
        with _distinct_warnings() as caught:
            try:
                cfg = _prepare_design(
                    spec, cfg, available, treatment_time, excluded, options
                )
                row["n_donors"] = len(cfg["controls_identifier"])
                model = _fit_design(data, cfg, options)
                Z1, synthetic, gap = _trajectory(model, data, available)
                w = np.asarray(model.W, dtype=float)
                X0, _ = model.dataprep.make_covariate_mats()
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
                    dict(spec_id=name, donor=d, weight=v)
                    for d, v in zip(cfg["controls_identifier"], w)
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
                dict(spec_id=name, warning_type=kind, message=message)
                for kind, message in caught
            )
        summaries.append(row)
    summary = pd.DataFrame(summaries)
    if windows:
        scores, statuses = [], []
        for name in summary.spec_id:
            folds = validation["validation"].query("spec_id == @name")
            failed = (folds.status == "failed").any()
            gaps = validation["validation_paths"].query("spec_id == @name").gap
            scores.append(np.nan if failed else _rms(gaps))
            statuses.append(
                "failed"
                if failed
                else "warning"
                if (folds.status == "warning").any()
                else "ok"
            )
        summary["validation_rmse"] = scores
        summary["validation_status"] = statuses
        if validation_max_rmse is not None:
            summary["validation_pass"] = summary.validation_rmse <= validation_max_rmse
    return SensitivityResult(
        summary=summary,
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
            validation_windows=windows,
            validation_max_rmse=validation_max_rmse,
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
        **validation,
    )
