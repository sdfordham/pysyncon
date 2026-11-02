Specification sensitivity
=========================

``run_sensitivity`` refits classical ``Synth`` for a declared list of designs.
The treated unit, outcome, treatment date and ATT periods stay fixed. Donor
pools, predictors, predictor aggregation windows and fitting periods can vary.
Each run uses a fresh ``Dataprep`` and estimator with the same solver settings.

Declare the registry before examining post-treatment estimates. The runner
checks that predictor and fitting windows precede treatment. Researchers must
also ensure that precomputed predictor columns contain no future information
and declare known treated or contaminated donors in ``excluded_donors``.
These checks do not establish no spillovers, no anticipation or stability of
the untreated outcome relationship.

Proposition 99 example
----------------------

The example uses the public Proposition 99 panel distributed with
`mlsynth <https://github.com/jgreathouse9/mlsynth/blob/3dd0ce9ee0ee140c379ae487cfebedf997cc57d9/basedata/augmented_cali_long.csv>`_.
Archive the downloaded panel with the analysis. The registry below illustrates
design choices; it is not a claim that these designs have equal credibility.

.. code-block:: python

    import pandas as pd
    from pysyncon.sensitivity import run_sensitivity

    panel = pd.read_csv(
        "https://raw.githubusercontent.com/jgreathouse9/mlsynth/"
        "3dd0ce9ee0ee140c379ae487cfebedf997cc57d9/basedata/augmented_cali_long.csv"
    )
    donors = sorted(set(panel.state) - {"California"})
    base = dict(
        dependent="cigsale", unit_variable="state", time_variable="year",
        treatment_identifier="California", controls_identifier=donors,
        predictors=["p_cig"], predictors_op="mean",
        time_predictors_prior=range(1970, 1989),
        time_optimize_ssr=range(1970, 1989),
        special_predictors=[("cigsale", [1975, 1980, 1988], "mean")],
    )
    result = run_sensitivity(
        data=panel, base_dataprep=base,
        specifications=[
            {"name": "baseline"},
            {"name": "short_pre", "time_optimize_ssr": range(1980, 1989)},
            {"name": "outcomes_only", "predictors": []},
        ],
        treatment_time=1989, att_periods=range(1989, 2001),
    )
    result.summary
    result.dispersion()
    ax = result.plot()

Reading the results
-------------------

``summary`` contains one row per attempt. ``pre_rmse`` uses the same complete
pre-treatment period for every design, while ``fit_rmse`` uses its fitting
window. ATT is the unweighted mean gap over the specified common ATT periods.
Every finite fit contributes its full counterfactual trajectory, unrounded
donor weights and predictor weights. ``registry`` retains merged input designs,
``baseline`` retains the original baseline, and ``metadata`` holds the common
periods, exclusions and resolved optimizer settings. ``fits`` retains estimators
and their outer optimization results. An unchanged baseline is inserted if absent.
The common trajectory uses the treated unit's observed periods; additional periods
from unrelated units do not expand it. A package version is ``None`` when its
distribution metadata is unavailable, as can happen in an uninstalled checkout.

``excluded_donors`` must be a collection of IDs present in the original panel;
scalar strings and unknown IDs raise before any fit. With fixed ``custom_V``,
designs that override ``predictors`` or ``special_predictors`` fail rather than
reassigning positional predictor weights. Donor and fitting-window changes remain
available with fixed predictor weights.

Failures retain a row with missing ATT and an exception in ``failures``.
Warnings, including solver nonconvergence, appear in ``warnings`` and mark the
summary row as ``warning``; their finite estimates remain visible. There is no
automatic fit-quality cutoff. The default curve shows all attempts and marks
failures without assigning them an effect value.
Warnings can originate from candidate solves during the predictor-weight search,
even when the final solution converges. A warning status records a computational
diagnostic, not a conclusion that the selected solution failed. Native ``Synth``
fits now emit inner and outer nonconvergence warnings as well.
Donor-solver messages distinguish predictor-weight search from the final solve.

``dispersion()`` reports the number attempted, estimated and failed alongside
the range, median, sample standard deviation and sign shares among finite
estimates, including fits with warnings. Sign shares describe the registered
designs; they are not probabilities of a causal effect. Estimate dispersion
is not a confidence interval. This varies designs for the original treated
unit and is separate from placebo inference. Other estimators are outside
this interface.

Check pre-treatment prediction
------------------------------

To compare how well the same designs predict outcomes they have not fitted,
declare held-out pre-treatment windows in the original call:

.. code-block:: python

    result = run_sensitivity(
        data=panel, base_dataprep=base,
        specifications=[
            {"name": "baseline"},
            {"name": "short_pre", "time_optimize_ssr": range(1980, 1989)},
        ],
        treatment_time=1989, att_periods=range(1989, 2001),
        validation_windows=[range(1984, 1989)],
        validation_max_rmse=20,
    )
    result.summary
    result.validation
    ax = result.plot_validation()
    result.dispersion(validation_only=True)

The cutoff of 20 cigarette packs per capita is illustrative, not a recommended
standard. Declare any cutoff before inspecting estimates. ``validation_pass``
flags the declared predictive criterion; every attempted design remains in the
tables and figures. Passing does not certify convergence or causal identification.

For each window, training ends before its first period. Predictor averages,
special predictors, scaling and predictor-weight optimization all use this
earlier history. Matching windows are intersected with that history; a required
window with no observations produces a failed fold, with no substitute predictor
or donor pool. Precomputed predictor columns must themselves be free of future
information. The actual treatment date, outcome and ATT periods remain fixed,
and the final estimator is independently refitted on its original full pre-period.

One window gives a single holdout. For rolling-origin checks, supply several
disjoint windows, for example ``[range(1980, 1984), range(1984, 1989)]``. Later
folds may train on earlier folds' validation observations. Each window must use
observed pre-treatment periods and have earlier training observations. All designs
are scored on the same periods. The pooled RMSE gives each held-out period equal
weight, including when windows have different lengths. A failed fold makes the
aggregate score unavailable; the runner never averages only successful folds.

``validation`` has one row per design and fold, with RMSE, training/validation
counts, status and any exception. ``validation_paths``, ``validation_weights``
and ``validation_warnings`` preserve predictions, donor weights and diagnostics.
``n_train`` counts available earlier periods; each design's actual matching
windows remain in its validation registry.
``validation_registry[spec_id][fold_id]`` records the effective matching windows,
including empty windows that fail; ``validation_fits`` preserves native fits
when available. The original registry and baseline remain accessible. Fold IDs
follow the chronological order stored in ``metadata['validation_windows']``.

``plot_validation()`` compares the final ATT with pooled validation RMSE, labels
designs and names attempts lacking either quantity. It draws a declared cutoff
without hiding other designs. This diagnoses predictive sensitivity; good
pre-treatment forecasts alone do not establish the identifying assumptions.

API
---

.. automodule:: pysyncon.sensitivity
   :members:
