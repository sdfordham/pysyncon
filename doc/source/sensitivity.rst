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
unit and is separate from placebo inference. Pre-treatment validation and
other estimators are outside this initial interface.

API
---

.. automodule:: pysyncon.sensitivity
   :members:
