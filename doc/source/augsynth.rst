
Augmented Synthetic Control Method
==================================

The *Augmented Synthetic Control Method* is due to Ben-Michael, Feller & Rothstein
:cite:`augsynth2021` and adapts the :doc:`Synthetic Control Method <synth>` in an
effort to adjust for poor pre-treatment fit.

The authors do this by adjusting the Synthetic Control Method estimate by adding
a term that is an imbalance in a particular function of the pre-treatment outcomes.
In the *Ridge Augmented Synthetic Control Method* this function is
linear in the pre-treatment outcomes and fit by ridge regression of the control
post-treatment outcomes against pre-treatment outcomes.

In particular, the method constructs a vector of weights :math:`w = (w_1, w_2, \dots, w_k)`
such that

.. math::
   w = w_\mathrm{scm} + w_\mathrm{aug},

where :math:`w_\mathrm{scm}` are the weights obtained from the standard
:doc:`Synthetic Control Method <synth>` and :math:`w_\mathrm{aug}` are
augmentations that are included when the treated unit lies outside the
convex hull defined by the control units. The weights may be negative and
larger than 1, the degree of extrapolation is controlled by a ridge
parameter :math:`\lambda`.

In general, this method will obtain weights at least as good as the synthetic
control method in terms of pre-treatment fit.

The :class:`AugSynth` class
***************************

The :class:`AugSynth <pysyncon.AugSynth>` class implements the Ridge Augmented
Synthetic Control Method. The expected way to use the class is to first create a
:class:`Dataprep <pysyncon.Dataprep>` object that defines the study data and
then use it as input to a :class:`AugSynth <pysyncon.AugSynth>` object. See the
`examples folder <https://github.com/sdfordham/pysyncon/tree/main/examples>`_
of the repository for examples illustrating usage.

The implementation follows the same algorithm as the R
`augsynth package <https://github.com/ebenmichael/augsynth>`_ with the option
``progfunc="Ridge"`` and aims to produce results that can be reconciled with
that package, in particular:

- The pre-treatment outcomes are centered at the control unit mean per time
  period. If covariates are included (``use_covariates=True``, the default),
  they are centered at the control unit mean, scaled by the ratio of the
  standard deviation of the centered control outcomes to their own standard
  deviation, and concatenated with the pre-treatment outcomes into a single
  design matrix.
- The synthetic control weights :math:`w_\mathrm{scm}` are obtained from the
  usual quadratic minimization problem (weights that sum to 1 and are
  non-negative) on this design matrix, so that the covariates are balanced
  jointly with the outcomes.
- With ``residualize=True`` (the ``residualize = TRUE`` option of the R
  package), the covariates are not balanced directly: the centered
  pre-treatment outcomes are regressed on the centered (unscaled) covariates
  with OLS on the control units and the residuals form the design matrix;
  after the ridge step the covariates are re-added to the weights exactly, so
  that the final weights balance the covariates exactly. The weights before
  the re-add are stored in ``no_cov_weights``.
- The augmentation :math:`w_\mathrm{aug}` is obtained by ridge regression of
  the imbalance of the synthetic control fit on the design matrix, with the
  ridge parameter :math:`\lambda` controlling the degree of extrapolation.
- When ``lambda_`` is not supplied, it is selected by cross-validation over a
  grid of 21 log-spaced values generated from the largest singular value of
  the design matrix. In each fold a pre-treatment time period is held out and
  the weights are re-fit on the remaining time periods; the cross-validation
  error measures how well the augmented weights predict the treated unit's
  held-out pre-treatment outcomes. The 1-standard-error rule is used to choose
  the final value.
- The pre-treatment periods are exactly the time periods before the first
  post-treatment period, i.e. ``dataprep.time_optimize_ssr`` must contain all
  of the time periods that occur before the treatment time.

In addition to the weights, the fit computes the ``ridge_mhat`` outcome model:
a ridge regression of the post-treatment control outcomes on the design
matrix, using the same :math:`\lambda`. The model is used to estimate the bias
of the augmented weights in each post-treatment period, with
``bias_est = mhat[treated] - synw @ mhat[controls]`` (using the synthetic
control weights) and ``avg_bias`` the average over the post-treatment periods
- reproducing the *Avg Estimated Bias* diagnostic of the R package.

The fit also stores the pre-treatment fit diagnostics of the R package:
``l2_imbalance``, ``unif_l2_imbalance`` and ``scaled_l2_imbalance`` (computed
on the original outcome matrices, so that ``1 - scaled_l2_imbalance`` is the
percentage improvement over uniform weights) and, when covariates are used,
``covariate_l2_imbalance`` and ``scaled_covariate_l2_imbalance``.

.. autoclass:: pysyncon.AugSynth
   :members:
   :inherited-members:
