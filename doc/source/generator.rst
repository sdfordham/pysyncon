Sample data generation
======================

The package provides a method for generating fake data for testing purposes.

Linear Factor model
*******************

Let :math:`Y_{it}^N` (resp. :math:`Y_{it}^I`) denote the outcome for unit :math:`i` at time :math:`t`
in the absence of treatment (resp. in the presence of treatment). The :class:`LinearFactorModel`
generates sample potential outcomes data according to a Linear
Factor model:

.. math::

    Y_{jt}^N &= \theta_t^T Z_j + \lambda_t^T \mu_j + \epsilon_{tj},\\
    Y_{jt}^I &= Y_{jt}^N + \delta_t,

where :math:`Z_j` denotes a vector of observable covariates, :math:`\mu_j` is a vector of unobservable
covariates and :math:`\epsilon_{tj}` are mean-zero normal shocks. The vector :math:`\delta_t` denotes
a vector of treatment effects and the remaining variables are model parameters.

.. autoclass:: pysyncon.generator.LinearFactorModel
   :members:
   :inherited-members: