Synthetic Control Method
========================

The synthetic control method is due to
`Abadie, Diamond & Hainmueller <http://dx.doi.org/10.1198/jasa.2009.ap08746>`_.
This method constructs a vector of non-negative weights
:math:`w = (w_1, w_2, \dots, w_k)` whose sum is 1 and :math:`k` is the number
of control units that minimises

.. math::
   \|x_1-X_0w\|_V,

where

   - :math:`\|A\|_V=\sqrt{A^TVA}`, where :math:`V` is a diagonal matrix
     with non-negative entries that captures the relationship between the
     outcome variable and the predictors,
   - :math:`X_0` is a matrix of the values for the control units of the chosen
     statistic for the chosen predictors over the selected (pre-intervention)
     time-period (each column corresponds to a control),
   - :math:`x_1` is a vector of the corresponding values for the treated unit.

The matrix :math:`V` can be supplied otherwise it is part of the
optimisation problem: it is obtained by minimising the quantity
   
.. math::
   \|z_1-Z_0\operatorname{diag}(V)^T\|,
   
where

   - :math:`Z_0` is a matrix of the values of the outcome variable for the
     control units over the (pre-intervention) time-period (each column
     corresponds to a control),
   - :math:`z_1` is a vector of the corresponding values for the treated unit.

The total minimisation problem is then solved for :math:`(V,W)` (note that for
a given :math:`V`, minimising for :math:`W` is a quadratic programming problem
and thus has a global minimum).

.. autoclass:: pysyncon.Synth
   :members:
