Synthetic Control Method
========================

The synthetic control method is due to
`Abadie, Diamond & Hainmueller <http://dx.doi.org/10.1198/jasa.2009.ap08746>`_.
This method constructs a vector of non-negative weights
:math:`w = (w_1, w_2, \dots, w_k)` whose sum is 1 and :math:`k` is the number
of control units that minimises

.. math::
   \|x_1-X_0w\|_V

where

   - where :math:`\|A\|_V=\sqrt{A^TVA}`, where :math:`V` is a diagonal matrix
     with non-negative entries that captures the relationship between the
     outcome variable and the predictors,
   - :math:`X_0` is a matrix of the values for the control units of the chosen
     statistic for the chosen predictors over the selected (pre-intervention)
     time-period (each column corresponds to a control),
   - :math:`x_1` is a vector of the corresponding values for the treated unit.

.. autoclass:: pysyncon.Synth
   :members:
