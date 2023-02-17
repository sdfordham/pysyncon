Synthetic Control Method
========================

The synthetic control method is due to
`Abadie, Diamond & Hainmueller <http://dx.doi.org/10.1198/jasa.2009.ap08746>`_.
This method constructs a vector of non-negative weights
:math:`w = (w_1, w_2, \dots, w_k)` whose sum is 1 and :math:`k` is the number
of control units that minimizes

.. math::
   \|x_1-X_0w^T\|_V,

where

   - :math:`\|A\|_V=\sqrt{A^TVA}`, where :math:`V` is a diagonal matrix
     with non-negative entries that captures the relationship between the
     outcome variable and the predictors,
   - :math:`X_0` is a matrix of the values for the control units of the chosen
     statistic for the chosen predictors over the selected (pre-intervention)
     time-period (each column corresponds to a control),
   - :math:`x_1` is a (column) vector of the corresponding values for the
     treated unit.

The matrix :math:`V` can be supplied otherwise it is part of the
optimization problem: it is obtained by minimizing the quantity
   
.. math::
   \|z_1-Z_0w^T\|,
   
where

   - :math:`Z_0` is a matrix of the values of the outcome variable for the
     control units over the (pre-intervention) time-period (each column
     corresponds to a control),
   - :math:`z_1` is a (column) vector of the corresponding values for the
     treated unit.

The :class:`Synth` class
************************

The :class:`Synth <pysyncon.Synth>` class implements the synthetic control
method. The expected way to use the class is to first create a
:class:`Dataprep <pysyncon.Dataprep>` object that defines the study data and
then use it as input to a :class:`Synth <pysyncon.Synth>` object. See the
`examples folder <https://github.com/sdfordham/pysyncon/tree/main/examples>`_
of the repository for examples illustrating usage.

The implementation is based on the same method in the R
`Synth package <https://cran.r-project.org/web/packages/Synth/index.html>`_
and aims to produce results that can be reconciled with that package.

.. autoclass:: pysyncon.Synth
   :members:
   :inherited-members:
