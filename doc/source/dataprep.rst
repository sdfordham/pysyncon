:class:`Dataprep` class
========================

This class and its API are based on the similarly named function in the R
`Synth package <https://cran.r-project.org/web/packages/Synth/index.html>`_.

The ``dataprep`` class defines all the information necessary for the synthetic
control study. It takes in as argument a ``pandas.DataFrame`` `foo` containing
the panel data, a list of predictors, special predictors, the statistical operation to
apply to the predictors over the selected time frame, the dependant variable,
the columns denoting the unit labels, the label denoting the control units,
the label denoting the treated unit, the time period to carry out the optimisation
procedure over and the time period to apply the statistical operation to the
predictors. See below for further details about each individual argument, and also see
the `examples folder <https://github.com/sdfordham/pysyncon/tree/main/examples>`_
of the repository to see how this class is set up in three real research contexts.

The principal difference between the function signature here and the one in
the ``R`` ``synth`` package is that whereas there are two arguments `unit.variable`
and `unit.names.variable` in that package, in this package these are
consolidated into one argument `unit_variable` as here it is unnecessary to have
both.

.. autoclass:: pysyncon.Dataprep
   :members:
