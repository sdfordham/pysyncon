Robust Synthetic Control Method
===============================

The Robust Synthetic Control Method is due to Amjad, Shah & Shen :cite:`robust2018`.

This method de-noises the data matrix of the control units by
applying a threshold to the singular values of the observation matrix
and then fits a linear model using ridge regression of the de-noised control
post-treatment outcomes against pre-treatment outcomes. Similarly to the
:doc:`Ridge Agumented Synthetic Control Method <augsynth>` the weights here
may be negative or larger than 1.

The :class:`RobustSynth` class
******************************

The :class:`RobustSynth <pysyncon.RobustSynth>` class implements the robust control
method. The expected way to use the class is to first create a
:class:`Dataprep <pysyncon.Dataprep>` object that defines the study data and
then use it as input to a :class:`RobustSynth <pysyncon.RobustSynth>` object. See the
`examples folder <https://github.com/sdfordham/pysyncon/tree/main/examples>`_
of the repository for examples illustrating usage.

.. autoclass:: pysyncon.RobustSynth
   :members:
   :inherited-members:
