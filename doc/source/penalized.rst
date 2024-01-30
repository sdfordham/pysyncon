Penalized Synthetic Control Method
==================================

The penalized synthetic control method is due to Abadie & L'Hour :cite:`penalized2021`.

This version of the synthetic control method adds a penalization term to the loss
function that has the effect of serving to reduce the interpolation bias. It does this
by penalizing pairwise discrepancies in any unit contributing to the synthetic control
and the treated unit.

The :class:`PenalizedSynth` class
*********************************

The :class:`PenalizedSynth <pysyncon.PenalizedSynth>` class implements the penalized
synthetic control method. The expected way to use the class is to first create a
:class:`Dataprep <pysyncon.Dataprep>` object that defines the study data and
then use it as input to a :class:`PenalizedSynth <pysyncon.RobustSynth>` object. See the
`examples folder <https://github.com/sdfordham/pysyncon/tree/main/examples>`_
of the repository for examples illustrating usage.

.. autoclass:: pysyncon.PenalizedSynth
   :members:
   :inherited-members:
