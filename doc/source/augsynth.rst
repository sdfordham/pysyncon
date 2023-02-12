
Augmented Synthetic Control Method
==================================

The augmented synthetic control method is due to
`Ben-Michael, Feller & Rothstein <http://doi.org/10.3386/w28885>`_.
This method constructs a vector of weights :math:`w_ = (w_1, w_2, \dots, w_k)`
such that

.. math::
   w = w_\mathrm{scm} + w_\mathrm{aug},

where :math:`w_\mathrm{scm}` are the weights obtained from the standard
synthetic control method and :math:`w_\mathrm{aug}` are augmentations that are
included when the treated unit lies outside the convex hull defined by the
control units. The weights may be negative and larger than 1, the degree of
extrapolation is controlled by the ridge parameter :math:`\lambda`.

In general, this method will obtain weights at least as good as the synthetic
control method in terms of pre-treatment fit.

The :class:`AugSynth` class
***************************

The :class:`AugSynth <pysyncon.AugSynth>` class implements the augmented synthetic
control method. The expected way to use the class is to first create a
:class:`Dataprep <pysyncon.Dataprep>` object that defines the study data and
then use it as input to a :class:`AugSynth <pysyncon.Synth>` object. See the
`examples folder <https://github.com/sdfordham/pysyncon/tree/main/examples>`_
of the repository for examples illustrating usage.

The implementation is based on the same method in the R
`augsynth package <https://github.com/ebenmichael/augsynth>`_
and aims to produce results that can be reconciled with that package.

.. autoclass:: pysyncon.AugSynth
   :members:
   