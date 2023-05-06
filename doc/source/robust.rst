Robust Synthetic Control Method
===============================

The robust synthetic control method is due to
`Amjad, Shah & Shen <https://www.jmlr.org/papers/volume19/17-777/17-777.pdf>`_.

This method denoises the data matrix of the control units by
applying a threshold to the singular values of the matrix.

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