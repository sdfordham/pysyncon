pysyncon
========

pysyncon is a Python package that provides methods for the synthetic control
method and derivative methods.

The types of synthetic control studies available in the package are:

.. toctree::
   :maxdepth: 1

   Synthetic Control Method <synth>
   Augmented Synthetic Control Method <augsynth>
   Robust Synthetic Control Method <robust>
   Penalized Synthetic Control Method <penalized>

The package also provides a method for performing permutation tests/placebo
tests with the above methods:

.. toctree::
   :maxdepth: 1

   Placebo Tests <placebo>

The main helper class that is used to describe the study data and used as
input to a synthetic control method is the dataprep class:

.. toctree::
   :maxdepth: 1

   Dataprep <dataprep>

How to use the package
**********************

There are notebooks in the examples folder of the package illustrating how
to use the package `here <https://github.com/sdfordham/pysyncon/tree/main/examples>`_.
