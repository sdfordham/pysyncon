Placebo Tests
=============

A placebo test is used to assess the significance of a synthetic control study
by running the study once for each control unit set as treated unit and the
remaining control units set as controls. See
`Abadie & Gardeazabal <https://www.aeaweb.org/articles?id=10.1257/000282803321455188>`_
(section I.B) for a motivation. An example of usage is in the python notebook
reproducing the weights from that paper in the package repository
`here <https://github.com/sdfordham/pysyncon/tree/main/examples/basque.ipynb>`_

The :class:`PlaceboTest` class
************************

.. autoclass:: pysyncon.utils.PlaceboTest
   :members:
