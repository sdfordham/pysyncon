

# pysyncon ![](https://img.shields.io/badge/python-3.8+-blue.svg) [![codecov](https://codecov.io/gh/sdfordham/pysyncon/graph/badge.svg?token=hmi7xHQ4OT)](https://codecov.io/gh/sdfordham/pysyncon)

A python module for the synthetic control method that provides implementations of:

- Synthetic Control Method (Abadie & Gardeazabal 2003)
- Robust Synthetic Control Method (Amjad, Shah & Shen 2018)
- Augmented Synthetic Control Method (Ben-Michael, Feller & Rothstein 2021)
- Penalized Synthetic Control Method (Abadie & L'Hour 2021)

The package also provides methods for performing placebo tests with the above.

The implementations of the Synthetic Control method aims to be reconcilable with the R package [Synth](https://CRAN.R-project.org/package=Synth) and similarly the implementation of the Augmented Synthetic Control method and the R package [augsynth](https://github.com/ebenmichael/augsynth).

## Installation
Install it from PyPI using pip:

````bash
python -m pip install pysyncon
````

## Usage

Documentation is available on [github-pages](https://sdfordham.github.io/pysyncon/). In the examples folder are notebooks reproducing the weights from:

- The Economic Costs of Conflict: A Case Study of the Basque Country, Alberto Abadie and Javier Gardeazabal; The American Economic Review Vol. 93, No. 1 (Mar., 2003), pp. 113-132. ([notebook here](examples/basque.ipynb))
- The worked example 'Prison construction and Black male incarceration' from the last chapter of 'Causal Inference: The Mixtape' by Scott Cunningham. ([notebook here](examples/texas.ipynb))
- Comparative Politics and the Synthetic Control Method, Alberto Abadie, Alexis Diamond and Jens Hainmueller; American Journal of Political Science Vol. 59, No. 2 (April 2015), pp. 495-510. ([notebook here](examples/germany.ipynb))
