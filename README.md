

# pysyncon ![](https://img.shields.io/badge/python-3.8+-blue.svg) [![codecov](https://codecov.io/gh/sdfordham/pysyncon/graph/badge.svg?token=hmi7xHQ4OT)](https://codecov.io/gh/sdfordham/pysyncon)

A python module for the synthetic control method that provides implementations of:

- Synthetic Control Method (Abadie & Gardeazabal 2003)
- Robust Synthetic Control Method (Amjad, Shah & Shen 2018)
- Augmented Synthetic Control Method (Ben-Michael, Feller & Rothstein 2021)
- Penalized Synthetic Control Method (Abadie & L'Hour 2021)

The package also provides methods for performing placebo tests and generating confidence intervals.

The implementation of the synthetic control method aims to be reconcilable with the R package [Synth](https://CRAN.R-project.org/package=Synth) and similarly the implementation of the Augmented synthetic control method and the R package [augsynth](https://github.com/ebenmichael/augsynth).

## Installation
Install it from PyPI using pip:

````bash
python -m pip install pysyncon
````

## Usage

Documentation is available on [github-pages](https://sdfordham.github.io/pysyncon/). In the examples folder are notebooks reproducing the weights/ATT from:

- The Economic Costs of Conflict: A Case Study of the Basque Country, Alberto Abadie and Javier Gardeazabal; The American Economic Review Vol. 93, No. 1 (Mar., 2003), pp. 113-132. ([notebook here](examples/basque.ipynb))
- The Kansas income-tax cuts example from The Augmented Synthetic Control Method, Eli Ben-Michael, Avi Feller & Jesse Rothstein; Journal of the American Statistical Association Vol. 116, No. 536 (2021), pp. 1789-1803. ([notebook here](examples/augsynth/kansas.ipynb))
- The worked example 'Prison construction and Black male incarceration' from the last chapter of the first edition of 'Causal Inference: The Mixtape' by Scott Cunningham. ([notebook here](examples/texas.ipynb))
- Comparative Politics and the Synthetic Control Method, Alberto Abadie, Alexis Diamond and Jens Hainmueller; American Journal of Political Science Vol. 59, No. 2 (April 2015), pp. 495-510. ([notebook here](examples/germany.ipynb))

### Reproducing the results from the papers

The optimisers used (scipy's Nelder-Mead and SLSQP) are sensitive to the BLAS
backend and to the installed numpy/scipy versions, so computed weights and
estimates can vary slightly across environments — typically in the third
decimal place of the weights and a few percent of point estimates.
Reproducing the published values to full precision requires the reference
environment that the notebook-parity tests ([`parity_tests/`](parity_tests/))
are verified against on CI:

- Python 3.9
- numpy 2.0.2
- scipy 1.13.1
- pandas 2.3.3
- matplotlib 3.9.4
- ubuntu-22.04

These are pinned in the parity job ([`.github/workflows/parity.yml`](.github/workflows/parity.yml)).
Even with these versions, results on other hardware may still differ
slightly, as the underlying BLAS libraries dispatch on the CPU.

## Citation

If you use this package in your research, you can cite it as below.

```
@software{pysyncon,
  author = {Fordham, Stiofán},
  month = dec,
  title = {{pysyncon: a Python package for the Synthetic Control Method}},
  url = {https://github.com/sdfordham/pysyncon},
  year = {2022}
}
```
