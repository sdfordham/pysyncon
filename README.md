# pyaugsynth

A python module that aims at replicating the methods in the R packages [Synth](https://CRAN.R-project.org/package=Synth) and [augsynth](https://github.com/ebenmichael/augsynth) and attempts to provide as similar an API as possible to the former package.

## Correctness

In the examples folder are notebooks using this module to reproduce the weights from:

- The Economic Costs of Conflict: A Case Study of the Basque Country, Alberto Abadie and Javier Gardeazabal; The American Economic Review Vol. 93, No. 1 (Mar., 2003), pp. 113-132.
- The worked example 'Prison construction and Black male incarceration' from the last chapter of 'Causal Inference: The Mixtape (V. 1.8)' by Scott Cunningham.
- Synthetic Control Methods for Comparative Case Studies: Estimating the Effect of California’s Tobacco Control Program, Alberto Abadie, Alexis Diamond & Jens Hainmueller;  Journal of the American Statistical Association Volume 105, 2010 - Issue 490, pp. 493-505.
- Comparative Politics and the Synthetic Control Method, Alberto Abadie, Alexis Diamond and Jens Hainmueller; American Journal of Political Science Vol. 59, No. 2 (April 2015), pp. 495-510.

## Differences between pyaugsynth and Synth

- By default Synth internally applies both Nelder-Mead & BFGS schemes with two different starting points for the optimisation problem, and then returns the best weights. This module will only apply one optimisation method with one starting point at a time for transparency to the user (just wrap the method here in a ``for`` loop to replicate the Synth approach).

- The R package [rgenoud](https://CRAN.R-project.org/package=rgenoud) provides baked-in methods for a GA based optimisation for starting point search in Synth, no equivalent is provided here, however the main `Synth` class can be used with a GA package thru the `custom_V` option of the `fit` class method.

## Differences between pyaugsynth and augsynth

- By default pyaugsynth will find the optimum value for the ridge parameter using a baked-in cross-validation technique. This approach is not followed here, $\lambda$ may be provided by the user otherwise it is set to a default value, following the custom around hyperparameters in other python packages such as ``Scipy``, ``sklearn`` etc.

