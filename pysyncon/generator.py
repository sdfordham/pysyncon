from __future__ import annotations
from typing import Optional

import numpy as np
import pandas as pd


class LinearFactorModel:
    """Generates potential outcomes following a linear factor model"""

    def __init__(
        self,
        observed_dist: tuple[int] = (0, 1),
        observed_params_dist: tuple[int] = (0, 10),
        unobserved_dist: tuple[int] = (0, 1),
        unobserved_params_dist: tuple[int] = (0, 10),
        effect_dist: tuple[int] = (0, 20),
        shocks_dist: tuple[int] = (0, 1),
        seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        """Generates potential outcomes following a linear factor model

        Parameters
        ----------
        observed_dist : tuple, optional
            Parameters for the uniform distribution that the observed
            covariates follow, by default (0, 1)
        observed_params_dist : tuple, optional
            Parameters for the uniform distribution that the observed
            covariates model parameters follow, by default (0, 10)
        unobserved_dist : tuple, optional
            Parameters for the uniform distribution that the unobserved
            covariates follow, by default (0, 1)
        unobserved_params_dist : tuple, optional
            Parameters for the uniform distribution that the unobserved
            covariates model parameters follow, by default (0, 10)
        effect_dist : tuple, optional
            Uniform distribution parameters that the effect follows,
            by default (0, 20)
        shocks_dist : tuple, optional
            Normal distribution parameters that the shocks follow, by default (0, 1)
        seed : int, optional
            Random number generator seed, by default None
        rng : numpy.random.Generator, optional
            Provide a numpy random number generator, by default None
        """
        self.observed_dist = observed_dist
        self.observed_params_dist = observed_params_dist
        self.unobserved_dist = unobserved_dist
        self.unobserved_params_dist = unobserved_params_dist
        self.effect_dist = effect_dist
        self.shocks_dist = shocks_dist
        self.seed = seed
        self.rng = rng

    def generate(
        self,
        n_units: int,
        n_observable: int,
        n_unobservable: int,
        n_periods_pre: int,
        n_periods_post: int,
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Generate the matrices (:math:`X_0`, :math:`X_1`, :math:`Z_0`,
        :math:`Z_1`) that can be used as input to a synthetic control
        method (using the notation of Abadie & Gardeazabal :cite:`basque2003`).

        Parameters
        ----------
        n_units : int
            Number of units in the model
        n_observable : int
            Number of observable covariates in the model
        n_unobservable : int
            Number of unobservable covariates in the model
        n_periods_pre : int
            Number of time periods prior to the intervention
        n_periods_post : int
            Number of time periods post the intervention

        Returns
        -------
        tuple[pandas.DataFrame, pandas.Series, pandas.DataFrame, pandas.Series]
            Returns a tuple of 4 pandas objects: :math:`X_0` a pandas DataFrame
            of shape (n_periods_pre + n_periods_post, n_units - 1), :math:`X_1` a
            pandas Series of shape (n_periods_pre + n_periods_post, 1), :math:`Z_0`
            a pandas DataFrame of shape (n_observable, n_units - 1), :math:`Z_1`
            a pandas Series of shape (n_observable, 1).
        """
        rng = self.rng(self.seed) if self.rng else np.random.default_rng(seed=self.seed)

        n_periods = n_periods_pre + n_periods_post

        delta = rng.uniform(*self.effect_dist, size=n_periods).reshape(-1, 1)
        delta = np.column_stack([delta] * n_units)

        Z = rng.uniform(*self.observed_dist, size=(n_observable, n_units))
        mu = rng.uniform(*self.unobserved_dist, size=(n_unobservable, n_units))
        theta = rng.uniform(*self.observed_params_dist, size=(n_observable, n_periods))
        lambda_ = rng.uniform(
            *self.unobserved_params_dist, size=(n_unobservable, n_periods)
        )
        epsilon = rng.normal(*self.shocks_dist, size=(n_periods, n_units))

        Y_N = theta.T @ Z + lambda_.T @ mu + epsilon
        Y_I = Y_N + delta

        X0 = pd.DataFrame(
            data=Z[:, 1:],
            columns=range(2, n_units + 1),
            index=[f"observable{i}" for i in range(1, n_observable + 1)],
        )
        X1 = pd.Series(
            data=Z[:, 0],
            name=1,
            index=[f"observable{i}" for i in range(1, n_observable + 1)],
        )
        Z0 = pd.DataFrame(
            data=Y_N[:, 1:],
            columns=range(2, n_units + 1),
            index=range(1, n_periods + 1),
        )
        Z1 = pd.Series(
            data=np.concatenate(
                [Y_N[:n_periods_pre, 0], Y_I[n_periods_pre:, 0]], axis=0
            ),
            name=1,
            index=range(1, n_periods + 1),
        )
        return X0, X1, Z0, Z1
