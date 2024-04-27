import unittest
import numpy as np
import pandas as pd

from pysyncon.generator import LinearFactorModel


class TestLinearFactorModel(unittest.TestCase):
    def setUp(self):
        self.n_units = np.random.randint(low=10, high=20)
        self.n_observable = np.random.randint(low=10, high=15)
        self.n_unobservable = np.random.randint(low=10, high=15)
        self.n_periods_pre = np.random.randint(low=50, high=80)
        self.n_periods_post = np.random.randint(low=10, high=20)

    def test_matrix_dims(self):
        lfm = LinearFactorModel()
        X0, X1, Z0, Z1 = lfm.generate(
            n_units=self.n_units,
            n_observable=self.n_observable,
            n_unobservable=self.n_unobservable,
            n_periods_pre=self.n_periods_pre,
            n_periods_post=self.n_periods_post,
        )

        self.assertEqual(X0.shape, (self.n_observable, self.n_units - 1))
        self.assertEqual(X1.shape, (self.n_observable,))
        self.assertEqual(
            Z0.shape, (self.n_periods_pre + self.n_periods_post, self.n_units - 1)
        )
        self.assertEqual(Z1.shape, (self.n_periods_pre + self.n_periods_post,))
