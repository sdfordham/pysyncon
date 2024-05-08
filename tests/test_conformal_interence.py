import unittest
import pandas as pd
import numpy as np

from pysyncon import Synth
from pysyncon.inference import ConformalInference


class TestConformalInference(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng()
        self.alpha = 0.05
        self.scm = Synth()
        self.Z0 = pd.DataFrame(
            data=self.rng.random(size=(30, 10)),
            index=range(1, 31),
            columns=range(1, 11),
        )
        self.Z1 = pd.Series(
            data=self.rng.random(size=(30,)), index=range(1, 31), name=0
        )
        self.pre_periods = range(1, 21)
        self.post_periods = range(21, 31)
        self.max_iter = 20
        self.tol = 0.1
        self.step_sz = None
        self.step_sz_div = 20.0
        self.verbose = False

    def test_alpha(self):
        kwargs = {
            "scm": self.scm,
            "Z0": self.Z0,
            "Z1": self.Z1,
            "pre_periods": self.pre_periods,
            "post_periods": self.post_periods,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "step_sz": self.step_sz,
            "step_sz_div": self.step_sz_div,
            "verbose": self.verbose,
        }

        conformal_inf = ConformalInference()

        cases = [-1.0, 0.0, 1.0, 17.0]
        for case in cases:
            with self.subTest(case=case):
                self.assertRaises(
                    ValueError, conformal_inf.confidence_intervals, alpha=case, **kwargs
                )

        cases = [True, ["foo"], {"foo": "bar"}]
        for case in cases:
            with self.subTest(case=case):
                self.assertRaises(
                    TypeError, conformal_inf.confidence_intervals, alpha=case, **kwargs
                )

    def test_max_iter(self):
        kwargs = {
            "alpha": self.alpha,
            "scm": self.scm,
            "Z0": self.Z0,
            "Z1": self.Z1,
            "pre_periods": self.pre_periods,
            "post_periods": self.post_periods,
            "tol": self.tol,
            "step_sz": self.step_sz,
            "step_sz_div": self.step_sz_div,
            "verbose": self.verbose,
        }

        conformal_inf = ConformalInference()

        cases = [-17, 0]
        for case in cases:
            with self.subTest(case=case):
                self.assertRaises(
                    ValueError,
                    conformal_inf.confidence_intervals,
                    max_iter=case,
                    **kwargs
                )

        cases = [5.2, 10.0]
        for case in cases:
            with self.subTest(case=case):
                self.assertRaises(
                    TypeError,
                    conformal_inf.confidence_intervals,
                    max_iter=case,
                    **kwargs
                )

    def test_tol(self):
        kwargs = {
            "alpha": self.alpha,
            "scm": self.scm,
            "Z0": self.Z0,
            "Z1": self.Z1,
            "pre_periods": self.pre_periods,
            "post_periods": self.post_periods,
            "max_iter": self.max_iter,
            "step_sz": self.step_sz,
            "step_sz_div": self.step_sz_div,
            "verbose": self.verbose,
        }

        conformal_inf = ConformalInference()

        cases = [-4.2, 0.0]
        for case in cases:
            with self.subTest(case=case):
                self.assertRaises(
                    ValueError, conformal_inf.confidence_intervals, tol=case, **kwargs
                )

        cases = [-4, 0]
        for case in cases:
            with self.subTest(case=case):
                self.assertRaises(
                    TypeError, conformal_inf.confidence_intervals, tol=case, **kwargs
                )

    def test_step_sz(self):
        kwargs = {
            "alpha": self.alpha,
            "scm": self.scm,
            "Z0": self.Z0,
            "Z1": self.Z1,
            "pre_periods": self.pre_periods,
            "post_periods": self.post_periods,
            "tol": self.tol,
            "max_iter": self.max_iter,
            "step_sz_div": self.step_sz_div,
            "verbose": self.verbose,
        }

        conformal_inf = ConformalInference()

        cases = [-4.2, 0.0]
        for case in cases:
            with self.subTest(case=case):
                self.assertRaises(
                    ValueError,
                    conformal_inf.confidence_intervals,
                    step_sz=case,
                    **kwargs
                )

        cases = [-4, 0]
        for case in cases:
            with self.subTest(case=case):
                self.assertRaises(
                    TypeError,
                    conformal_inf.confidence_intervals,
                    step_sz=case,
                    **kwargs
                )

    def test_step_sz_div(self):
        kwargs = {
            "alpha": self.alpha,
            "scm": self.scm,
            "Z0": self.Z0,
            "Z1": self.Z1,
            "pre_periods": self.pre_periods,
            "post_periods": self.post_periods,
            "tol": self.tol,
            "max_iter": self.max_iter,
            "step_sz": self.step_sz,
            "verbose": self.verbose,
        }

        conformal_inf = ConformalInference()

        cases = [-4.2, 0.0]
        for case in cases:
            with self.subTest(case=case):
                self.assertRaises(
                    ValueError,
                    conformal_inf.confidence_intervals,
                    step_sz_div=case,
                    **kwargs
                )

        cases = [-4, 0]
        for case in cases:
            with self.subTest(case=case):
                self.assertRaises(
                    TypeError,
                    conformal_inf.confidence_intervals,
                    step_sz_div=case,
                    **kwargs
                )
