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
        self.X0 = pd.DataFrame(
            data=self.rng.random(size=(4, 10)),
            index=range(1, 5),
            columns=range(1, 11),
        )
        self.X1 = pd.Series(data=self.rng.random(size=(4,)), index=range(1, 5), name=0)
        self.pre_periods = list(range(1, 21))
        self.post_periods = list(range(21, 31))
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

    def test_step_sz_tol(self):
        kwargs = {
            "alpha": self.alpha,
            "scm": self.scm,
            "Z0": self.Z0,
            "Z1": self.Z1,
            "pre_periods": self.pre_periods,
            "post_periods": self.post_periods,
            "max_iter": self.max_iter,
            "step_sz_div": self.step_sz_div,
            "verbose": self.verbose,
        }

        conformal_inf = ConformalInference()

        # Step-size is less than tolerance
        self.assertRaises(
            ValueError,
            conformal_inf.confidence_intervals,
            tol=1.0,
            step_sz=0.1,
            **kwargs
        )

        # Step-size = tolerance
        self.assertRaises(
            ValueError,
            conformal_inf.confidence_intervals,
            tol=1.0,
            step_sz=1.0,
            **kwargs
        )

    def test_step_sz_guessing(self):
        kwargs = {
            "alpha": self.alpha,
            "scm": self.scm,
            "Z0": self.Z0,
            "Z1": self.Z1,
            "pre_periods": self.pre_periods,
            "post_periods": self.post_periods,
            "max_iter": self.max_iter,
            "step_sz_div": self.step_sz_div,
            "verbose": self.verbose,
            "scm_fit_args": {"X0": self.X0, "X1": self.X1},
        }

        conformal_inf = ConformalInference()

        # No step-size and a big tolerance
        # (step-size guessing)
        _, n_c = self.Z0.shape
        self.scm.W = np.full(n_c, 1.0 / n_c)
        conformal_inf.confidence_intervals(tol=100.0, **kwargs)
        self.scm.W = None

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

    def test_no_weights(self):
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
        self.assertRaises(ValueError, conformal_inf.confidence_intervals, **kwargs)

    def test_root_search(self):
        cases_roots_x0 = [
            ((-1, 3), 0.5),
            ((-1, 3), 1.0),
            ((-1, 3), 2.5),
            ((-1, 400), 0.5),
            ((-1, 400), 100),
            ((-1, 400), 399),
        ]
        cases_step_sz = [0.1, 1.0]

        ci = ConformalInference()
        tol = 0.01
        for case_root_x0 in cases_roots_x0:
            for case_step_sz in cases_step_sz:
                case = (case_root_x0, case_step_sz)
                with self.subTest(case=case):
                    ((lower, upper), x0) = case_root_x0

                    res = ci._root_search(
                        fn=lambda x: (lower - x) * (x - upper),
                        x0=x0,
                        direction=-1,
                        tol=tol,
                        step_sz=case_step_sz,
                        max_iter=100,
                    )
                    self.assertAlmostEqual(lower, res, delta=tol)

                    res = ci._root_search(
                        fn=lambda x: (lower - x) * (x - upper),
                        x0=x0,
                        direction=1,
                        tol=tol,
                        step_sz=case_step_sz,
                        max_iter=100,
                    )
                    self.assertAlmostEqual(upper, res, delta=tol)

        self.assertRaises(
            Exception,
            ci._root_search,
            fn=lambda x: (-1 - x) * (x - 400),
            x0=200,
            direction=-1,
            tol=0.01,
            step_sz=1.0,
            max_iter=1,
        )
