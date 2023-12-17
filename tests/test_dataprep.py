import unittest
import numpy as np
import pandas as pd

from pysyncon import Dataprep


class TestDataprep(unittest.TestCase):
    def setUp(self):
        # 1 -> treated, (2, 3) -> controls
        self.foo = pd.DataFrame(
            {
                "time": [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
                "name": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
                "dependent": np.random.random(12),
                "predictor1": np.random.random(12),
                "predictor2": np.random.random(12),
            }
        )
        self.predictors = ["predictor1"]
        self.predictors_op = "mean"
        self.dependent = "dependent"
        self.unit_variable = "name"
        self.time_variable = "time"
        self.treatment_identifier = 1
        self.controls_identifier = [2, 3]
        self.time_predictors_prior = [2, 3]
        self.time_optimize_ssr = [1, 2, 3]
        self.special_predictors = [
            ("predictor1", [2], "mean"),
            ("predictor2", [1, 2], "median"),
            ("predictor2", [1, 2], "std"),
        ]
        self.str_no_special = (
            "Dataprep\n"
            "Treated unit: 1\n"
            "Dependent variable: dependent\n"
            "Control units: 2, 3\n"
            "Time range in data: 1 - 4\n"
            "Time range for loss minimization: [1, 2, 3]\n"
            "Time range for predictors: [2, 3]\n"
            "Predictors: predictor1\n"
        )
        self.str_special = (
            "Dataprep\n"
            "Treated unit: 1\n"
            "Dependent variable: dependent\n"
            "Control units: 2, 3\n"
            "Time range in data: 1 - 4\n"
            "Time range for loss minimization: [1, 2, 3]\n"
            "Time range for predictors: [2, 3]\n"
            "Predictors: predictor1\n"
            "Special predictors:\n"
            "    `predictor1` over `[2]` using `mean`\n"
            "    `predictor2` over `[1, 2]` using `median`\n"
            "    `predictor2` over `[1, 2]` using `std`\n"
        )

    def test_init_arg_foo(self):
        kwargs = {
            "predictors": self.predictors,
            "predictors_op": self.predictors_op,
            "dependent": self.dependent,
            "unit_variable": self.unit_variable,
            "time_variable": self.time_variable,
            "treatment_identifier": self.treatment_identifier,
            "controls_identifier": self.controls_identifier,
            "time_predictors_prior": self.time_predictors_prior,
            "time_optimize_ssr": self.time_optimize_ssr,
            "special_predictors": self.special_predictors,
        }

        self.assertRaises(TypeError, Dataprep, foo=np.array([]), **kwargs)
        self.assertRaises(TypeError, Dataprep, foo=list(), **kwargs)
        self.assertRaises(TypeError, Dataprep, foo=tuple(), **kwargs)

    def test_init_arg_predictors(self):
        kwargs = {
            "foo": self.foo,
            "predictors_op": self.predictors_op,
            "dependent": self.dependent,
            "unit_variable": self.unit_variable,
            "time_variable": self.time_variable,
            "treatment_identifier": self.treatment_identifier,
            "controls_identifier": self.controls_identifier,
            "time_predictors_prior": self.time_predictors_prior,
            "time_optimize_ssr": self.time_optimize_ssr,
            "special_predictors": self.special_predictors,
        }

        self.assertRaises(ValueError, Dataprep, predictors=["badval"], **kwargs)

    def test_init_arg_predictors_op(self):
        kwargs = {
            "foo": self.foo,
            "predictors": self.predictors,
            "dependent": self.dependent,
            "unit_variable": self.unit_variable,
            "time_variable": self.time_variable,
            "treatment_identifier": self.treatment_identifier,
            "controls_identifier": self.controls_identifier,
            "time_predictors_prior": self.time_predictors_prior,
            "time_optimize_ssr": self.time_optimize_ssr,
            "special_predictors": self.special_predictors,
        }

        self.assertRaises(ValueError, Dataprep, predictors_op="badval", **kwargs)

    def test_init_arg_dependent(self):
        kwargs = {
            "foo": self.foo,
            "predictors": self.predictors,
            "predictors_op": self.predictors_op,
            "unit_variable": self.unit_variable,
            "time_variable": self.time_variable,
            "treatment_identifier": self.treatment_identifier,
            "controls_identifier": self.controls_identifier,
            "time_predictors_prior": self.time_predictors_prior,
            "time_optimize_ssr": self.time_optimize_ssr,
            "special_predictors": self.special_predictors,
        }

        self.assertRaises(ValueError, Dataprep, dependent="badval", **kwargs)

    def test_init_arg_unit_variable(self):
        kwargs = {
            "foo": self.foo,
            "predictors": self.predictors,
            "predictors_op": self.predictors_op,
            "dependent": self.dependent,
            "time_variable": self.time_variable,
            "treatment_identifier": self.treatment_identifier,
            "controls_identifier": self.controls_identifier,
            "time_predictors_prior": self.time_predictors_prior,
            "time_optimize_ssr": self.time_optimize_ssr,
            "special_predictors": self.special_predictors,
        }

        self.assertRaises(ValueError, Dataprep, unit_variable="badval", **kwargs)

    def test_init_arg_time_variable(self):
        kwargs = {
            "foo": self.foo,
            "predictors": self.predictors,
            "predictors_op": self.predictors_op,
            "dependent": self.dependent,
            "unit_variable": self.unit_variable,
            "treatment_identifier": self.treatment_identifier,
            "controls_identifier": self.controls_identifier,
            "time_predictors_prior": self.time_predictors_prior,
            "time_optimize_ssr": self.time_optimize_ssr,
            "special_predictors": self.special_predictors,
        }

        self.assertRaises(ValueError, Dataprep, time_variable="badval", **kwargs)

    def test_init_arg_treatment_identifier(self):
        kwargs = {
            "foo": self.foo,
            "predictors": self.predictors,
            "predictors_op": self.predictors_op,
            "dependent": self.dependent,
            "unit_variable": self.unit_variable,
            "time_variable": self.time_variable,
            "controls_identifier": self.controls_identifier,
            "time_predictors_prior": self.time_predictors_prior,
            "time_optimize_ssr": self.time_optimize_ssr,
            "special_predictors": self.special_predictors,
        }

        self.assertRaises(ValueError, Dataprep, treatment_identifier="badval", **kwargs)

    def test_init_arg_controls_identifier(self):
        kwargs = {
            "foo": self.foo,
            "predictors": self.predictors,
            "predictors_op": self.predictors_op,
            "dependent": self.dependent,
            "unit_variable": self.unit_variable,
            "time_variable": self.time_variable,
            "treatment_identifier": self.treatment_identifier,
            "time_predictors_prior": self.time_predictors_prior,
            "time_optimize_ssr": self.time_optimize_ssr,
            "special_predictors": self.special_predictors,
        }

        self.assertRaises(ValueError, Dataprep, controls_identifier=[1], **kwargs)
        self.assertRaises(ValueError, Dataprep, controls_identifier=[5], **kwargs)

    def test_init_arg_special_predictors(self):
        kwargs = {
            "foo": self.foo,
            "predictors": self.predictors,
            "predictors_op": self.predictors_op,
            "dependent": self.dependent,
            "unit_variable": self.unit_variable,
            "time_variable": self.time_variable,
            "treatment_identifier": self.treatment_identifier,
            "controls_identifier": self.controls_identifier,
            "time_predictors_prior": self.time_predictors_prior,
            "time_optimize_ssr": self.time_optimize_ssr,
        }

        self.assertRaises(
            ValueError, Dataprep, special_predictors=[("predictor1", [1])], **kwargs
        )
        self.assertRaises(
            ValueError,
            Dataprep,
            special_predictors=[("badval", [1], "mean")],
            **kwargs,
        )
        self.assertRaises(
            ValueError,
            Dataprep,
            special_predictors=[("predictor1", [1], "badval")],
            **kwargs,
        )

    def test_make_covariate_mats(self):
        kwargs = {
            "foo": self.foo,
            "predictors": self.predictors,
            "dependent": self.dependent,
            "unit_variable": self.unit_variable,
            "time_variable": self.time_variable,
            "treatment_identifier": self.treatment_identifier,
            "controls_identifier": self.controls_identifier,
            "time_predictors_prior": self.time_predictors_prior,
            "time_optimize_ssr": self.time_optimize_ssr,
            "special_predictors": self.special_predictors,
        }

        # Non-special
        for op in ("mean", "std", "median"):
            dataprep = Dataprep(predictors_op=op, **kwargs)
            X0, X1 = dataprep.make_covariate_mats()

            # Treated
            for predictor in self.predictors:
                mask_treated = (
                    self.foo[self.unit_variable] == self.treatment_identifier
                ) & (self.foo[self.time_variable].isin(self.time_predictors_prior))
                self.assertAlmostEqual(
                    self.foo[mask_treated][predictor].agg(op), X1.loc[predictor]
                )

            # Control
            for control in self.controls_identifier:
                mask_control = self.foo[self.unit_variable] == control
                for predictor in self.predictors:
                    mask = mask_control & self.foo[self.time_variable].isin(
                        self.time_predictors_prior
                    )
                    self.assertAlmostEqual(
                        self.foo[mask][predictor].agg(op), X0.loc[predictor, control]
                    )

        # Special
        dataprep = Dataprep(predictors_op="mean", **kwargs)
        X0, X1 = dataprep.make_covariate_mats()

        # Treated
        for idx, (predictor, time_period, op) in enumerate(self.special_predictors, 1):
            mask_treated = (
                self.foo[self.unit_variable] == self.treatment_identifier
            ) & (self.foo[self.time_variable].isin(time_period))
            column_name = f"special.{idx}.{predictor}"
            self.assertAlmostEqual(
                self.foo[mask_treated][predictor].agg(op), X1.loc[column_name]
            )

        # Control
        for control in self.controls_identifier:
            mask_control = self.foo[self.unit_variable] == control
            for idx, (predictor, time_period, op) in enumerate(
                self.special_predictors, 1
            ):
                mask = mask_control & self.foo[self.time_variable].isin(time_period)
                column_name = f"special.{idx}.{predictor}"
                self.assertAlmostEqual(
                    self.foo[mask][predictor].agg(op), X0.loc[column_name, control]
                )

    def test_make_outcome_mats(self):
        kwargs = {
            "foo": self.foo,
            "predictors": self.predictors,
            "predictors_op": self.predictors_op,
            "dependent": self.dependent,
            "unit_variable": self.unit_variable,
            "time_variable": self.time_variable,
            "treatment_identifier": self.treatment_identifier,
            "controls_identifier": self.controls_identifier,
            "time_predictors_prior": self.time_predictors_prior,
            "time_optimize_ssr": self.time_optimize_ssr,
            "special_predictors": self.special_predictors,
        }

        dataprep = Dataprep(**kwargs)
        # All time
        all_time = dataprep.foo[dataprep.time_variable].unique()
        Z0, Z1 = dataprep.make_outcome_mats(time_period=all_time)

        # Treated
        mask_treated = self.foo[self.unit_variable] == self.treatment_identifier
        pd.testing.assert_series_equal(
            self.foo[mask_treated]
            .set_index(self.time_variable)[self.dependent]
            .rename(self.treatment_identifier),
            Z1,
        )

        # Control
        for control in self.controls_identifier:
            mask = self.foo[self.unit_variable] == control
            pd.testing.assert_series_equal(
                self.foo[mask]
                .set_index(self.time_variable)[self.dependent]
                .rename(control),
                Z0[control],
            )

        # Set time
        Z0, Z1 = dataprep.make_outcome_mats()

        # Treated
        mask_treated = (self.foo[self.unit_variable] == self.treatment_identifier) & (
            self.foo[self.time_variable].isin(self.time_optimize_ssr)
        )
        pd.testing.assert_series_equal(
            self.foo[mask_treated]
            .set_index(self.time_variable)[self.dependent]
            .rename(self.treatment_identifier),
            Z1,
        )

        # Control
        for control in self.controls_identifier:
            mask = (self.foo[self.unit_variable] == control) & (
                self.foo[self.time_variable].isin(self.time_optimize_ssr)
            )
            pd.testing.assert_series_equal(
                self.foo[mask]
                .set_index(self.time_variable)[self.dependent]
                .rename(control),
                Z0[control],
            )

    def test_str(self):
        kwargs_no_special = {
            "foo": self.foo,
            "predictors": self.predictors,
            "predictors_op": self.predictors_op,
            "dependent": self.dependent,
            "unit_variable": self.unit_variable,
            "time_variable": self.time_variable,
            "treatment_identifier": self.treatment_identifier,
            "controls_identifier": self.controls_identifier,
            "time_predictors_prior": self.time_predictors_prior,
            "time_optimize_ssr": self.time_optimize_ssr,
        }
        dataprep = Dataprep(**kwargs_no_special)
        self.assertEqual(self.str_no_special, str(dataprep))

        kwargs_special = {
            "foo": self.foo,
            "predictors": self.predictors,
            "predictors_op": self.predictors_op,
            "dependent": self.dependent,
            "unit_variable": self.unit_variable,
            "time_variable": self.time_variable,
            "treatment_identifier": self.treatment_identifier,
            "controls_identifier": self.controls_identifier,
            "time_predictors_prior": self.time_predictors_prior,
            "time_optimize_ssr": self.time_optimize_ssr,
            "special_predictors": self.special_predictors,
        }
        dataprep = Dataprep(**kwargs_special)
        self.assertEqual(self.str_special, str(dataprep))
