from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .dataprep import Dataprep, IsinArg_t
from .synth import WeightOptimizerMixin


class AugSynth(WeightOptimizerMixin):
    def __init__(self) -> None:
        self.dataprep: Optional[Dataprep] = None
        self.W: Optional[np.ndarray] = None

    def fit(
        self,
        dataprep: Optional[Dataprep],
        ridge_param : float = 0.001
    ) -> None:
        self.dataprep = dataprep
        # Follow the paper variable names for now...

        # X <- Z0[sorted(Z0.columns)].T
        # y <- same as X except for post treatment years
        # Z <- X0[sorted(X0.columns)].T
        X0, X1 = dataprep.make_covariate_mats()
        Z0, Z1 = dataprep.make_outcome_mats()

        X_c = Z0.subtract(Z0.mean(axis=1), axis=0) # X_cent
        X_1 = Z1.subtract(Z0.mean(axis=1), axis=0) # X_1

        Z_c = X0.subtract(X0.mean(axis=1), axis=0) # Z_c <- Z_c[sorted(Z_c.columns)].T
        Z_1 = X1.subtract(X0.mean(axis=1), axis=0)

        Z_c_std = Z_c.std(axis=1)
        X_c_std = X_c.to_numpy().std(ddof=1).item()

        Z_c_normal = Z_c.divide(Z_c_std, axis=0) * X_c_std  # Z_c after "standardize covariates"
        Z_1_normal = Z_1.divide(Z_c_std, axis=0) * X_c_std  # Z_1 after "standardize covariates"

        X_c_stacked = pd.concat([X_c, Z_c_normal], axis=0)  # X_c after "concatenate"
        X_1_stacked = pd.concat([X_1, Z_1_normal], axis=0)  # X_1 after "concatenate"

        V_mat = np.diag([1. / Z0.shape[0]] * Z0.shape[0])
        print(Z0.shape, V_mat.shape, Z1.shape)
        W, _, _ = self.w_optimize(V_mat=V_mat, X0=Z0.to_numpy(), X1=Z1.to_numpy(), Z0=Z0.to_numpy(), Z1=Z1.to_numpy())

        W_ridge = self.solve_ridge(X_1_stacked.to_numpy(), X_c_stacked.to_numpy(), W, ridge_param)
        self.W = W + W_ridge

    def solve_ridge(self, A, B, W, lmbda):
        M = (A - B @ W)
        N = np.linalg.inv(B @ B.T + lmbda * np.identity(B.shape[0]))
        return M @ (N @ B)

    def path_plot(
        self,
        time_period: Optional[IsinArg_t] = None,
        treatment_time: Optional[int] = None,
        grid: bool = True,
    ) -> None:
        if self.W is None:
            raise ValueError("No weight matrix available; fit data first.")
        if self.dataprep is None:
            raise ValueError("dataprep must be set for automatic plots.")

        Z0, Z1 = self.dataprep.make_outcome_mats(time_period=time_period)
        ts_synthetic = (Z0 * self.W).sum(axis=1).rename("Synthetic")
        Z1.plot(ylabel=self.dataprep.dependent, color="black", linewidth=1)
        ts_synthetic.plot(
            ylabel=self.dataprep.dependent,
            color="black",
            linewidth=1,
            linestyle="dashed",
        )

        if treatment_time:
            plt.axvline(x=treatment_time, ymin=0.05, ymax=0.95, linestyle="dashed")
        plt.legend()
        plt.grid(grid)
        plt.show()

    def gaps_plot(
        self,
        time_period: Optional[IsinArg_t] = None,
        treatment_time: Optional[int] = None,
        grid: bool = True,
    ) -> None:
        if self.dataprep is None:
            raise ValueError("dataprep must be set for automatic plots.")
        if self.W is None:
            raise ValueError("No weight matrix available; fit data first.")

        Z0, Z1 = self.dataprep.make_outcome_mats(time_period=time_period)
        ts_synthetic = (Z0 * self.W).sum(axis=1)
        ts_gap = Z1 - ts_synthetic
        ts_gap.plot(ylabel=self.dataprep.dependent, color="black", linewidth=1)

        plt.hlines(
            y=0,
            xmin=min(ts_gap.index),
            xmax=max(ts_gap.index),
            color="black",
            linestyle="dashed",
        )
        if treatment_time:
            plt.axvline(x=treatment_time, ymin=0.05, ymax=0.95, linestyle="dashed")
        plt.grid(grid)
        plt.show()

    def weights(self, threshold: float = 0.0, round: int = 3) -> pd.Series:
        if self.W is None:
            raise ValueError("No weight matrix available; fit data first.")
        if self.dataprep is None:
            weights_ser = pd.Series(self.W, name="weights")
        else:
            weights_ser = pd.Series(
                self.W, index=list(self.dataprep.controls_identifier), name="weights"
            )
        return weights_ser[weights_ser >= threshold].round(round)
