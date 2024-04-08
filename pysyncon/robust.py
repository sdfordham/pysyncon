from __future__ import annotations
from typing import Optional

import numpy as np

from .dataprep import Dataprep
from .base import BaseSynth


class RobustSynth(BaseSynth):
    """Implementation of the robust synthetic control method due to
    Amjad, Shah & Shen :cite:`robust2018`.
    """

    def __init__(self) -> None:
        super().__init__()
        self.W: Optional[np.ndarray] = None
        self.lambda_: Optional[float] = None

    def fit(
        self,
        dataprep: Dataprep,
        lambda_: float,
        threshold: Optional[float] = None,
        sv_count: Optional[int] = None,
    ) -> None:
        """Fit the model/calculate the weights.

        Parameters
        ----------
        dataprep : Dataprep
            :class:`Dataprep` object containing data to model.
        lambda_ : float
            Ridge parameter to use.
        threshold: float, optional
            Remove singular values that are less than this threshold.
        sv_count: int, optional
            Keep this many of the largest singular values when
            reducing the outcome matrix
        """
        if (
            isinstance(dataprep.treatment_identifier, (list, tuple))
            and len(dataprep.treatment_identifier) > 1
        ):
            raise ValueError("RobustSynth requires exactly one treated unit.")
        self.dataprep = dataprep
        time_period_min = dataprep.foo[dataprep.time_variable].astype("int").min()
        time_period_max = dataprep.foo[dataprep.time_variable].astype("int").max()

        X0, X1 = dataprep.make_outcome_mats(
            time_period=range(time_period_min, 1 + time_period_max)
        )
        Y = X0.T.values

        M_hat = self._sv_decomposition(Y, threshold, sv_count).T

        time_optim_end = 1 + dataprep.time_optimize_ssr[-1]
        end_idx = X0.index.to_list().index(time_optim_end)
        M_hat_neg = M_hat[:end_idx, :]
        Y1_neg = X1.to_numpy()[:end_idx]

        self.W = np.matmul(
            np.linalg.inv(
                M_hat_neg.T @ M_hat_neg + lambda_ * np.identity(M_hat_neg.shape[1])
            ),
            M_hat_neg.T @ Y1_neg,
        )

    def _sv_decomposition(
        self,
        Y: np.ndarray,
        threshold: Optional[float] = None,
        sv_count: Optional[int] = None,
    ) -> np.ndarray:
        """Calculate the :math:`\hat{M}` matrix from the paper (see :cite:`robust2018`) by
        carrying out an SVD of the outcome matrix and remove the specified number
        of singular values.

        Parameters
        ----------
        Y : np.ndarray
            The outcome matrix (:math:`Y` matrix in the notation of the paper).
        threshold : Optional[float], optional
            Remove singular values that are less that `threshold`,
            either this must be specified or `sv_count`, by default None
        sv_count : Optional[int], optional
            Keep this many of the largest singular values,
            either this must be specified or `threshold`, by default None

        Returns
        -------
        np.ndarray
            The :math:`\hat{M}` matrix from the paper (see :cite:`robust2018`).

        Raises
        ------
        ValueError
            If neither `threshold` nor `sv_count` are supplied.

        :meta private:
        """
        if not threshold and not sv_count:
            raise ValueError("One of `threshold` or `sv_count` must be supplied.")

        if threshold:
            idx = 0
            while s[idx] > threshold:
                idx += 1
        else:
            idx = sv_count

        u, s, v = np.linalg.svd(Y)
        s_res = np.zeros_like(Y)
        s_res[:idx, :idx] = np.diag(s[:idx])

        r, c = Y.shape
        p_hat = max(np.count_nonzero(Y) / (r * c), 1 / (r * c))
        M_hat = (1 / p_hat) * (u @ s_res @ v)
        return M_hat
