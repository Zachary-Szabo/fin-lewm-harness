"""Two-state Gaussian HMM with a causal forward pass.

Parameters are estimated on the training returns only: a two-component
``GaussianMixture`` supplies the emission means and variances, and a
Laplace-smoothed count of hard assignments supplies the transition
matrix. The component with the larger variance is the high-vol state.

``update`` runs one step of the forward algorithm. There is no backward
pass and no Viterbi decode, so ``regime_prob`` is the filtered
probability ``P(high-vol | obs_0..t)``, not a smoothed probability.
``surprise`` is the negative log likelihood of ``obs.ret`` under the
one-step predictive mixture. ``pred_vol`` is the square root of the
filtered mixture variance. ``filtered_price`` stays at ``obs.close``
because this model does not estimate a latent price.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import norm
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture

from lewm_harness.filters.base import RegimeFilter
from lewm_harness.types import Bar, FilterState

_MIN_TRAIN = 20


class HMMFilter(RegimeFilter):
    """Causal 2-state Gaussian HMM on simple returns."""

    def __init__(self, random_state: int = 0) -> None:
        self.random_state = random_state
        self.means_: np.ndarray | None = None
        self.stds_: np.ndarray | None = None
        self.transmat_: np.ndarray | None = None
        self.startprob_: np.ndarray | None = None
        self.high_vol_state: int = 1
        self._log_alpha: np.ndarray | None = None
        self._have_posterior = False

    def fit(self, train_df: pd.DataFrame) -> None:
        returns = np.asarray(train_df["ret"].to_numpy(dtype=np.float64), dtype=np.float64)
        returns = returns[np.isfinite(returns)]
        if returns.size < _MIN_TRAIN:
            raise ValueError(f"HMMFilter.fit requires at least {_MIN_TRAIN} finite returns")
        mixture = GaussianMixture(
            n_components=2,
            covariance_type="diag",
            random_state=self.random_state,
            n_init=1,
            max_iter=500,
            tol=1e-3,
            reg_covar=1e-8,
        )
        features = returns.reshape(-1, 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            mixture.fit(features)
        variances = np.asarray(mixture.covariances_, dtype=np.float64).reshape(2)
        means = np.asarray(mixture.means_, dtype=np.float64).reshape(2)
        # Stable sort keeps the mapping deterministic when variances tie.
        order = np.argsort(variances, kind="mergesort")
        self.means_ = means[order]
        self.stds_ = np.sqrt(np.maximum(variances[order], 1e-18))
        original_labels = mixture.predict(features)
        remap = np.empty(2, dtype=np.int64)
        remap[order] = np.array([0, 1], dtype=np.int64)
        labels = remap[original_labels]
        counts = np.ones((2, 2), dtype=np.float64)
        np.add.at(counts, (labels[:-1], labels[1:]), 1.0)
        self.transmat_ = counts / counts.sum(axis=1, keepdims=True)
        leave_low = float(self.transmat_[0, 1])
        leave_high = float(self.transmat_[1, 0])
        stay_mass = leave_low + leave_high
        if stay_mass <= 0.0:
            self.startprob_ = np.array([0.5, 0.5], dtype=np.float64)
        else:
            # Stationary distribution of a 2-state chain.
            self.startprob_ = np.array(
                [leave_high / stay_mass, leave_low / stay_mass],
                dtype=np.float64,
            )
        self.high_vol_state = 1
        self.reset()

    def reset(self) -> None:
        """Drop the forward messages. Emission and transition parameters remain."""

        self._log_alpha = None
        self._have_posterior = False

    def update(self, obs: Bar) -> FilterState:
        if (
            self.means_ is None
            or self.stds_ is None
            or self.transmat_ is None
            or self.startprob_ is None
        ):
            raise RuntimeError("HMMFilter.update called before a successful fit")
        if not self._have_posterior or self._log_alpha is None:
            log_pred = np.log(self.startprob_)
        else:
            log_pred = logsumexp(self._log_alpha[:, None] + np.log(self.transmat_), axis=0)
        log_emit = norm.logpdf(float(obs.ret), loc=self.means_, scale=self.stds_)
        log_joint = log_pred + log_emit
        log_norm = logsumexp(log_joint)
        self._log_alpha = np.asarray(log_joint - log_norm, dtype=np.float64)
        self._have_posterior = True
        weights = np.exp(self._log_alpha)
        regime_prob = float(weights[self.high_vol_state])
        pred_vol = float(np.sqrt(np.sum(weights * self.stds_**2)))
        return FilterState(
            filtered_price=float(obs.close),
            regime_prob=regime_prob,
            surprise=float(-log_norm),
            pred_vol=pred_vol,
        )
