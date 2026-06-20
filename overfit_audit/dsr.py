"""Deflated Sharpe Ratio (DSR) and supporting estimators.

Reference: López de Prado, M. (2018). *Advances in Financial Machine Learning*,
Chapter 8.  Bailey & López de Prado (2014). "The Deflated Sharpe Ratio:
Correcting for Selection Bias, Backtest Overfitting, and Non-Normality".
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

_EULER_MASCHERONI: float = 0.5772156649015328


def probabilistic_sharpe_ratio(
    sr: float,
    T: int,
    skew: float,
    kurt: float,
    sr_benchmark: float = 0.0,
) -> float:
    """Probability that the true Sharpe ratio exceeds *sr_benchmark*.

    Given a sample Sharpe ratio estimated from T observations, returns the
    one-sided p-value that the strategy's true Sharpe exceeds sr_benchmark,
    corrected for the non-normality of returns.

    The standard error of the Sharpe estimator under non-normal returns is:

        sr_std = sqrt( (1 - skew * sr + ((kurt - 1) / 4) * sr^2) / (T - 1) )

    where **kurt is the full (non-excess) kurtosis** (equals 3 for a normal
    distribution).  The (kurt - 1) term is therefore (kurt - 1), not the
    excess kurtosis directly.  For normal returns this reduces to the
    classical formula sqrt((1 + sr^2/2) / (T-1)) ≈ sqrt(1/(T-1)).

    PSR = Φ( (sr - sr_benchmark) / sr_std )

    Parameters
    ----------
    sr:
        Observed (annualised or per-period) Sharpe ratio.
    T:
        Number of observations used to estimate *sr*.
    skew:
        Sample skewness of the return series.
    kurt:
        Sample **full** kurtosis (not excess kurtosis; normal = 3).
    sr_benchmark:
        The Sharpe ratio we test against.  Defaults to 0.

    Returns
    -------
    float
        Probability in (0, 1).
    """
    sr_std = math.sqrt(
        (1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr**2) / (T - 1)
    )
    return float(norm.cdf((sr - sr_benchmark) / sr_std))


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """Expected maximum Sharpe ratio across *n_trials* independent strategies.

    Uses the Extreme Value Theory approximation for the expected maximum of
    n_trials draws from a distribution with the given variance:

        E[max SR] = sqrt(sr_variance) *
                    ( (1 - γ) * Φ^{-1}(1 - 1/n) + γ * Φ^{-1}(1 - 1/(n·e)) )

    where γ ≈ 0.5772 is the Euler–Mascheroni constant, Φ^{-1} is the
    inverse standard-normal CDF, n = n_trials, and e is Euler's number.

    This captures the fact that even purely random strategies will produce
    a best-in-class Sharpe that grows as O(sqrt(log n)).

    Parameters
    ----------
    n_trials:
        Number of independently tested strategies.
    sr_variance:
        Variance of the distribution of Sharpe estimates across trials.

    Returns
    -------
    float
        Expected maximum Sharpe ratio.
    """
    sr_std = math.sqrt(sr_variance)
    gamma = _EULER_MASCHERONI
    e = math.e

    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * e))

    return float(sr_std * ((1.0 - gamma) * z1 + gamma * z2))


def deflated_sharpe_ratio(
    observed_sr: float,
    sr_estimates: np.ndarray,
    T: int,
    skew: float,
    kurt: float,
) -> float:
    """Deflated Sharpe Ratio: PSR with a data-driven selection-bias benchmark.

    Rather than testing against a fixed benchmark of 0, the DSR accounts for
    the fact that the "best" strategy was chosen from many candidates.  The
    benchmark is set to the *expected maximum Sharpe* one would obtain from
    len(sr_estimates) independent noise strategies with the same dispersion
    as the observed trial Sharpes.

    A DSR near 0 means the observed Sharpe is indistinguishable from what
    luck alone would produce across that many trials.  A DSR near 1 indicates
    the observed Sharpe genuinely exceeds the selection-bias-corrected bar.

    Parameters
    ----------
    observed_sr:
        Sharpe ratio of the strategy being evaluated.
    sr_estimates:
        Array of Sharpe ratios from all strategies/trials (including the
        selected one).  Used to estimate the variance of chance performance.
    T:
        Number of observations in the backtest.
    skew:
        Skewness of the selected strategy's return series.
    kurt:
        Full kurtosis of the selected strategy's return series (normal = 3).

    Returns
    -------
    float
        Probability in (0, 1).
    """
    n_trials = len(sr_estimates)
    sr_variance = float(np.var(sr_estimates, ddof=1))
    benchmark = expected_max_sharpe(n_trials, sr_variance)
    return probabilistic_sharpe_ratio(
        observed_sr, T, skew, kurt, sr_benchmark=benchmark
    )
