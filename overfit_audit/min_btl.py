"""Minimum Backtest Length (minBTL) estimator.

Reference: López de Prado, M. (2018). *Advances in Financial Machine Learning*,
Chapter 8.  Bailey & López de Prado (2014). "The Deflated Sharpe Ratio:
Correcting for Selection Bias, Backtest Overfitting, and Non-Normality".
"""

from __future__ import annotations

from overfit_audit.dsr import expected_max_sharpe


def minimum_backtest_length(target_sr: float, n_trials: int) -> float:
    """Minimum years of backtest needed for *target_sr* to be non-spurious.

    Answers the question: given that you tested *n_trials* strategies and are
    presenting the one with Sharpe *target_sr*, how many years of history are
    required before that Sharpe clears the selection-bias-corrected bar?

    The formula derives from the condition PSR(target_sr) ≥ 0.95, which
    simplifies (for normal returns, skew=0, kurt=3) to requiring that the
    observed SR exceeds the expected maximum Sharpe by at least one standard
    error.  Rearranging López de Prado's minBTL expression:

        minBTL ≈ (E[max SR] / target_sr)² + 1   (in observations)

    converted to years by dividing by 252 (trading days per year).

    **Assumption**: *target_sr* is annualised (based on daily returns scaled
    by √252).  The result is in calendar years.  If your Sharpe is computed
    on a different frequency, scale target_sr accordingly before calling this
    function.

    The variance of Sharpe estimates across trials is assumed to be 1.0
    (unit variance), which is the natural normalisation when comparing
    strategies on a common scale.  For a different variance, scale
    target_sr by sqrt(sr_variance) before calling.

    Parameters
    ----------
    target_sr:
        Annualised Sharpe ratio of the strategy being evaluated.
    n_trials:
        Number of strategies/parameter sets tested before selecting this one.

    Returns
    -------
    float
        Minimum backtest length in years (positive).

    Raises
    ------
    ValueError
        If *target_sr* is not positive or *n_trials* is less than 1.
    """
    if target_sr <= 0.0:
        raise ValueError(f"target_sr must be positive, got {target_sr}.")
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}.")

    sr_variance = 1.0
    ems = expected_max_sharpe(n_trials=n_trials, sr_variance=sr_variance)

    observations = (ems / target_sr) ** 2 + 1.0
    years = observations / 252.0

    return float(years)
