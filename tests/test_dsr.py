"""Tests for overfit_audit.dsr."""

import numpy as np

from overfit_audit.dsr import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
)

# ---------------------------------------------------------------------------
# probabilistic_sharpe_ratio
# ---------------------------------------------------------------------------


def test_psr_at_benchmark_is_half() -> None:
    """PSR should be ~0.5 when sr equals the benchmark."""
    psr = probabilistic_sharpe_ratio(
        sr=1.0, T=1000, skew=0.0, kurt=3.0, sr_benchmark=1.0
    )
    assert abs(psr - 0.5) < 1e-6


def test_psr_increases_with_T() -> None:
    """More observations → tighter SE → higher confidence that sr > 0."""
    psr_small = probabilistic_sharpe_ratio(sr=0.5, T=100, skew=0.0, kurt=3.0)
    psr_large = probabilistic_sharpe_ratio(sr=0.5, T=10_000, skew=0.0, kurt=3.0)
    assert psr_large > psr_small


def test_psr_in_unit_interval() -> None:
    """PSR must always lie in (0, 1)."""
    for sr in [-2.0, 0.0, 0.5, 1.0, 3.0]:
        psr = probabilistic_sharpe_ratio(sr=sr, T=500, skew=-0.5, kurt=4.0)
        assert 0.0 < psr <= 1.0


# ---------------------------------------------------------------------------
# expected_max_sharpe
# ---------------------------------------------------------------------------


def test_expected_max_sharpe_increases_with_n_trials() -> None:
    """The more trials, the higher the expected best-by-chance Sharpe."""
    ems_10 = expected_max_sharpe(n_trials=10, sr_variance=1.0)
    ems_100 = expected_max_sharpe(n_trials=100, sr_variance=1.0)
    ems_1000 = expected_max_sharpe(n_trials=1000, sr_variance=1.0)
    assert ems_10 < ems_100 < ems_1000


def test_expected_max_sharpe_scales_with_variance() -> None:
    """E[max SR] should scale with sqrt(sr_variance)."""
    ems_low = expected_max_sharpe(n_trials=100, sr_variance=0.25)
    ems_high = expected_max_sharpe(n_trials=100, sr_variance=1.0)
    assert ems_high > ems_low


# ---------------------------------------------------------------------------
# deflated_sharpe_ratio
# ---------------------------------------------------------------------------


def test_dsr_noise_selection_is_low() -> None:
    """DSR of the best noise strategy chosen from many trials should be < 0.5."""
    rng = np.random.default_rng(0)
    T = 1260
    N = 500
    returns = rng.standard_normal((T, N)) * 0.01
    sr_estimates = returns.mean(axis=0) / returns.std(axis=0, ddof=1)
    best_idx = int(np.argmax(sr_estimates))
    col = returns[:, best_idx]

    import scipy.stats as ss

    dsr = deflated_sharpe_ratio(
        observed_sr=sr_estimates[best_idx],
        sr_estimates=sr_estimates,
        T=T,
        skew=float(ss.skew(col)),
        kurt=float(ss.kurtosis(col, fisher=False)),
    )
    assert dsr < 0.5


def test_dsr_genuine_edge_is_high() -> None:
    """DSR of a truly high Sharpe chosen from few trials should be > 0.9."""
    rng = np.random.default_rng(1)
    T = 2520
    N = 5
    # One strategy with annualised SR ≈ 3; others near zero
    daily_drift = 3.0 / np.sqrt(252) * 0.01
    good_returns = rng.standard_normal(T) * 0.01 + daily_drift
    noise_returns = rng.standard_normal((T, N - 1)) * 0.01

    import scipy.stats as ss

    sr_good = float(good_returns.mean() / good_returns.std(ddof=1))
    sr_noise = noise_returns.mean(axis=0) / noise_returns.std(axis=0, ddof=1)
    sr_estimates = np.concatenate([[sr_good], sr_noise])

    dsr = deflated_sharpe_ratio(
        observed_sr=sr_good,
        sr_estimates=sr_estimates,
        T=T,
        skew=float(ss.skew(good_returns)),
        kurt=float(ss.kurtosis(good_returns, fisher=False)),
    )
    assert dsr > 0.9
