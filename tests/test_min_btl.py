"""Tests for overfit_audit.min_btl."""

import pytest

from overfit_audit.min_btl import minimum_backtest_length

# ---------------------------------------------------------------------------
# return type and sign
# ---------------------------------------------------------------------------


def test_returns_positive_float() -> None:
    result = minimum_backtest_length(target_sr=1.0, n_trials=100)
    assert isinstance(result, float)
    assert result > 0.0


# ---------------------------------------------------------------------------
# monotonicity
# ---------------------------------------------------------------------------


def test_minbtl_increases_with_n_trials() -> None:
    """More trials tested → higher bar → longer backtest required."""
    btl_10 = minimum_backtest_length(target_sr=1.0, n_trials=10)
    btl_100 = minimum_backtest_length(target_sr=1.0, n_trials=100)
    btl_1000 = minimum_backtest_length(target_sr=1.0, n_trials=1000)
    assert btl_10 < btl_100 < btl_1000


def test_minbtl_decreases_with_higher_target_sr() -> None:
    """A higher claimed Sharpe needs proportionally less history to be credible."""
    btl_low = minimum_backtest_length(target_sr=0.5, n_trials=100)
    btl_mid = minimum_backtest_length(target_sr=1.0, n_trials=100)
    btl_high = minimum_backtest_length(target_sr=2.0, n_trials=100)
    assert btl_low > btl_mid > btl_high


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def test_raises_on_non_positive_sr() -> None:
    with pytest.raises(ValueError, match="target_sr"):
        minimum_backtest_length(target_sr=0.0, n_trials=10)


def test_raises_on_negative_sr() -> None:
    with pytest.raises(ValueError, match="target_sr"):
        minimum_backtest_length(target_sr=-1.0, n_trials=10)


def test_raises_on_zero_trials() -> None:
    with pytest.raises(ValueError, match="n_trials"):
        minimum_backtest_length(target_sr=1.0, n_trials=0)
