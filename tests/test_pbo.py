"""Tests for overfit_audit.pbo."""

import numpy as np
import pytest

from overfit_audit.pbo import probability_of_backtest_overfitting

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _noise_matrix(T: int, N: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((T, N)) * 0.01


def _edge_matrix(T: int, N: int, drift: float = 0.002, seed: int = 0) -> np.ndarray:
    """One strategy with persistent positive drift; the rest pure noise."""
    rng = np.random.default_rng(seed)
    matrix = rng.standard_normal((T, N)) * 0.01
    matrix[:, 0] += drift
    return matrix


# ---------------------------------------------------------------------------
# return type / structure
# ---------------------------------------------------------------------------


def test_returns_dict_with_required_keys() -> None:
    result = probability_of_backtest_overfitting(_noise_matrix(160, 20), n_splits=10)
    assert "pbo" in result
    assert "logits" in result


def test_pbo_in_unit_interval() -> None:
    result = probability_of_backtest_overfitting(_noise_matrix(160, 20), n_splits=10)
    assert 0.0 <= result["pbo"] <= 1.0


def test_logits_is_ndarray() -> None:
    result = probability_of_backtest_overfitting(_noise_matrix(160, 20), n_splits=10)
    assert isinstance(result["logits"], np.ndarray)
    assert result["logits"].ndim == 1


# ---------------------------------------------------------------------------
# statistical behaviour
# ---------------------------------------------------------------------------


def test_pbo_near_half_on_pure_noise() -> None:
    """On a pure-noise matrix IS selection has no predictive power → PBO ≈ 0.5."""
    result = probability_of_backtest_overfitting(
        _noise_matrix(T=500, N=50, seed=7), n_splits=10
    )
    assert 0.25 <= result["pbo"] <= 0.75


def test_pbo_low_with_genuine_edge() -> None:
    """One strategy with real drift should win OOS most of the time → PBO < 0.2."""
    result = probability_of_backtest_overfitting(
        _edge_matrix(T=500, N=20, drift=0.003, seed=3), n_splits=10
    )
    assert result["pbo"] < 0.2


# ---------------------------------------------------------------------------
# edge cases / validation
# ---------------------------------------------------------------------------


def test_raises_on_odd_n_splits() -> None:
    with pytest.raises(ValueError, match="even"):
        probability_of_backtest_overfitting(_noise_matrix(100, 10), n_splits=7)


def test_raises_when_too_few_rows() -> None:
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(_noise_matrix(4, 10), n_splits=10)
