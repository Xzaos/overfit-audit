"""Probability of Backtest Overfitting (PBO) via Combinatorially Symmetric
Cross-Validation (CSCV).

Reference: Bailey, D. H., & López de Prado, M. (2014). "The Probability of
Backtest Overfitting". *Journal of Computational Finance*, 20(4), 39–69.
"""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np

_MAX_COMBINATIONS: int = 10_000
_SUBSAMPLE_SEED: int = 42


def probability_of_backtest_overfitting(
    returns_matrix: np.ndarray,
    n_splits: int = 16,
) -> dict[str, object]:
    """Estimate the Probability of Backtest Overfitting via CSCV.

    Partitions the return history into *n_splits* contiguous blocks and
    enumerates all ways to choose n_splits/2 blocks as in-sample (IS), with
    the remainder as out-of-sample (OOS).  For each partition:

      1. Compute the per-period Sharpe ratio of every strategy in-sample.
      2. Identify the IS-best strategy.
      3. Rank that strategy among all strategies out-of-sample (rank in [1, N]).
      4. Convert to a relative rank w = rank / (N + 1) ∈ (0, 1).
      5. Compute logit = log(w / (1 - w)).

    PBO is the fraction of partitions where logit ≤ 0, i.e. the IS-winner
    ranks at or below the OOS median — a sign that IS selection does not
    transfer to OOS performance.

    If the total number of combinations C(n_splits, n_splits/2) exceeds
    10 000, a random subsample of 10 000 combinations is drawn with
    seed=42 so results remain reproducible while keeping runtime bounded.

    Parameters
    ----------
    returns_matrix:
        Array of shape (T, N) — T time periods, N strategy return series.
        Each column is one strategy; each row is one period.
    n_splits:
        Number of contiguous blocks to partition the time axis into.
        Must be even.  Defaults to 16.

    Returns
    -------
    dict with keys:
        ``"pbo"``    — float in [0, 1], fraction of partitions where the
                       IS-best strategy underperforms OOS median.
        ``"logits"`` — np.ndarray of per-partition logit values.

    Raises
    ------
    ValueError
        If *n_splits* is odd or returns_matrix has fewer rows than n_splits.
    """
    if n_splits % 2 != 0:
        raise ValueError(f"n_splits must be even, got {n_splits}.")

    T, N = returns_matrix.shape
    if T < n_splits:
        raise ValueError(
            f"returns_matrix has {T} rows but n_splits={n_splits} "
            "requires at least that many."
        )

    block_size = T // n_splits
    blocks = [
        returns_matrix[i * block_size : (i + 1) * block_size, :]
        for i in range(n_splits)
    ]

    half = n_splits // 2
    all_combos = list(combinations(range(n_splits), half))

    rng = np.random.default_rng(_SUBSAMPLE_SEED)
    if len(all_combos) > _MAX_COMBINATIONS:
        indices = rng.choice(len(all_combos), size=_MAX_COMBINATIONS, replace=False)
        sampled_combos = [all_combos[i] for i in indices]
    else:
        sampled_combos = all_combos

    logits: list[float] = []

    for is_indices in sampled_combos:
        oos_indices = tuple(i for i in range(n_splits) if i not in set(is_indices))

        is_returns = np.concatenate([blocks[i] for i in is_indices], axis=0)
        oos_returns = np.concatenate([blocks[i] for i in oos_indices], axis=0)

        is_sharpes = _sharpe(is_returns)
        best = int(np.argmax(is_sharpes))

        oos_sharpes = _sharpe(oos_returns)
        rank = int(np.sum(oos_sharpes <= oos_sharpes[best]))
        w = rank / (N + 1)
        w = np.clip(w, 1e-8, 1.0 - 1e-8)
        logits.append(math.log(w / (1.0 - w)))

    logits_arr = np.array(logits, dtype=float)
    pbo = float(np.mean(logits_arr <= 0.0))

    return {"pbo": pbo, "logits": logits_arr}


def _sharpe(matrix: np.ndarray) -> np.ndarray:
    """Per-period Sharpe ratio for each column of *matrix*."""
    mu = matrix.mean(axis=0)
    sigma = matrix.std(axis=0, ddof=1)
    sigma = np.where(sigma == 0.0, np.finfo(float).eps, sigma)
    return mu / sigma
