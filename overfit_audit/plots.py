"""Diagnostic plots for overfitting detection.

All functions use the Agg backend so they are safe in headless environments.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde


def plot_pbo_distribution(logits: np.ndarray, save_path: str) -> None:
    """Histogram and KDE of the CSCV logit distribution.

    Plots the distribution of per-partition logit values produced by
    :func:`overfit_audit.pbo.probability_of_backtest_overfitting`, with a
    vertical line at logit=0.  Values to the left of zero represent
    partitions where the IS-best strategy underperformed the OOS median.
    The fraction of mass left of zero equals PBO.

    Parameters
    ----------
    logits:
        1-D array of logit values, one per CSCV partition.
    save_path:
        File path (including extension, e.g. ``figures/pbo.png``) where the
        figure is saved.
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.hist(
        logits, bins=40, density=True, alpha=0.4,
        color="steelblue", label="Logit histogram",
    )

    if len(logits) > 1:
        kde = gaussian_kde(logits)
        x = np.linspace(logits.min() - 0.5, logits.max() + 0.5, 300)
        ax.plot(x, kde(x), color="steelblue", linewidth=2, label="KDE")

    ax.axvline(0.0, color="crimson", linewidth=1.5, linestyle="--", label="Logit = 0")

    pbo = float(np.mean(logits <= 0.0))
    ax.set_title(f"CSCV Logit Distribution  (PBO = {pbo:.3f})")
    ax.set_xlabel("Logit of OOS relative rank")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_is_oos_degradation(
    is_sharpes: np.ndarray,
    oos_sharpes: np.ndarray,
    save_path: str,
) -> None:
    """Scatter of in-sample vs out-of-sample Sharpe across CSCV partitions.

    Each point is one CSCV partition: x = IS Sharpe of the best strategy,
    y = OOS Sharpe of that same strategy.  Points above the diagonal indicate
    OOS outperformance; points below indicate degradation (the common case
    under overfitting).

    Parameters
    ----------
    is_sharpes:
        1-D array of in-sample Sharpe ratios for the IS-best strategy,
        one value per CSCV partition.
    oos_sharpes:
        1-D array of OOS Sharpe ratios for the same strategy, aligned
        with *is_sharpes*.
    save_path:
        File path where the figure is saved.
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(
        is_sharpes, oos_sharpes, alpha=0.4, s=20, color="steelblue", label="Partitions"
    )

    all_vals = np.concatenate([is_sharpes, oos_sharpes])
    lo, hi = float(all_vals.min()), float(all_vals.max())
    pad = (hi - lo) * 0.05 or 0.1
    diag = np.array([lo - pad, hi + pad])
    ax.plot(
        diag, diag, color="crimson", linewidth=1.5, linestyle="--", label="IS = OOS"
    )

    ax.set_xlabel("In-sample Sharpe (best strategy)")
    ax.set_ylabel("Out-of-sample Sharpe (same strategy)")
    ax.set_title("IS vs OOS Sharpe Degradation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
