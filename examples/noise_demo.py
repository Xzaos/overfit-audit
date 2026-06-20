"""Noise demo: overfitting detection on pure-noise strategies.

Generates 1000 strategies with zero true edge and shows how naive IS
selection produces an inflated Sharpe that the DSR and PBO correctly
identify as spurious.

Run from the repo root:
    python examples/noise_demo.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import scipy.stats as ss

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from overfit_audit.dsr import deflated_sharpe_ratio, expected_max_sharpe
from overfit_audit.pbo import probability_of_backtest_overfitting
from overfit_audit.plots import plot_pbo_distribution

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

SEED = 42
T = 1260          # ~5 trading years of daily returns
N = 1000          # strategies with zero true edge
DAILY_VOL = 0.01  # 1% daily vol -> annualised vol ~16%
TRADING_DAYS = 252

FIGURES_DIR = pathlib.Path(__file__).resolve().parents[1] / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Generate pure-noise return matrix
# ---------------------------------------------------------------------------

rng = np.random.default_rng(SEED)
returns = rng.standard_normal((T, N)) * DAILY_VOL   # shape (T, N)

# ---------------------------------------------------------------------------
# 2. Naive IS selection: pick the strategy with the highest Sharpe
# ---------------------------------------------------------------------------

daily_sharpes = returns.mean(axis=0) / returns.std(axis=0, ddof=1)
best_idx = int(np.argmax(daily_sharpes))
best_col = returns[:, best_idx]

naive_sr_annualised = float(daily_sharpes[best_idx] * np.sqrt(TRADING_DAYS))

print("=" * 60)
print("STEP 1 -- Naive in-sample selection")
print(f"  Strategies tested             : {N}")
print(f"  Observations per strategy     : {T} days (~{T // TRADING_DAYS} years)")
print(f"  Best in-sample SR (annualised): {naive_sr_annualised:.3f}")

# ---------------------------------------------------------------------------
# 3. Deflated Sharpe Ratio
# ---------------------------------------------------------------------------

skew = float(ss.skew(best_col))
kurt = float(ss.kurtosis(best_col, fisher=False))   # full kurtosis

sr_variance = float(np.var(daily_sharpes, ddof=1))
benchmark = expected_max_sharpe(n_trials=N, sr_variance=sr_variance)

dsr = deflated_sharpe_ratio(
    observed_sr=daily_sharpes[best_idx],
    sr_estimates=daily_sharpes,
    T=T,
    skew=skew,
    kurt=kurt,
)

print()
print("STEP 2 -- Deflated Sharpe Ratio")
print(f"  Skewness of best strategy     : {skew:.3f}")
print(f"  Full kurtosis                 : {kurt:.3f}")
print(f"  sr_variance (all {N} trials)  : {sr_variance:.8f}")
print(f"  benchmark E[max SR]           : {benchmark:.6f}  <- expected max by chance")
print(f"  observed daily SR             : {daily_sharpes[best_idx]:.6f}")
print(f"  DSR                           : {dsr:.4f}  (~0.5 = right at chance level)")

# ---------------------------------------------------------------------------
# 4. Probability of Backtest Overfitting
# ---------------------------------------------------------------------------

print()
print("STEP 3 -- Probability of Backtest Overfitting (CSCV, n_splits=16)")
print("  Running CSCV ... ", end="", flush=True)

pbo_result = probability_of_backtest_overfitting(returns, n_splits=16)
pbo = pbo_result["pbo"]
logits = pbo_result["logits"]

print("done.")
print(f"  PBO                           : {pbo:.3f}")

# ---------------------------------------------------------------------------
# 5. Save logit distribution plot
# ---------------------------------------------------------------------------

plot_path = str(FIGURES_DIR / "pbo_logit_distribution.png")
plot_pbo_distribution(logits, save_path=plot_path)
print()
print(f"  Logit distribution saved -> {plot_path}")

# ---------------------------------------------------------------------------
# 6. Verdict (single seed)
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("VERDICT (seed=42)")
print(
    f"  Apparent annualised Sharpe : {naive_sr_annualised:.2f}  "
    f"(top pick from {N} noise strategies)"
)
print(f"  DSR                        : {dsr:.3f}")
print(f"  PBO                        : {pbo:.3f}")
print()
print(
    f"  DSR and PBO are computed from a single noise realization (seed=42).\n"
    f"  Their EXPECTED values across many seeds are DSR ~0.5 and PBO ~0.5\n"
    f"  -- chance level. This run shows DSR={dsr:.3f}, PBO={pbo:.3f}: still\n"
    f"  far from the false confidence a naive Sharpe of"
    f" {naive_sr_annualised:.2f} would\n"
    f"  suggest, illustrating that even an unusually 'lucky' noise draw is\n"
    f"  correctly flagged as statistically unremarkable, not as genuine skill."
)
print("=" * 60)

# ---------------------------------------------------------------------------
# 7. Multi-seed convergence: empirical evidence that E[DSR] ~ E[PBO] ~ 0.5
# ---------------------------------------------------------------------------

# n_splits=10 gives C(10,5)=252 combinations: exhaustive and fast (~1 s/seed).
# n_splits=16 (used above) subsamples 10 000 combos and takes ~60 s/seed,
# which would make a 20-seed loop impractical.
_SWEEP_SPLITS = 10

N_SEEDS = 20
print()
print(
    f"Multi-seed sweep ({N_SEEDS} seeds, n_splits={_SWEEP_SPLITS})"
    " -- verifying E[DSR] ~ E[PBO] ~ 0.5"
)
print("-" * 60)

seed_dsrs: list[float] = []
seed_pbos: list[float] = []

for s in range(N_SEEDS):
    rng_s = np.random.default_rng(s)
    ret_s = rng_s.standard_normal((T, N)) * DAILY_VOL

    sr_s = ret_s.mean(axis=0) / ret_s.std(axis=0, ddof=1)
    best_s = int(np.argmax(sr_s))
    col_s = ret_s[:, best_s]

    dsr_s = deflated_sharpe_ratio(
        observed_sr=sr_s[best_s],
        sr_estimates=sr_s,
        T=T,
        skew=float(ss.skew(col_s)),
        kurt=float(ss.kurtosis(col_s, fisher=False)),
    )

    pbo_s = probability_of_backtest_overfitting(
        ret_s, n_splits=_SWEEP_SPLITS
    )["pbo"]

    seed_dsrs.append(dsr_s)
    seed_pbos.append(float(pbo_s))
    print(f"  seed={s:>2d}  DSR={dsr_s:.3f}  PBO={pbo_s:.3f}")

mean_dsr = float(np.mean(seed_dsrs))
mean_pbo = float(np.mean(seed_pbos))
print("-" * 60)
print(f"  Mean DSR across {N_SEEDS} seeds: {mean_dsr:.3f}  (expect ~0.5)")
print(f"  Mean PBO across {N_SEEDS} seeds: {mean_pbo:.3f}  (expect ~0.5)")
print()
print(
    "  No genuine skill detected -- the apparent Sharpe is an artifact "
    f"of selection across {N} trials."
)
print("=" * 60)
