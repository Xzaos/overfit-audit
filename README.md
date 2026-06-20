# overfit-audit

Tools to tell whether a backtested Sharpe is real or an artifact of multiple
testing: the Deflated Sharpe Ratio, Probability of Backtest Overfitting (CSCV),
and Minimum Backtest Length.

## The core problem
Run enough strategies and one of them will look brilliant by chance. These tools
quantify that.

- **Deflated Sharpe Ratio** — deflates an observed Sharpe by the number of trials,
  return skew/kurtosis, and sample length. Under pure noise, DSR converges to
  ~0.5 (chance level) across many trials/seeds — not 0. A single noise realization
  can land anywhere; DSR tells you whether a specific observed Sharpe beats the
  *expected* best of N trials, not whether any noise can ever look good.
- **PBO (CSCV)** — the probability that the in-sample-best strategy underperforms
  the median out-of-sample. Converges to ~0.5 under pure noise (no predictive
  value in selection).
- **Minimum Backtest Length** — the sample you need before a target Sharpe is
  non-spurious given N trials.

## The decisive demo
```bash
python examples/noise_demo.py
```
Runs CSCV/DSR on 1000 pure-noise strategies across 20 random seeds. A single
seed can look deceptively good (DSR=0.72, PBO=0.32 on seed=42) purely from
sampling variance — but averaged across seeds, DSR converges to ~0.475 and
PBO to ~0.500, both at chance level. This is the honest, statistically rigorous
result: no individual run "proves" no skill, but the distribution does.

## Failure modes & assumptions
- DSR and PBO are statistics with their own sampling variance — a single run
  can mislead. Always interpret in light of the seed-averaged behavior, not
  one realization.
- CSCV cost grows combinatorially in n_splits; the implementation subsamples
  combinations above 10,000, trading exactness for tractability.
- These are necessary, not sufficient, checks — passing them does not certify a
  live edge; consistently failing them across seeds is a strong stop signal.

## References
Bailey & López de Prado, *The Deflated Sharpe Ratio* and *The Probability of
Backtest Overfitting*.
