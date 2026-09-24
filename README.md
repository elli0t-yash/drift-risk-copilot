# drift-risk-copilot

Portfolio risk copilot for BFSI users (AI Builder Cup 2026). This checkpoint
covers the **compute layer** only — a Rust workspace that fetches market
data, fits a factor risk model, and runs portfolio-risk experiments,
returning a fully-cited `EvidenceTrace` for every result. The `agent` and
`server` crates are empty stubs, built in a later checkpoint.

## Workspace layout

```
crates/
  compute/   # data layer, factor model, experiments, evidence trace (this checkpoint)
    src/data.rs         Yahoo fetch + cache + NSE calendar alignment + log returns
    src/model.rs         OLS factor fits, Ledoit-Wolf shrinkage, stock covariance
    src/experiments.rs    FactorShock, RiskDecomposition (implemented); CvarRebalance (design note)
    src/trace.rs          EvidenceTrace and its sub-structs
    src/bin/experiment.rs CLI: runs one experiment from a JSON file
    examples/              FactorShock / RiskDecomposition inputs for a 10-stock Nifty portfolio
    tests/                  Synthetic-data unit/integration tests (no network required)
  agent/     # stub — LLM-facing layer, not yet built
  server/    # stub — HTTP API, not yet built
data/cache/  # cached raw price CSVs (gitignored; fetched on first run)
```

## Running an experiment

```
cargo run -p compute --bin experiment -- crates/compute/examples/factor_shock_nifty10.json
cargo run -p compute --bin experiment -- crates/compute/examples/risk_decomposition_nifty10.json
```

Add `--refresh` to refetch price series instead of reading `data/cache/`.
Each run prints a JSON `EvidenceTrace` to stdout.

## Return frequency

`FactorShockInput`/`RiskDecompositionInput` take an optional `frequency`
(`"Daily"` | `"Weekly"`, default `"Daily"` — unchanged from the first
checkpoint) and an optional `window` (periods at that frequency; omit to
get `frequency.default_window()`: 252 for Daily, 156 for Weekly).
`Weekly` returns are **non-overlapping**, computed between successive
last-NSE-trading-day-of-the-week closes (`data::weekly_resample_indices`),
not a rolling 5-day window. Annualization (252 vs. 52) is still derived
from a single function, `model::annualize_matrix`/`annualize_scalar`,
now parameterized by `Frequency` instead of a bare constant.

**`examples/*_weekly.json`** run the same 10-stock portfolio at
`Weekly`/156 for comparison against the `Daily`/252 default.

### Daily vs. Weekly comparison (10-stock Nifty portfolio, live data, run 2026-09-24)

Betas on USDINR / BRENT / GOLD / RATES_PROXY (MARKET omitted — large and
stable across both frequencies, ~0.7-1.1 for all ten names):

| ticker | USDINR (D) | USDINR (W) | BRENT (D) | BRENT (W) | GOLD (D) | GOLD (W) | RATES_PROXY (D) | RATES_PROXY (W) |
|---|---|---|---|---|---|---|---|---|
| RELIANCE.NS | -0.059 | -0.505 | -0.011 | 0.046 | -0.016 | -0.083 | -0.732 | -0.597 |
| HDFCBANK.NS | 0.021 | 0.166 | 0.016 | -0.021 | -0.038 | 0.017 | 0.668 | 1.080 |
| ICICIBANK.NS | 0.205 | -0.384 | -0.012 | 0.004 | -0.033 | -0.031 | 0.868 | 0.928 |
| INFY.NS | -0.207 | -0.156 | 0.022 | 0.002 | -0.051 | -0.117 | -1.342 | -1.257 |
| TCS.NS | -0.364 | 0.085 | -0.002 | 0.022 | -0.052 | 0.029 | -1.333 | -0.974 |
| LT.NS | 0.093 | 0.106 | -0.003 | -0.005 | 0.020 | -0.019 | -0.037 | 0.292 |
| ITC.NS | -0.050 | 0.388 | -0.004 | -0.023 | -0.046 | 0.074 | -0.573 | -0.573 |
| KOTAKBANK.NS | 0.244 | -0.109 | -0.002 | 0.005 | 0.037 | 0.127 | 0.416 | 0.413 |
| BHARTIARTL.NS | 0.067 | -0.268 | 0.036 | 0.042 | 0.035 | 0.009 | -0.547 | -0.580 |
| TMPV.NS | 0.083 | 1.225 | -0.052 | -0.086 | -0.060 | -0.111 | -0.011 | -0.760 |

**Takeaway:** the USDINR and RATES_PROXY betas are noticeably less stable
across frequency than MARKET (expected — currency and rate-proxy signal is
noisier per-name and weekly regressions have ~6x fewer observations per
year); BRENT and GOLD betas are small and noisy at both frequencies (no
stock in this portfolio has meaningfully commodity-linked earnings).
Shrinkage intensity is higher weekly (0.053) than daily (0.038), consistent
with a noisier per-period factor covariance estimate needing more
shrinkage toward the target.

Factor correlation matrix (Daily / Weekly, same order MARKET, USDINR,
BRENT, GOLD, RATES_PROXY):

```
Daily                                    Weekly
         MKT   USD   BRT   GLD   RTP              MKT   USD   BRT   GLD   RTP
MKT    1.00 -0.28 -0.27  0.24  0.20     MKT     1.00 -0.23 -0.29  0.04  0.04
USD   -0.28  1.00  0.23 -0.19 -0.02     USD    -0.23  1.00  0.24 -0.04 -0.04
BRT   -0.27  0.23  1.00 -0.17 -0.19     BRT    -0.29  0.24  1.00 -0.08 -0.14
GLD    0.24 -0.19 -0.17  1.00  0.11     GLD     0.04 -0.04 -0.08  1.00 -0.03
RTP    0.20 -0.02 -0.19  0.11  1.00     RTP     0.04 -0.04 -0.14 -0.03  1.00
```

MARKET/USDINR and MARKET/BRENT correlations are stable in sign and
magnitude across frequency (~-0.23 to -0.29); GOLD's and RATES_PROXY's
correlations with everything else shrink toward zero weekly, consistent
with those being the noisiest factor pair at daily frequency (aliased
short-horizon noise that partially cancels over a week).

FactorShock (Nifty -12%, Brent +20%, propagate=true) implied moves:

| factor | Daily | Weekly |
|---|---|---|
| USDINR | +2.42% | +1.63% |
| GOLD | -6.34% | -1.12% |
| RATES_PROXY | -1.79% | -0.67% |
| portfolio P&L (INR) | -1,175,389 | -1,132,802 |

RiskDecomposition shares (`fraction_of_vol`):

| | Daily | Weekly |
|---|---|---|
| portfolio vol (annualized) | 15.52% | 13.50% |
| MARKET | 82.4% | 82.2% |
| specific risk | 18.8% | 17.4% |
| USDINR / BRENT / GOLD / RATES_PROXY (combined) | -1.2% | 0.4% |

Vol estimates are reasonably close (15.5% vs 13.5%); the systematic/specific
split is nearly identical (~82/18 both ways), which is the most
frequency-robust number here. **Not changing the default** pending review —
Daily/252 stays the default per the spec until you confirm one way or the
other; Weekly's much smaller effective sample (260 weeks in a 5y fetch vs.
1235 days) makes its factor-covariance and small-beta estimates visibly
noisier, which shows up as instability in the USDINR/RATES_PROXY betas
above.

### `INR=X` vs. `USDINR=X`

Checked both tickers against the `^NSEI` master calendar over the same 5y
fetch: identical raw observation counts (1300), identical missing-date sets
relative to NSEI (149 dates each), and identical gap-run-length histograms
(all singleton 1-day gaps, no clustering — `Counter({1: 149})` for both).
No behavioral difference; kept `INR=X` (already in use, and the shorter of
the two equivalent tickers).

## Tests

```
cargo test --workspace
```

All tests use synthetic in-memory `MarketData` (see `tests/common/mod.rs`) —
no network access required.

## Dependencies and why

| Crate | Why |
|---|---|
| `nalgebra` | Matrix algebra for OLS (SVD-based pseudo-inverse solve), Ledoit-Wolf shrinkage, and portfolio covariance/Euler decomposition. |
| `serde` / `serde_json` | Serialize every experiment input/output and the `EvidenceTrace` to JSON. |
| `schemars` (with `chrono` feature) | JSON Schema derivation for `Experiment` and trace types, for the future agent/tool-calling layer. |
| `chrono` | Calendar-correct date handling for the NSE trading calendar and price series. |
| `thiserror` | Structured `ComputeError` variants instead of stringly-typed errors. |
| `reqwest` (blocking) | HTTP client for Yahoo Finance's chart endpoint. |
| `csv` | Reading/writing the on-disk price cache. |
| `clap` (derive) | CLI argument parsing (`--refresh`, `--cache-dir`) for the `experiment` binary — not in the original justified list, added because the spec requires a `--refresh` flag and hand-rolled arg parsing would be worse than a one-line derive. |

## Judgment calls

- **Ticker substitution:** `TATAMOTORS.NS` 404s on Yahoo's chart endpoint
  (Tata Motors demerged its commercial-vehicle business in 2025); the
  example portfolio uses `TMPV.NS` (Tata Motors Passenger Vehicles, the
  surviving Yahoo-listed entity) instead. This is a data-availability
  substitution for the example only, not a code change.
- **Calendar trimming:** after forward-filling non-NSE series, any leading
  calendar dates where a series still has no observed *or* filled price
  (i.e. before that series' first print) are trimmed from the whole
  dataset, so every series has a valid price for the entire surviving
  window. This is not explicitly specified but is required for `MarketData`
  to have rectangular, gap-free return arrays.
- **OLS solver:** used SVD-based pseudo-inverse (`nalgebra`'s `svd().solve`)
  rather than a normal-equations solve, for numerical stability if factors
  are ever near-collinear over a given window.
- **Ledoit-Wolf target:** implemented the identity-target (not the
  single-index or constant-correlation target) variant of Ledoit & Wolf
  (2004), since the spec asks for "scaled identity" specifically; `rho_hat`
  is taken as 0, which holds for that target (no off-diagonal estimation
  error to correlate against).
- **Annualization:** applied in exactly one place —
  `model::annualize_matrix` / `annualize_scalar` — called only when
  building the final `Sigma` / `F` used by experiments; the OLS fit and
  Ledoit-Wolf shrinkage themselves operate entirely on daily returns.
- **`RATES_PROXY` sign:** defined as `^NSEBANK` return minus `^NSEI` return
  (bank-sector excess return over the market), matching the checkpoint spec
  verbatim.
- **FactorShock attribution vs. per-holding P&L:** both are reported in INR
  and are cross-checked as invariants (`sum(per_holding.pnl_inr) ==
  portfolio_pnl_inr` and `sum(factor_attribution_inr) ==
  portfolio_pnl_inr`), which hold exactly because the shock model is linear
  with no intercept term (noted explicitly in every trace's `outputs.note`).

## CvarRebalance — design note (not implemented this checkpoint)

**Formulation.** Rockafellar-Uryasev (2000) CVaR minimization as a linear
program over `S` historical (or simulated) return scenarios `r_s`, decision
weights `w`, and an auxiliary VaR variable `zeta`:

```
minimize   zeta + (1 / (S * (1 - alpha))) * sum_s u_s
subject to u_s >= -(r_s . w) - zeta,   u_s >= 0,   for all s
           sum_i w_i = 1
           0 <= w_i <= cap_i                          (long-only + per-name cap)
           sum_i |w_i - w0_i| <= tau                   (turnover limit, linearized)
```

The turnover constraint is linearized in the standard way: introduce
`w_i = w0_i + p_i - n_i` with `p_i, n_i >= 0`, and replace `|w_i - w0_i|`
with `p_i + n_i` in the turnover sum. Commission cost, `turnover * value *
commission_bps`, is either (a) subtracted from a separate expected-return
constraint if one is added later, or (b) reported alongside the CVaR
objective as a secondary output — it does not need to enter the LP objective
for a pure risk-minimization rebalance, only for a risk/cost-tradeoff
variant.

**Scenario source.** Propose **raw historical asset returns** (the same
trailing window as the factor model, e.g. 252 days) rather than
factor-model-simulated scenarios, for this checkpoint's follow-on:
historical scenarios need no distributional assumption on residuals and
directly reflect realized joint tail behavior (including the factor-model's
own unexplained co-movements, which a Gaussian factor-model resample would
understate). A factor-model-simulated variant (sampling factor shocks from
`F`, mapping through `B`, adding simulated idiosyncratic noise from `D`) is
a reasonable v2 for scenario augmentation when the historical window is
short, but should be a separate, explicitly-labeled scenario source in the
trace (`data_quality`/`model_params` would need a `scenario_source` field),
not silently blended with historical scenarios.

**Solver.** `good_lp` with the `clarabel` backend (pure Rust, no system
dependency on an external solver binary, keeps the whole compute layer
statically linkable) is preferred over `good_lp` + `highs` (C++ binding) or
a hand-rolled simplex, for build simplicity and to avoid adding a
non-Rust toolchain dependency to CI.

**Infeasibility reporting.** If the LP is infeasible (e.g. `per_name_cap`
too tight to reach `sum(w) = 1` under the turnover budget from `w0`), the
trace should report `outputs.status: "infeasible"` plus a `diagnostics`
field naming which constraint group was detected as binding/unsatisfiable
(cap sum vs. required weight, or turnover budget vs. distance from `w0` to
the feasible cap region) — computed by a small pre-solve feasibility check
(e.g., is `sum(min(cap_i, w0_i + tau_i_share))` >= 1) rather than by parsing
solver-specific infeasibility certificates, so the message stays
solver-agnostic if the backend changes later.
