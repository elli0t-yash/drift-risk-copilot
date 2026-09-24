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
