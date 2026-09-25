# drift-risk-copilot

Portfolio risk copilot for BFSI users (AI Builder Cup 2026). A Rust
workspace: `compute` fetches market data, fits a factor risk model, and
runs portfolio-risk experiments, returning a fully-cited `EvidenceTrace`
for every result; `agent` turns a natural-language request into one of
those experiments via Gemini, runs it, and narrates the result back with a
verbatim-number grounding check; `server` is an axum binary (`/health`,
`/experiment`, `/ask`, plus an embedded single-page UI) that calls
`agent::pipeline::run`, with a Dockerfile and Cloud Run config to deploy
it. The backend is deployable and demo-ready as of this checkpoint.

## Workspace layout

```
crates/
  compute/   # data layer, factor model, experiments, evidence trace
    src/data.rs         Yahoo fetch + cache + NSE calendar alignment + log returns
    src/model.rs         OLS factor fits, Ledoit-Wolf shrinkage, stock covariance, regime-conditional F
    src/regime.rs          3-state Gaussian HMM (Baum-Welch, Viterbi) for market-regime detection
    src/experiments.rs    FactorShock, RiskDecomposition
    src/cvar.rs            CvarRebalance: Rockafellar-Uryasev LP via good_lp + clarabel
    src/trace.rs          EvidenceTrace and its sub-structs
    src/bin/experiment.rs CLI: runs one experiment from a JSON file
    examples/              FactorShock / RiskDecomposition / CvarRebalance inputs, 10-stock Nifty portfolio
    tests/                  Synthetic-data unit/integration tests (no network required)
  agent/     # NL -> Experiment -> EvidenceTrace -> grounded narration
    src/gemini.rs        Async Gemini client: request/response types, retrying HTTP transport
    src/schema.rs         JSON Schema (via schemars) for the three per-experiment function declarations
    src/parse.rs           NL -> Experiment via a single Gemini function-calling turn
    src/narrate.rs          EvidenceTrace -> plain-language narration via Gemini
    src/grounding.rs        Verbatim-number check on narration vs. trace, with retry
    src/pipeline.rs          agent::pipeline::run: the one function `server` calls
    examples/demo_pipeline.rs  One-off demo: mocked Gemini + a real compute call (see below)
    tests/                    Mocked-Gemini unit/integration tests (no network to Gemini)
  server/    # axum HTTP API + embedded UI
    src/main.rs           Router, tracing setup, GEMINI_API_KEY startup check
    src/routes.rs          /health, /experiment, /ask, static-file fallback handlers
    src/backend.rs          Backend trait (RealBackend wraps compute+agent) + error mapping
    src/validate.rs          Portfolio validation shared by /experiment and /ask
    src/error.rs             ApiError ({error, code} JSON responses) + AppJson extractor
    src/logging.rs            Request logging middleware (method, path, status, latency)
    src/tests.rs               Route tests against a MockBackend (no network)
    static/index.html         Embedded single-page UI (include_str!, no build step)
data/cache/  # cached raw price CSVs (gitignored; fetched on first run)
Dockerfile     # multi-stage build -> gcr.io/distroless/cc-debian12
cloudrun.yaml  # Cloud Run service config
```

## Running an experiment

```
cargo run -p compute --bin experiment -- crates/compute/examples/factor_shock_nifty10.json
cargo run -p compute --bin experiment -- crates/compute/examples/risk_decomposition_nifty10.json
cargo run -p compute --bin experiment -- crates/compute/examples/cvar_rebalance_nifty10.json
```

Add `--refresh` to refetch price series instead of reading `data/cache/`.
Each run prints a JSON `EvidenceTrace` to stdout.

> Note: the frequency comparison below was run before the `GOLD` factor
> label was renamed to `GOLD_USD` (see "Trace additions for
> explainability"); `GOLD` in this section refers to what the trace now
> calls `GOLD_USD`. The numbers themselves are unaffected by the rename.

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

## Trace additions for explainability

`FactorShockOutput` (nested in every `FactorShock` `EvidenceTrace`) now
also carries:

- `factor_correlation`: the fit-time factor correlation matrix
  (`FACTOR_NAMES` order), so a reader can see e.g. MARKET/BRENT correlation
  without recomputing it from the covariance.
- `conditional_coefficients`: `F_uk * F_kk^-1`, keyed
  `implied_factor -> { given_factor: coefficient }`. Each implied move is
  exactly `Sum_k coefficient_k * given_log_shock_k`, so a reader can
  attribute, say, "why did GOLD_USD move -6.6% log?" to specific coefficient
  x given-shock products instead of trusting an opaque number.
- `gold_inr_implied_move`: gold priced in INR is `GOLD_USD * USDINR`, so its
  log return is the sum of the `GOLD_USD` and `USDINR` log shocks (given or
  implied, whichever applies). The fitted `GOLD_USD` factor alone excludes
  the rupee move a domestic gold holder actually realizes, so this is
  reported as a separate, clearly-labelled derived field rather than
  folded into `GOLD_USD`.
- `model_params.frequency` / `model_params.window_periods` (added in the
  frequency changes above) record which frequency/window the fit used.

The `GOLD` factor label is renamed `GOLD_USD` everywhere (`FACTOR_NAMES`,
`factor_returns` keys, `shocks_pct` keys, betas, trace output) to make
explicit that it is USD-denominated gold, not INR-denominated gold — see
`gold_inr_implied_move` above for the INR-denominated derived move. The
Yahoo ticker constant (`data::GOLD`, `"GC=F"`) is unchanged; only the
factor *label* moved.

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
| `reqwest` (blocking + async, `rustls-tls`, `default-features = false`) | HTTP client for Yahoo Finance's chart endpoint (blocking, `compute`) and Gemini's `generateContent` endpoint (async, `agent`). Switched from the default `native-tls`/OpenSSL backend to `rustls-tls` in the server checkpoint — see "Docker" below for why. |
| `csv` | Reading/writing the on-disk price cache. |
| `clap` (derive) | CLI argument parsing (`--refresh`, `--cache-dir`) for the `experiment` binary — not in the original justified list, added because the spec requires a `--refresh` flag and hand-rolled arg parsing would be worse than a one-line derive. |
| `good_lp` (`clarabel` backend only, `default-features = false`) | The Rockafellar-Uryasev LP for `CvarRebalance`. `clarabel` is a pure-Rust interior-point solver (no C/C++ toolchain or system solver binary needed), matching the design note's preference for build simplicity over a HiGHS/CBC binding. |
| `tokio` (`rt-multi-thread`, `macros`, `time`) | Async runtime for `agent`'s Gemini calls (`reqwest`'s async client) and retry backoff (`tokio::time::sleep`); `rt-multi-thread`/`macros` also back `#[tokio::main]`/`#[tokio::test]`. |
| `async-trait` | `agent::gemini::GeminiClient` is an async trait (needed so `parse`/`narrate`/`pipeline` can be generic over a real HTTP client or a test mock); stable Rust doesn't yet support `async fn` in traits used as trait objects/generically without this. |
| `regex` | Number extraction in `agent::grounding` (lakh/percent/plain numeric tokens) — a hand-rolled parser would be far more error-prone for this than a well-tested regex engine. |
| `axum` | The HTTP framework for `server` — spec-named, and a natural fit given `tokio`/`tower` are already in the dependency tree via `reqwest`/`agent`. |
| `tracing` / `tracing-subscriber` (`json`, `env-filter`) | Structured JSON request logging (method, path, status, latency), `RUST_LOG`-overridable level, per spec. |
| `tower` (dev-only, `util`) / `http-body-util` (dev-only) | `ServiceExt::oneshot` and response-body reading for `server`'s route tests, run in-process against the axum `Router` with no real network listener. |

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

## CvarRebalance

Implemented in `src/cvar.rs` via `good_lp` + the `clarabel` backend (pure
Rust, no external solver binary), per the design note above with these
amendments:

**Formulation.** Rockafellar-Uryasev CVaR minimization, but with the
`1/(S(1-beta))` coefficient replaced by `1/k` where `k =
round(S*(1-beta))` is an **integer tail scenario count** rather than the
continuous `S(1-beta)`. For equally-weighted historical scenarios this
makes the LP exactly equivalent to "minimize the average of the k worst
historical losses" — at optimum, `zeta*` is exactly the k-th worst loss
(VaR) and the objective is exactly the mean of the k worst losses (CVaR),
which is what lets the "LP objective == directly-computed CVaR" invariant
below hold to solver tolerance (~1e-10) rather than only approximately.

```
minimize   zeta + (1/k) * sum_s u_s
subject to u_s >= -(r_s . w) - zeta,   u_s >= 0,        for all s
           sum_i w_i = 1
           0 <= w_i <= per_name_cap                      (long-only + per-name cap)
           w_i = w0_i + buy_i - sell_i,  buy_i, sell_i >= 0
           sum_i (buy_i + sell_i) <= turnover_limit       (turnover, linearized)
```

**Scenarios.** Simple returns (`exp(log) - 1`) of the holdings' own
historical log returns (`data::MarketData.stock_returns`) — raw historical,
not factor-model-simulated, per the design note's recommendation. Full
available history by default; `window` (periods) is configurable.

**Historical VaR/CVaR, computed independently of the LP.** For a weight
vector `w` (before or after), `historical_stats` sorts the `k` worst
scenario losses `Loss_s = -(r_s . w)` directly from the scenario matrix and
reports `historical_var` (the k-th worst loss) and `historical_cvar` (their
mean) — a fresh computation from data + weights, not a copy of the solver's
reported objective, so `lp_objective_cvar` and `stats_after.historical_cvar`
are independent cross-checks of each other (see invariants below).

**Pre-solve feasibility check** (before ever calling the solver):
1. `per_name_cap * n_stocks >= 1` — otherwise long-only weights can never
   sum to 1.
2. A necessary lower bound on turnover: names already over `per_name_cap`
   must sell down to it, and — since weights must still sum to 1 — that
   sold capital must be bought back elsewhere, so turnover is at least
   `2 * sum_i max(0, w0_i - per_name_cap)`. If that exceeds `turnover_limit`,
   report infeasible without solving. (This is a *necessary*, not
   *sufficient*, condition; the LP solve remains the authoritative
   feasibility check for anything this doesn't catch.)

Either pre-solve failure, or the solver itself returning
`ResolutionError::Infeasible` / any other non-optimal status, produces a
structured result rather than a thrown error: `CvarRebalanceOutput.status`
(`"optimal"` | `"infeasible"` | `"solver_error"`) plus `diagnostics: Option<String>`,
both inside a normal `Ok(...)` `EvidenceTrace` — so a caller always gets a
citable trace, even for a failed rebalance, with the failing check recorded
as a `passed: false` invariant.

**Invariants:** weights sum to 1 (1e-9), turnover <= `turnover_limit` +
1e-6, and LP objective == directly-computed historical CVaR of the solution
(1e-6) — all three checked in `run_cvar_rebalance` and included in every
`EvidenceTrace.invariants`.

### Sample trace (10-stock example, cap 20%, turnover 30%, beta 0.95)

```
scenario_count: 1232, tail_scenario_count: 62
stats_before: { historical_var: 0.0136, historical_cvar: 0.0204 }
stats_after:  { historical_var: 0.0128, historical_cvar: 0.0187 }
lp_objective_cvar: 0.018721952952451708   (matches historical_cvar to 1.4e-14)
turnover: 0.300 (binding at the limit)
commission_cost_inr: 3000.0  (0.30 * 1e7 * 10bps)
weights_after: TMPV.NS -> ~0 (cut essentially to zero), ITC.NS -> 0.181,
               BHARTIARTL.NS -> 0.139 (both bid up), others near-unchanged
invariants: all 3 passed
```

Full trace: `cargo run -p compute --bin experiment -- crates/compute/examples/cvar_rebalance_nifty10.json`.

### Judgment calls specific to CvarRebalance

- **`commission_bps` default:** 10 bps (0.10%), a reasonable blended
  estimate for Indian equity delivery trades (brokerage + STT + other
  statutory charges); always caller-overridable, no default was specified
  in the brief.
- **`per_name_cap` applies uniformly to every name**, including in the
  `n * cap >= 1` feasibility check — with few holdings and a tight cap,
  this can force residual weight onto a name the optimizer would otherwise
  zero out (confirmed in `heavy_tail_asset_is_cut_to_near_zero_when_turnover_allows`,
  where the cap had to be raised to 1.0 to let the test isolate the
  CVaR-driven effect from the cap-driven one).
- **`model_params.factor_names`/`shrinkage_intensity` don't apply** to this
  experiment (no factor model is fit); left as an empty vec / 0.0 rather
  than adding an experiment-specific trace variant, with an explicit note
  in `outputs.note` saying so.

## Agent pipeline (`agent::pipeline::run`)

`compute` is unchanged in this checkpoint. `agent` adds the NL -> Experiment
-> EvidenceTrace -> grounded-narration pipeline `server` will call:

```
agent::pipeline::run(client, user_message, portfolio)
  -> parse::parse_experiment   (1 Gemini call, function-calling)
  -> compute_trace              (real compute::data/model/experiments/cvar call,
                                  on a blocking thread via tokio::task::spawn_blocking)
  -> grounding::grounded_narrate (1-3 Gemini calls: narrate, then up to 2 grounding retries)
```

### Gemini client (`agent::gemini`)

`GeminiClient` is an async trait with one method, `generate`; `HttpGeminiClient`
is the real POST-to-`generateContent` implementation (API key from
`GEMINI_API_KEY`, retrying up to 3 attempts with exponential backoff — 250ms,
500ms — on HTTP 429/503, surfacing any other status as `GeminiError::Status`).
Being a trait (not a concrete struct) is what makes `parse`/`narrate`/`pipeline`
testable without network access: tests supply a `MockGeminiClient` with a
queue of canned responses instead.

### Schema (`agent::schema`)

`experiment_json_schema()` derives a JSON Schema from `compute::experiments::Experiment`
via `schemars::schema_for!`, then strips the `portfolio` field from every
variant's `properties`/`required` (recursively, including inside `definitions`)
before it's shown to Gemini — see judgment calls below for why.

### Grounding check (`agent::grounding`)

The core piece. `extract_numbers` recognizes three token shapes, tried in
priority order (most specific first) via one alternation-based regex, so
e.g. `"₹11.7 lakh"` is consumed whole rather than also matching `"11.7"`
generically:

1. **Lakh:** `(?:₹\s*)?(-?[\d,]+(?:\.\d+)?)\s*lakh\b` → `value * 100_000`.
2. **Percent:** `(-?[\d,]+(?:\.\d+)?)\s*%` → `value / 100`.
3. **Plain:** `(?:₹\s*)?(-?[\d,]+(?:\.\d+)?)` → `value` as-is.

The minus sign accepts both ASCII `-` and Unicode `−` (U+2212), since
Gemini (and Indian financial prose generally) uses both; commas are
stripped before parsing, which normalizes both Western (`1,175,389`) and
Indian (`11,75,389`) digit grouping identically. `numeric_leaves` flattens
every numeric JSON leaf out of `serde_json::to_value(&trace)` (recursively;
strings/bools/nulls ignored). A narration number matches a trace number if
`|a - b| / max(|a|, |b|, 1e-9) <= 0.02` (the `1e-9` floor avoids
division-by-zero when both are ~0, without changing behavior anywhere the
spec's 2% figure actually matters).

`grounded_narrate` calls `narrate`, checks, and — if any number is
unmatched — retries up to twice with the failing tokens named in an
appended system instruction, per the spec's exact retry wording. If still
failing after 2 retries (3 calls total), it returns the last narration with
`grounding_warnings` populated rather than suppressing the response.

### Sample `PipelineResult` — FactorShock, live 10-stock portfolio

No `GEMINI_API_KEY` is available in this environment, so `agent::pipeline::run`
below used a **scripted mock Gemini client** (`crates/agent/examples/demo_pipeline.rs`)
for the two Gemini calls, while the compute step hit live Yahoo data exactly
as the `experiment` CLI does. The narration text was written by hand,
honoring the narrate system prompt's rules, then run through the *real*
`grounding::check_grounding` (not mocked) — this is a demonstration of the
grounding machinery on a genuine trace, not a live Gemini call:

```
$ cargo run -p agent --example demo_pipeline
```

**Parsed experiment:** `FactorShock { shocks_pct: {MARKET: -12.0, BRENT: 20.0}, propagate: true, ... }`
(portfolio injected from the caller, not from Gemini's args).

**Trace summary** (`trace.outputs.result`, full JSON via the command above):

```
given_shocks:    MARKET  -12.00% (simple)     BRENT  +20.00% (simple)
implied_shocks:  USDINR  +2.53%                GOLD_USD  -6.36%             RATES_PROXY  -1.80%
portfolio_pnl_inr: -1,177,846.39
invariants: sum(per_holding.pnl_inr) == portfolio_pnl_inr        -> passed
            sum(factor_attribution_log_inr) == portfolio_log_pnl_inr -> passed
```

**Narration:**

> A -12% shock to MARKET combined with a +20% shock to BRENT produces a
> portfolio loss of approximately -1,177,846 INR on this ten-stock Nifty
> portfolio. Because the user specified only these two factors, the
> remaining three factors are model-estimated from this portfolio's return
> history via the factor covariance: USDINR is implied to move +2.53%,
> GOLD_USD -6.36%, and RATES_PROXY -1.80%, each shown separately from the
> two given shocks above. These implied moves are not user inputs; they
> follow from the historical correlation between MARKET, BRENT and the
> other factors. The loss is dominated by the MARKET shock, given the
> portfolio's substantial equity beta exposure.

**`grounding_warnings`: `[]`** — every number in the narration matched a
trace value on the first attempt; no retry was needed. See "flag
immediately if grounding_warnings fires" below.

### Judgment calls

- **`portfolio` stripped from the schema Gemini sees**, not just documented
  in the function description: with `Portfolio` present as a required field
  in each variant's schema but the model told "don't extract this," a small
  model can still feel obligated to invent a plausible-looking (wrong)
  portfolio object, wasting tokens and risking a parse failure if its
  shape is malformed. Stripping it removes the temptation entirely; the
  real portfolio is always spliced into the function-call args
  (`args["portfolio"] = ...`) before deserializing into `Experiment`,
  overwriting whatever Gemini did or didn't include.
- **2% relative tolerance, not absolute:** an absolute tolerance would be
  either too loose for small numbers (betas, shrinkage intensities are
  often < 0.1) or too tight for large INR amounts (portfolio values in the
  millions), so every comparison is scaled by the larger of the two
  magnitudes (floored at `1e-9` to stay finite at/near zero).
- **Number-matching, not phrase-matching:** the grounding check verifies
  every *number* the narration states is real, not that the *sentence*
  containing it is accurate (e.g. it can't catch a narration that swaps
  which factor a correct number belongs to). This matches the spec's
  literal ask ("every number... must appear verbatim") but is worth naming
  as a limitation — a stronger check would need entity/number pairing,
  out of scope here.
- **Blocking compute on `spawn_blocking`:** `compute`'s data/model/CVaR
  path is synchronous (blocking `reqwest`, CPU-bound linear algebra/LP
  solve); `pipeline::run` moves it to a blocking thread rather than making
  `compute` itself async, since `compute` has no other reason to depend on
  an async runtime and the CLI (`experiment`) needs to stay synchronous too.
- **`commission_bps`/`window`/`frequency` defaults are unchanged** from the
  compute-layer checkpoints; `agent` doesn't override or second-guess them.

## Server (`server::main`)

```
GET  /health        -> { status: "ok", version } (200)
POST /experiment     -> { portfolio, experiment } in, EvidenceTrace out
POST /ask              -> { portfolio, message } in, AskResponse out
GET  /*                  -> embedded single-page UI (static/index.html)
```

`ExperimentRequest.experiment` is the tagged `Experiment` variant's JSON
*minus* `portfolio` (e.g. `{"type": "FactorShock", "shocks_pct": {...}}`) —
the same "caller supplies portfolio separately" pattern as `agent::parse`
(§ Agent pipeline). The route handler splices `req.portfolio` into that
JSON before deserializing into `compute::experiments::Experiment`, so a
caller never has to repeat the portfolio inside the experiment object.

**Validation** (`validate::validate_portfolio`, shared by both POST routes):
at least 2 holdings, every weight > 0, `total_value_inr` > 0, weights sum
to `1.0 +/- 0.01`. Tickers are deliberately not checked — `compute::data`
already errors clearly on a bad one, and a second ticker-format check here
would just be one more place to keep in sync. Every failure returns 400
with `{"error": "...", "code": "invalid_portfolio"}`.

**Error mapping** (`backend::BackendError` -> `error::ApiError`):

| Backend error | HTTP status | `code` |
|---|---|---|
| `ParseError::Unrecognised` (the model's one-sentence explanation) | 422 | `unrecognised_request` |
| any `compute::ComputeError` | 500 | `compute_error` |
| any Gemini error other than a missing API key (already refused at startup) | 503 | `gemini_unavailable` |
| malformed/missing-field JSON body (`AppJson`'s rejection) | 400 | `invalid_json` |
| anything else unexpected | 500 | `internal_error` |

**Testability:** routes depend on a `Backend` trait (`run_experiment`,
`run_ask`), not on `compute`/`agent` directly. `RealBackend` wraps live
calls (via `tokio::task::spawn_blocking` for the blocking compute path);
`src/tests.rs` supplies a `MockBackend` with canned results, so the 5
required tests need no network access to Yahoo Finance or Gemini.

**Logging:** one `tracing::info!` event per request (`method`, `path`,
`status`, `latency_ms`), emitted by a small `axum::middleware::from_fn`
wrapper rather than `tower_http::trace::TraceLayer`, so the exact fields
logged match the spec precisely instead of `TraceLayer`'s span-based
defaults. `tracing_subscriber::fmt().json()`, level from `RUST_LOG`
(default `info`).

### UI (`static/index.html`)

Single file, embedded via `include_str!` — no build step, no npm, no
external fonts/scripts, plain system fonts, `#2563EB` as the one accent
colour, two-column layout above 900px. Portfolio builder (add/remove
ticker/weight rows + a client-side Validate button running the same rule
as the server's `validate_portfolio`), a prompt box with a Direct toggle
that reveals structured params per experiment type, a submit button with a
spinner, a narration panel (yellow banner when `grounding_warnings` is
non-empty, exact wording per spec), and a collapsible evidence-trace
`<pre>` block with copy-to-clipboard.

## Sample responses

### `GET /health`

```json
{"status":"ok","version":"0.1.0"}
```

(Live, from a running binary — `GEMINI_API_KEY=<dummy> PORT=8099 ./target/debug/server`.)

### `POST /ask` (mocked pipeline — see the agent checkpoint's note on no live Gemini access)

Same approach as the agent checkpoint's demo: `src/tests.rs` used a
`MockBackend` returning a `PipelineResult` built from a real FactorShock
input and a hand-written, grounding-checked narration.

```json
{
  "experiment": {
    "type": "FactorShock",
    "portfolio": {
      "holdings": [
        {"ticker": "RELIANCE.NS", "weight": 0.6},
        {"ticker": "TCS.NS", "weight": 0.4}
      ],
      "total_value_inr": 1000000.0
    },
    "shocks_pct": {"BRENT": 20.0, "MARKET": -12.0},
    "propagate": true,
    "linear_approximation": false,
    "frequency": "Daily",
    "window": null
  },
  "trace": {
    "experiment": "RiskDecomposition",
    "data_window": {"frequency": "Daily", "window_periods": 252, "start": "2025-09-16", "end": "2026-09-24"},
    "data_quality": {"date_range_start": "2021-09-27", "date_range_end": "2026-09-24", "trading_days": 1234, "per_series": []},
    "model_params": {"frequency": "Daily", "window_periods": 252, "factor_names": ["MARKET"], "shrinkage_intensity": 0.0374, "annualization_factor": 252.0},
    "inputs": {},
    "outputs": {"result": {"portfolio_vol_annualized": 0.1552}},
    "invariants": [],
    "engine_version": "0.1.0"
  },
  "narration": "A -12% shock to MARKET combined with a +20% shock to BRENT produces a portfolio loss on this portfolio. USDINR, GOLD_USD and RATES_PROXY move as implied, model-estimated moves, shown separately from the two given shocks.",
  "grounding_warnings": []
}
```

(`trace` here is a small stand-in fixture from the test suite, not a full
live trace — the point of this response is the `AskResponse` *shape*, not
new trace content; a full trace looks exactly like the ones in
`crates/compute/examples/*.json`.)

## Docker

```
docker build -t drift-risk-copilot:server .
docker run -e GEMINI_API_KEY=<key> -p 8080:8080 drift-risk-copilot:server
```

Verified locally (`docker build` + `docker run` + a real `GET /health`
against the running container). Final image: **~12.5MB** (`docker save
drift-risk-copilot:server | wc -c` = 13,155,328 bytes; `docker images`'
own ~62MB figure includes buildx provenance/attestation metadata that
isn't part of the runtime image) — comfortably under the 50MB target.

### Build failures hit along the way (reported verbatim, per instructions)

**1. `rust:1.82-slim` as originally specified:**

```
error: failed to parse manifest at `/usr/local/cargo/registry/.../clap_lex-1.1.1/Cargo.toml`
Caused by:
  feature `edition2024` is required
  The package requires the Cargo feature called `edition2024`, but that feature is not
  stabilized in this version of Cargo (1.82.0 (8f40fc59f 2024-08-21)).
```

**2. `rust:1.85-slim`** (edition2024 stabilized in Cargo 1.85, tried next):

```
error: rustc 1.85.1 is not supported by the following packages:
  icu_collections@2.3.0 requires rustc 1.88
  icu_locale_core@2.3.0 requires rustc 1.88
  icu_normalizer@2.3.0 requires rustc 1.88
  ... (icu_normalizer_data, icu_properties, icu_properties_data, icu_provider @ rustc 1.88)
  idna_adapter@1.2.2 requires rustc 1.86
```

**3. `rust:1.90-slim`** — built successfully.

### Judgment calls

- **Rust version bumped from 1.82 to 1.90 (not the spec's exact pin).**
  This `Cargo.lock` was generated with a current toolchain (rustc 1.95),
  which resolved several transitive dependencies (`clap_lex`, the
  `icu_*` family via `idna`/`url`) to versions with an MSRV well above
  1.82. Re-pinning those dependencies to older, 1.82-compatible versions
  was the other option, but would mean carrying a second, artificially
  old dependency set for Docker only, diverging from what's actually
  tested locally — worse than moving the build image forward to a
  version that supports the lockfile that's actually shipped. 1.90 is
  the smallest bump that got a clean build in this environment (verified
  by trying 1.82, then 1.85, then 1.90 — see above).
- **`rustls-tls` instead of the default `native-tls`/OpenSSL backend for
  `reqwest`** (workspace-wide, both `compute` and `agent`): this is what
  actually makes `gcr.io/distroless/cc-debian12` viable as specified.
  `distroless/cc` ships glibc + libstdc++ but **no OpenSSL** — a normal
  `reqwest` build (native-tls) dynamically links `libssl.so`/`libcrypto.so`
  at runtime and would fail to start in that image. The alternative the
  checkpoint explicitly offered — static musl linking — would need either
  `openssl-sys`'s `vendored` feature (a full C build of OpenSSL inside
  the musl cross-build, plus a musl target + `musl-tools`) or switching to
  `rustls-tls` anyway; since `rustls-tls` alone already solves the
  OpenSSL-in-distroless problem with a normal glibc build and zero extra
  toolchain setup, it's the cleaner of the two options the spec allowed
  ("pick whichever is cleaner"). Confirmed working end-to-end: `--refresh`
  against live Yahoo Finance still succeeds after the switch (rustls
  validates Yahoo's cert chain fine via `webpki-roots`).
- **The `touch` before the final `cargo build` in the Dockerfile is
  load-bearing, not decorative** (see the Dockerfile's own comment for the
  full story): without it, `cargo build --release -p server` after copying
  the *real* source over the dummy-stub source finished in 0.08s doing
  nothing, and the resulting image's `/server` silently ran the dummy
  `fn main() {}` — exit code 0, no log output, port never opened. Caught by
  actually running the built container and hitting `/health`, not by
  trusting a "Finished" message. `find ... -exec touch {} +` on the real
  source before rebuilding fixes it (BuildKit's `COPY` doesn't always
  advance mtimes past cargo's fingerprint records from the earlier dummy
  build).
- **Dummy stubs cover every declared target in every workspace member's
  Cargo.toml** (`compute`'s `lib.rs` *and* its `bin/experiment.rs`,
  `agent`'s `lib.rs`, `server`'s `main.rs`), not just the crate(s) actually
  being built — Cargo parses every workspace member's manifest (for
  lockfile/dependency-graph resolution) even when building a single
  package with `-p`, and errors if a declared target's source file is
  missing.

## Cloud Run deploy

```
gcloud run services replace cloudrun.yaml --region=asia-south1
gcloud run services add-iam-policy-binding drift-risk-copilot \
  --region=asia-south1 --member=allUsers --role=roles/run.invoker
```

`cloudrun.yaml` references the image as `gcr.io/PROJECT_ID/drift-risk-copilot:latest`
(substitute the real project ID, and push the image built above to that
path first) and reads `GEMINI_API_KEY` from Secret Manager
(`secretKeyRef: {name: drift-gemini-key, key: latest}` — the secret must
exist and the Cloud Run service's runtime service account needs
`roles/secretmanager.secretAccessor` on it before `services replace` will
succeed). `minScale: 0` / `maxScale: 3`, 512Mi/1 CPU, `timeoutSeconds: 300`
(bumped from the originally-specified 60 -- confirmed live that a real
`/ask` request hit Cloud Run's own gateway timeout at exactly 60.1s: `/ask`
can chain up to 4 sequential Gemini round trips -- parse, then narrate,
then up to 2 grounding-retry narrate calls, each itself retrying up to 3x
internally on 429/503 -- which routinely exceeds 60s under real API
latency/rate-limiting, well beyond just the cold-start compute call the
original 60s was sized for).

## Regime-conditional factor covariance (`compute::regime`, `model::ModelConfig`)

A 3-state Gaussian HMM (`compute::regime`, Baum-Welch fit from scratch, no
external HMM crate) on Nifty (`^NSEI`) daily log returns, used to split the
factor covariance `F` by market regime instead of always pooling the full
window. Default is unchanged behaviour (`regime_covariance: false`
everywhere) — this is purely opt-in.

### The model

- **States, always reported in ascending-emission-variance order**: 0 =
  Bull (lowest vol), 1 = Bear (medium), 2 = Crisis (highest). The relabel
  happens once, after Baum-Welch converges, by sorting the three fitted
  states by their final emission variance — so which *internal* state index
  the EM fit happened to land on for "the high-vol regime" never matters;
  the label always does.
- **Forward/backward**: scaled (Rabiner 1989) — `alpha_hat_t(k)` normalized
  to sum to 1 at every `t`, with `log P(O) = sum_t ln(c_t)`; `beta` scaled
  by the *same* `c` array so `alpha_hat_t(k) * beta_t(k) == gamma_t(k)`
  exactly, verified by a test that checks this sums to 1 (within 1e-9) at
  every `t`, not just the final one.
- **Convergence**: log-likelihood improvement `< 1e-6` or 500 iterations.
- **Viterbi**: log-space, for the full regime-assignment sequence used to
  split factor returns.
- `smoothed_probs` (`gamma_T`, the *current* regime distribution) come with
  `smoothing_note: "full-history smoothed, not suitable for live trading
  signals"` — a full-history smoother uses future information (everything
  up to `T`) to estimate the state at `T`, which is fine as a point-in-time
  snapshot but not what a causal, real-time signal would look like.

### Wiring into the factor model

`model::ModelConfig { window, frequency, regime_covariance }` replaces the
old bare `(window, frequency)` pair for the regime-aware path
(`fit_factor_model_with_config`); the original `fit_factor_model(data,
tickers, window, frequency)` is kept as a thin non-regime wrapper so
**nothing outside `compute`'s own CLI/experiments/cvar needed to change**
— `agent`/`server` still call the old signature and compile unmodified.

When `regime_covariance: true`: the HMM fits on the *same* window's `MARKET`
factor series (already Nifty's own log returns, no separate fetch), factor
returns are split by the Viterbi sequence, and each regime gets its own
Ledoit-Wolf `F_k`. `FactorModel.factor_covariance_daily` — the field every
existing `factor_covariance()`/`stock_covariance()` call already reads —
becomes `F_{current_regime}`, so **RiskDecomposition needed zero changes to
its own math**: it was already just calling those methods. All three
regimes' `F_k` remain available via `factor_covariance_for_regime(k)` /
`stock_covariance_for_regime(k)`, which is what `FactorShock`'s
`crisis_comparison` uses.

**Fallback**: a regime with fewer than `MIN_REGIME_OBSERVATIONS` (30) days
in the window uses the full-window `F` instead of its own (too few
observations to shrink meaningfully), and a warning is recorded in
`model_params.regime_fallback_warnings`. This is a real, live-observed
case, not just a hypothetical — see the live run below.

### Per-experiment behaviour

- **RiskDecomposition**: `regime_covariance: bool` field; when true, vol
  and Euler contributions use `F_{current_regime}` automatically (see
  above). `model_params.regime_state` records which regime.
- **FactorShock**: `regime_covariance: bool` field; when true *and* the
  current regime isn't already Crisis, `outputs.result.crisis_comparison`
  reruns the same shock (including conditional propagation) using
  `F_crisis` instead of `F_current`, so a reader can see "how much worse
  would this look under crisis-regime correlations" without a second
  request. Absent (not zeroed) when the current regime already is Crisis,
  since that comparison would be a no-op.
- **CvarRebalance**: `regime_covariance: bool` field; per the design note,
  the LP and feasibility checks always use historical scenarios directly,
  *never* a factor-model covariance — so this can't gate optimality. It
  instead fits a regime-conditional factor model purely to report
  `regime_portfolio_vol_annualized_{before,after}` (`sqrt(w' Sigma_regime
  w)` for `weights_before`/`weights_after`) as a parametric cross-check
  alongside the historical CVaR/VaR, with `model_params.regime_state`
  recording which regime.

### Live run (10-stock Nifty portfolio)

**HMM fit**: `n_iter: 329`, `log_likelihood: 884.996`.

**Current regime**: **Bull** — `smoothed_probs: [0.907, 0.090, 0.003]`
(Bull/Bear/Crisis), `obs_count_per_regime: [184, 10, 58]` (Viterbi, out of
the 252-day window).

**Fallback warning actually fired** (not just tested synthetically):
`"regime_1 (Bear) has only 10 observations, fell back to full-window
covariance"` — Bear was too thin a slice of this particular 252-day window
to shrink its own `F`.

**RiskDecomposition, `portfolio_vol_annualized`**:

| | value |
|---|---|
| `regime_covariance: false` | 0.1549 |
| `regime_covariance: true` (Bull) | 0.1174 |

Meaningfully lower under the Bull-regime `F` than the full-window `F`, as
expected — the window's Bear/Crisis days pull the full-window covariance up.

**FactorShock (Nifty −12%, Brent +20%, `regime_covariance: true`)**,
current regime Bull, so `crisis_comparison` is present:

| | current regime (Bull) | `crisis_comparison` |
|---|---|---|
| `portfolio_pnl_inr` | −1,224,017 | −1,164,320 |
| implied `USDINR` | +2.05% | +2.17% |
| implied `GOLD_USD` | −4.15% | −6.02% |
| implied `RATES_PROXY` | +0.02% | −2.38% |

Both invariants passed in both runs. Reproduce: `cargo run --bin
experiment -- crates/compute/examples/risk_decomposition_nifty10_regime.json`
/ `factor_shock_nifty10_regime.json`.

### Tests

`compute::regime`'s own unit tests (synthetic 3-segment low/medium/high-vol
data): `gamma` sums to 1 at every `t`; Viterbi recovers each segment with
>85% accuracy; state labels come out variance-ordered regardless of which
order the segments appear in the data (a deliberately *not* variance-sorted
order — high, low, medium). `model`'s own tests inject a synthetic Viterbi
sequence directly (rather than coaxing a real HMM fit into an unlucky
split) to test the <30-obs fallback deterministically. `experiment_tests.rs`
covers RiskDecomposition's vol actually differing with/without
`regime_covariance`, and FactorShock's `crisis_comparison` presence/absence
by regime. One test — `F` is PSD for all three regimes on real NSEI data —
needs network and is `#[ignore]`d by default; run with `cargo test -p
compute --test model_tests -- --ignored`. Confirmed passing.

### Judgment calls

- **k-means init clusters on `|returns|`, not raw signed returns** — found
  live, not anticipated: since all three regimes are roughly zero-mean,
  clustering on signed values just splits points by *direction*
  ("very negative" / "near zero" / "very positive"), which has nothing to
  do with volatility regime. This made Baum-Welch converge to a poor local
  optimum on the very first synthetic test run (two of three fitted states
  ended up with similar variances, differentiated mostly by mean, on data
  with three well-separated *true* variances and zero true mean
  everywhere). Clustering on magnitude fixed it immediately; final
  per-state means/variances are still computed from the original signed
  data within each magnitude-assigned cluster.
- **30-observation fallback threshold**: not derived from anything more
  principled than "Ledoit-Wolf shrinkage needs enough observations to
  estimate a 5x5 sample covariance's off-diagonal structure at all" — 30
  points for 5 factors is already a thin sample (6 obs/factor), but Ledoit-
  Wolf shrinkage is specifically designed to be robust in exactly that
  small-T regime (it's the paper's whole point), so this is closer to "no
  smaller than this" than a precisely justified number. The live run above
  shows it firing in practice (Bear regime, 10 obs), which is reassuring
  that the threshold isn't so low it never triggers.
- **CvarRebalance's regime output is informational-only by design**,
  per the checkpoint spec's explicit "not in the LP itself" — it would be
  straightforward to instead use `Sigma_regime` in the pre-solve
  feasibility checks too, but those checks don't reference any covariance
  at all currently (they're pure cap/turnover arithmetic), so doing that
  would be a bigger, unrequested change to what "feasible" means for this
  experiment.
- **Regime HMM window for CvarRebalance** defaults to
  `frequency.default_window()` (252 daily), independent of CvarRebalance's
  own `window` (which defaults to *full available history* for scenarios) —
  these are two different jobs (regime detection wants a recent window;
  historical CVaR wants as much data as possible), so tying them together
  would have been actively wrong.
- **`fit_factor_model` (old signature) kept as a thin wrapper** rather than
  changing its signature and updating every call site across `agent`/
  `server`, per this checkpoint's explicit scope ("no other crate changes
  in this session").
