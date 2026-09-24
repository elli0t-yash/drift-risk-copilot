//! CvarRebalance: Rockafellar-Uryasev CVaR-minimizing long-only rebalance
//! over historical scenarios (raw simple returns of the holdings, per the
//! design note — not factor-model-simulated). See the README's "CvarRebalance"
//! section for the full formulation and the rationale for each choice below.

use std::collections::BTreeMap;

use good_lp::{clarabel, variable, Expression, ProblemVariables, ResolutionError, Solution, SolverModel};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::data::DataQuality;
use crate::error::{ComputeError, Result};
use crate::experiments::{log_to_simple, Portfolio};
use crate::model::Frequency;
use crate::trace::{DataWindow, EvidenceTrace, InvariantCheck, ModelParams};

fn default_confidence_level() -> f64 {
    0.95
}

/// Commission is charged on the traded (turnover) value; 10 bps (0.10%) is
/// a reasonable blended default for Indian equity brokerage + STT + other
/// statutory charges on a delivery trade, but is always caller-overridable.
fn default_commission_bps() -> f64 {
    10.0
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct CvarRebalanceInput {
    /// Current holdings (w0) and total portfolio value.
    pub portfolio: Portfolio,
    /// Confidence level beta for CVaR/VaR (e.g. 0.95 = worst 5% tail).
    #[serde(default = "default_confidence_level")]
    pub confidence_level: f64,
    /// Per-name maximum weight (fraction, e.g. 0.20 for 20%).
    pub per_name_cap: f64,
    /// Maximum turnover, Sum_i |w_i - w0_i| (fraction, both buys and sells
    /// counted; e.g. 0.30 allows up to 30% of the portfolio to trade).
    pub turnover_limit: f64,
    /// Commission rate in basis points, charged on `turnover * total_value_inr`.
    #[serde(default = "default_commission_bps")]
    pub commission_bps: f64,
    #[serde(default)]
    pub frequency: Frequency,
    /// Scenario window in periods; omit for the full available history.
    #[serde(default)]
    pub window: Option<usize>,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct CvarPortfolioStats {
    /// Historical VaR at `confidence_level`: the `tail_scenario_count`-th
    /// worst historical loss (decimal, positive = a loss).
    pub historical_var: f64,
    /// Historical CVaR at `confidence_level`: the mean of the
    /// `tail_scenario_count` worst historical losses (decimal, positive =
    /// a loss). Computed directly from the scenario matrix and a weight
    /// vector, independent of the LP's own reported objective value.
    pub historical_cvar: f64,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct CvarRebalanceOutput {
    /// "optimal" (LP solved), "infeasible" (pre-solve check or solver
    /// reported infeasible), or "solver_error" (any other non-optimal
    /// solver status).
    pub status: String,
    /// Human-readable explanation when `status != "optimal"`.
    pub diagnostics: Option<String>,
    pub confidence_level: f64,
    pub scenario_count: usize,
    /// `round(scenario_count * (1 - confidence_level))`, clamped to >= 1:
    /// how many worst-case historical scenarios the CVaR average is taken
    /// over.
    pub tail_scenario_count: usize,
    pub weights_before: BTreeMap<String, f64>,
    pub weights_after: Option<BTreeMap<String, f64>>,
    pub stats_before: CvarPortfolioStats,
    pub stats_after: Option<CvarPortfolioStats>,
    /// Sum_i |w_i - w0_i|, only present when `status == "optimal"`.
    pub turnover: Option<f64>,
    pub commission_cost_inr: Option<f64>,
    /// The LP's own optimal objective value (zeta* + (1/k) * Sum u_s*),
    /// reported alongside `stats_after.historical_cvar` (computed
    /// independently from the scenario matrix) so the two can be compared.
    pub lp_objective_cvar: Option<f64>,
}

/// Historical VaR/CVaR of the scenario-weighted loss distribution
/// `Loss_s = -Sum_i w_i * R[s][i]`, for the worst `k` scenarios.
fn historical_stats(scenarios: &[Vec<f64>], w: &[f64], k: usize) -> CvarPortfolioStats {
    let mut losses: Vec<f64> = scenarios
        .iter()
        .map(|row| {
            let portfolio_return: f64 = row.iter().zip(w.iter()).map(|(r, wi)| r * wi).sum();
            -portfolio_return
        })
        .collect();
    losses.sort_by(|a, b| b.partial_cmp(a).expect("returns are never NaN"));
    let k = k.min(losses.len()).max(1);
    let historical_var = losses[k - 1];
    let historical_cvar = losses[..k].iter().sum::<f64>() / k as f64;
    CvarPortfolioStats {
        historical_var,
        historical_cvar,
    }
}

pub fn run_cvar_rebalance(
    data_quality: &DataQuality,
    data: &crate::data::MarketData,
    input: &CvarRebalanceInput,
) -> Result<(CvarRebalanceOutput, EvidenceTrace)> {
    let tickers = input.portfolio.tickers();
    if tickers.is_empty() {
        return Err(ComputeError::InvalidInput(
            "portfolio must have at least one holding".to_string(),
        ));
    }
    let w0: Vec<f64> = input.portfolio.holdings.iter().map(|h| h.weight).collect();
    let n = tickers.len();

    let series: Vec<&Vec<f64>> = tickers
        .iter()
        .map(|t| {
            data.stock_returns
                .get(t)
                .ok_or_else(|| ComputeError::Model(format!("missing return series for {t}")))
        })
        .collect::<Result<Vec<_>>>()?;
    let total_obs = series[0].len();
    let window = input.window.unwrap_or(total_obs).min(total_obs).max(1);
    let start = total_obs - window;

    // scenarios[s][i] = simple return of stock i in scenario s.
    let scenarios: Vec<Vec<f64>> = (0..window)
        .map(|s| (0..n).map(|i| log_to_simple(series[i][start + s])).collect())
        .collect();
    let scenario_count = window;

    let beta = input.confidence_level;
    if !(0.0..1.0).contains(&beta) {
        return Err(ComputeError::InvalidInput(format!(
            "confidence_level must be in [0, 1), got {beta}"
        )));
    }
    let k = ((scenario_count as f64) * (1.0 - beta)).round().max(1.0) as usize;

    let stats_before = historical_stats(&scenarios, &w0, k);
    let weights_before: BTreeMap<String, f64> =
        tickers.iter().cloned().zip(w0.iter().copied()).collect();

    let data_window = DataWindow {
        frequency: input.frequency,
        window_periods: window,
        start: data.dates[data.dates.len() - window],
        end: *data.dates.last().unwrap(),
    };
    let model_params = ModelParams {
        frequency: input.frequency,
        window_periods: window,
        // CvarRebalance uses raw historical scenarios, not the fitted
        // factor model (see the design note); these fields don't apply.
        factor_names: Vec::new(),
        shrinkage_intensity: 0.0,
        annualization_factor: 1.0,
    };

    let make_trace = |output: &CvarRebalanceOutput, invariants: Vec<InvariantCheck>| -> Result<EvidenceTrace> {
        Ok(EvidenceTrace {
            experiment: "CvarRebalance".to_string(),
            inputs: serde_json::to_value(input)?,
            data_window: data_window.clone(),
            data_quality: data_quality.clone(),
            model_params: model_params.clone(),
            outputs: serde_json::json!({
                "result": output,
                "note": "Scenarios are simple returns (exp(log) - 1) of the holdings' own \
                         historical log returns, not factor-model-simulated. model_params' \
                         factor_names/shrinkage_intensity do not apply to this experiment.",
            }),
            invariants,
            engine_version: crate::trace::engine_version(),
        })
    };

    // --- Pre-solve feasibility check (necessary conditions; the LP solve
    // itself remains the authoritative feasibility check) ---
    let cap = input.per_name_cap;
    if cap <= 0.0 || cap > 1.0 {
        return Err(ComputeError::InvalidInput(format!(
            "per_name_cap must be in (0, 1], got {cap}"
        )));
    }
    if cap * (n as f64) < 1.0 - 1e-9 {
        let diagnostics = format!(
            "per_name_cap ({cap}) * n_stocks ({n}) = {:.4} < 1: even at every name's cap, \
             weights cannot sum to 1 under a long-only portfolio.",
            cap * (n as f64)
        );
        let output = infeasible_output(input, scenario_count, k, weights_before, stats_before, diagnostics.clone());
        let trace = make_trace(
            &output,
            vec![InvariantCheck {
                name: "pre-solve: per_name_cap * n_stocks >= 1".to_string(),
                passed: false,
                tolerance: 1e-9,
                detail: diagnostics,
            }],
        )?;
        return Ok((output, trace));
    }
    // Necessary (not sufficient) lower bound: names already over cap must
    // sell down to at least their cap, and that sold capital must be
    // bought back elsewhere to keep Sum w = 1, so turnover is at least
    // twice the total forced-sell amount.
    let min_required_sells: f64 = w0.iter().map(|&wi0| (wi0 - cap).max(0.0)).sum();
    if 2.0 * min_required_sells > input.turnover_limit + 1e-9 {
        let diagnostics = format!(
            "names over per_name_cap ({cap}) require selling at least {min_required_sells:.4} \
             of portfolio value, which must be redeployed elsewhere to keep weights summing to \
             1 -- a minimum turnover of {:.4}, exceeding turnover_limit ({}).",
            2.0 * min_required_sells,
            input.turnover_limit
        );
        let output = infeasible_output(input, scenario_count, k, weights_before, stats_before, diagnostics.clone());
        let trace = make_trace(
            &output,
            vec![InvariantCheck {
                name: "pre-solve: minimum turnover to satisfy caps <= turnover_limit".to_string(),
                passed: false,
                tolerance: 1e-9,
                detail: diagnostics,
            }],
        )?;
        return Ok((output, trace));
    }

    // --- Build and solve the LP ---
    let mut vars = ProblemVariables::new();
    let w: Vec<_> = (0..n).map(|_| vars.add(variable().min(0.0).max(cap))).collect();
    let buy: Vec<_> = (0..n).map(|_| vars.add(variable().min(0.0))).collect();
    let sell: Vec<_> = (0..n).map(|_| vars.add(variable().min(0.0))).collect();
    let zeta = vars.add_variable();
    let u: Vec<_> = (0..scenario_count).map(|_| vars.add(variable().min(0.0))).collect();

    let mut objective = Expression::from(zeta);
    for &uv in &u {
        objective += uv * (1.0 / k as f64);
    }

    let mut model = vars.minimise(objective).using(clarabel);

    let sum_w: Expression = w.iter().map(|&v| Expression::from(v)).sum();
    model = model.with(sum_w.eq(1.0));

    let turnover_expr: Expression = buy.iter().chain(sell.iter()).map(|&v| Expression::from(v)).sum();
    model = model.with(turnover_expr.leq(input.turnover_limit));

    for i in 0..n {
        // w_i - buy_i + sell_i = w0_i  <=>  w_i = w0_i + buy_i - sell_i
        let expr = Expression::from(w[i]) - Expression::from(buy[i]) + Expression::from(sell[i]);
        model = model.with(expr.eq(w0[i]));
    }

    for s in 0..scenario_count {
        // u_s + Sum_i R[s][i] * w_i + zeta >= 0
        let mut expr = Expression::from(u[s]) + Expression::from(zeta);
        for (&wi, &ri) in w.iter().zip(scenarios[s].iter()) {
            expr += wi * ri;
        }
        model = model.with(expr.geq(0.0));
    }

    let solution = match model.solve() {
        Ok(sol) => sol,
        Err(e) => {
            let (status, diagnostics) = classify_resolution_error(&e);
            let output = infeasible_output_with_status(
                input, scenario_count, k, weights_before, stats_before, status, diagnostics.clone(),
            );
            let trace = make_trace(
                &output,
                vec![InvariantCheck {
                    name: "solver status == optimal".to_string(),
                    passed: false,
                    tolerance: 0.0,
                    detail: diagnostics,
                }],
            )?;
            return Ok((output, trace));
        }
    };

    let w_star: Vec<f64> = w.iter().map(|&v| solution.value(v)).collect();
    let zeta_star = solution.value(zeta);
    let u_star: Vec<f64> = u.iter().map(|&v| solution.value(v)).collect();
    let lp_objective_cvar = zeta_star + u_star.iter().sum::<f64>() / k as f64;

    let weights_after: BTreeMap<String, f64> =
        tickers.iter().cloned().zip(w_star.iter().copied()).collect();
    let stats_after = historical_stats(&scenarios, &w_star, k);
    let turnover: f64 = w_star
        .iter()
        .zip(w0.iter())
        .map(|(after, before)| (after - before).abs())
        .sum();
    let commission_cost_inr =
        turnover * input.portfolio.total_value_inr * (input.commission_bps / 10_000.0);

    let sum_w_star: f64 = w_star.iter().sum();
    let tolerance = 1e-6;
    let invariants = vec![
        InvariantCheck::approx_eq("sum(weights_after) == 1", sum_w_star, 1.0, 1e-9),
        InvariantCheck {
            name: "turnover <= turnover_limit + tolerance".to_string(),
            passed: turnover <= input.turnover_limit + 1e-6,
            tolerance: 1e-6,
            detail: format!("turnover={turnover:.6} turnover_limit={}", input.turnover_limit),
        },
        InvariantCheck::approx_eq(
            "LP objective == directly-computed historical CVaR of the solution",
            lp_objective_cvar,
            stats_after.historical_cvar,
            tolerance,
        ),
    ];

    let output = CvarRebalanceOutput {
        status: "optimal".to_string(),
        diagnostics: None,
        confidence_level: beta,
        scenario_count,
        tail_scenario_count: k,
        weights_before,
        weights_after: Some(weights_after),
        stats_before,
        stats_after: Some(stats_after),
        turnover: Some(turnover),
        commission_cost_inr: Some(commission_cost_inr),
        lp_objective_cvar: Some(lp_objective_cvar),
    };
    let trace = make_trace(&output, invariants)?;
    Ok((output, trace))
}

fn classify_resolution_error(e: &ResolutionError) -> (String, String) {
    match e {
        ResolutionError::Infeasible => (
            "infeasible".to_string(),
            "solver reported the problem as infeasible".to_string(),
        ),
        other => ("solver_error".to_string(), format!("solver error: {other}")),
    }
}

fn infeasible_output(
    input: &CvarRebalanceInput,
    scenario_count: usize,
    k: usize,
    weights_before: BTreeMap<String, f64>,
    stats_before: CvarPortfolioStats,
    diagnostics: String,
) -> CvarRebalanceOutput {
    infeasible_output_with_status(
        input,
        scenario_count,
        k,
        weights_before,
        stats_before,
        "infeasible".to_string(),
        diagnostics,
    )
}

#[allow(clippy::too_many_arguments)]
fn infeasible_output_with_status(
    input: &CvarRebalanceInput,
    scenario_count: usize,
    k: usize,
    weights_before: BTreeMap<String, f64>,
    stats_before: CvarPortfolioStats,
    status: String,
    diagnostics: String,
) -> CvarRebalanceOutput {
    CvarRebalanceOutput {
        status,
        diagnostics: Some(diagnostics),
        confidence_level: input.confidence_level,
        scenario_count,
        tail_scenario_count: k,
        weights_before,
        weights_after: None,
        stats_before,
        stats_after: None,
        turnover: None,
        commission_cost_inr: None,
        lp_objective_cvar: None,
    }
}
