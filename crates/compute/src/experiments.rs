//! Experiments: the three portfolio-risk experiments the compute layer
//! exposes to the agent layer. `CvarRebalance` is a design note only (see
//! module docs at the bottom) and is not implemented in this checkpoint.

use std::collections::BTreeMap;

use nalgebra::{DMatrix, DVector};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::data::{DataQuality, FACTOR_NAMES};
use crate::error::{ComputeError, Result};
use crate::model::{annualize_scalar, FactorModel, Frequency};
use crate::trace::{DataWindow, EvidenceTrace, InvariantCheck, ModelParams};

/// One portfolio line: ticker plus portfolio weight (fraction of total
/// value, not percent).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Holding {
    pub ticker: String,
    pub weight: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Portfolio {
    pub holdings: Vec<Holding>,
    pub total_value_inr: f64,
}

impl Portfolio {
    pub fn tickers(&self) -> Vec<String> {
        self.holdings.iter().map(|h| h.ticker.clone()).collect()
    }

    pub fn weights(&self) -> DVector<f64> {
        DVector::from_iterator(
            self.holdings.len(),
            self.holdings.iter().map(|h| h.weight),
        )
    }
}

/// The three experiments the compute layer can run. `CvarRebalance` is a
/// design placeholder (see the module doc at the end of this file) and its
/// `run` path returns `ComputeError::Model` explaining it is unimplemented.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type")]
pub enum Experiment {
    FactorShock(FactorShockInput),
    RiskDecomposition(RiskDecompositionInput),
    CvarRebalance(CvarRebalanceInput),
}

// ---------------------------------------------------------------------
// (a) FactorShock
// ---------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct FactorShockInput {
    pub portfolio: Portfolio,
    /// Shocks as simple (user-facing) percent returns (e.g. -12.0 for
    /// -12%), keyed by factor name (one of `FACTOR_NAMES`: MARKET, USDINR,
    /// BRENT, GOLD_USD, RATES_PROXY).
    pub shocks_pct: BTreeMap<String, f64>,
    /// When true (default), unshocked factors receive their conditional
    /// expectation given the shocked factors, via the factor covariance.
    /// When false, unshocked factors are held at zero.
    #[serde(default = "default_true")]
    pub propagate: bool,
    /// When false (default), shocks are converted simple -> log before
    /// propagation and betas are applied in log-return space, converting
    /// back to simple returns (`exp(log_return) - 1`) for P&L. When true,
    /// reproduces the original checkpoint's behaviour: shocks are treated
    /// as literal decimal returns with no log conversion, and P&L is
    /// exactly linear in the shock (`value * (beta . shock)`).
    #[serde(default)]
    pub linear_approximation: bool,
    #[serde(default)]
    pub frequency: Frequency,
    /// Trailing window in periods at `frequency`. Defaults to
    /// `frequency.default_window()` (252 daily, 156 weekly) when omitted.
    #[serde(default)]
    pub window: Option<usize>,
}

fn default_true() -> bool {
    true
}

impl FactorShockInput {
    pub fn resolved_window(&self) -> usize {
        self.window.unwrap_or_else(|| self.frequency.default_window())
    }
}

/// A shock or implied move in both the user-facing simple-return form and
/// the log-return form used internally for propagation and beta
/// application. `simple = exp(log) - 1` and `log = ln(1 + simple)` exactly
/// (see `simple_to_log`/`log_to_simple`); both are carried in the trace so
/// nothing has to be recomputed to check the conversion.
#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct ShockValue {
    pub simple: f64,
    pub log: f64,
}

impl ShockValue {
    fn from_simple(simple: f64) -> Self {
        ShockValue {
            simple,
            log: simple_to_log(simple),
        }
    }

    fn from_log(log: f64) -> Self {
        ShockValue {
            simple: log_to_simple(log),
            log,
        }
    }
}

/// Simple return -> log return: `ln(1 + simple)`.
pub fn simple_to_log(simple: f64) -> f64 {
    (1.0 + simple).ln()
}

/// Log return -> simple return: `exp(log) - 1`.
pub fn log_to_simple(log: f64) -> f64 {
    log.exp() - 1.0
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct HoldingShockResult {
    pub ticker: String,
    pub value_inr: f64,
    /// beta . (full log shock vector). Equal to `simple_return` when
    /// `linear_approximation` is true (no log conversion applied).
    pub log_return: f64,
    /// `exp(log_return) - 1`, or (under `linear_approximation`) the same
    /// linear `beta . shock` value used directly with no conversion.
    pub simple_return: f64,
    pub pnl_inr: f64,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct FactorShockOutput {
    pub linear_approximation: bool,
    /// The shocks as given by the caller, in both simple and log form.
    pub given_shocks: BTreeMap<String, ShockValue>,
    /// Shocks implied by conditional expectation for factors the caller
    /// did not specify (empty if `propagate` is false, or if all factors
    /// were given), in both simple and log form.
    pub implied_shocks: BTreeMap<String, ShockValue>,
    pub per_holding: Vec<HoldingShockResult>,
    /// Sum of `per_holding.pnl_inr`: value * simple_return per holding,
    /// summed. Exact (not an approximation) regardless of
    /// `linear_approximation`.
    pub portfolio_pnl_inr: f64,
    /// Value-weighted sum of log returns, Sum_i value_i * log_return_i.
    /// This is what `factor_attribution_log_inr` sums to exactly (an exact
    /// identity in log-return space); it is *not* the same number as
    /// `portfolio_pnl_inr` once `exp(.) - 1` conversion is applied, because
    /// that conversion is convex, not linear. Equal to `portfolio_pnl_inr`
    /// under `linear_approximation`.
    pub portfolio_log_pnl_inr: f64,
    /// Attribution of `portfolio_log_pnl_inr` (not `portfolio_pnl_inr`) by
    /// factor: Sum_i value_i * beta_ik * log_shock_k. Exact in log-return
    /// space; see `portfolio_log_pnl_inr` doc for why it does not equal
    /// `portfolio_pnl_inr` outside `linear_approximation` mode.
    pub factor_attribution_log_inr: BTreeMap<String, f64>,
    /// Factor correlation matrix at fit time (`FACTOR_NAMES` order, matches
    /// `factor_correlation.factor_names`).
    pub factor_correlation: CorrelationMatrix,
    /// Conditional coefficients `F_uk * F_kk^-1`, keyed
    /// `implied_factor -> { given_factor: coefficient }`: each implied
    /// move's log shock is exactly `Sum_k coefficient_k * given_log_shock_k`
    /// over the given factors, so this is how much of each implied move is
    /// attributable to each given shock. Empty when `propagate` is false or
    /// no factors are both given and left unspecified.
    pub conditional_coefficients: BTreeMap<String, BTreeMap<String, f64>>,
    /// Gold priced in INR is `GOLD_USD * USDINR`, so its log return is
    /// `GOLD_USD`'s log shock plus `USDINR`'s log shock (given or implied,
    /// whichever applies to each). Reported because `GOLD_USD` alone (the
    /// fitted factor) does not include the rupee move a domestic gold
    /// holder actually realizes.
    pub gold_inr_implied_move: ShockValue,
}

/// A labelled square matrix, `FACTOR_NAMES` order, for JSON output.
#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct CorrelationMatrix {
    pub factor_names: Vec<String>,
    /// `rows[i][j]` = correlation between `factor_names[i]` and `factor_names[j]`.
    pub rows: Vec<Vec<f64>>,
}

/// Computes the conditional coefficients `F_uk * F_kk^-1` (unknown_idx x
/// known_idx). The conditional expectation is `s_u = coefficients * s_k`.
fn conditional_coefficients(
    f: &DMatrix<f64>,
    known_idx: &[usize],
    unknown_idx: &[usize],
) -> Result<DMatrix<f64>> {
    let f_kk = DMatrix::from_fn(known_idx.len(), known_idx.len(), |i, j| {
        f[(known_idx[i], known_idx[j])]
    });
    let f_uk = DMatrix::from_fn(unknown_idx.len(), known_idx.len(), |i, j| {
        f[(unknown_idx[i], known_idx[j])]
    });
    let f_kk_inv = f_kk.clone().try_inverse().ok_or_else(|| {
        ComputeError::Model(
            "factor covariance sub-block F_kk is singular; cannot condition".to_string(),
        )
    })?;
    Ok(f_uk * f_kk_inv)
}

/// Computes conditional expectations for the "unknown" factor block given
/// the "known" block, s_u = F_uk * F_kk^-1 * s_k.
fn conditional_expectation(
    f: &DMatrix<f64>,
    known_idx: &[usize],
    known_vals: &[f64],
    unknown_idx: &[usize],
) -> Result<(Vec<f64>, DMatrix<f64>)> {
    if known_idx.is_empty() || unknown_idx.is_empty() {
        return Ok((
            vec![0.0; unknown_idx.len()],
            DMatrix::zeros(unknown_idx.len(), known_idx.len()),
        ));
    }
    let coefficients = conditional_coefficients(f, known_idx, unknown_idx)?;
    let s_k = DVector::from_row_slice(known_vals);
    let s_u = &coefficients * s_k;
    Ok((s_u.iter().copied().collect(), coefficients))
}

pub fn run_factor_shock(
    data_quality: &DataQuality,
    data_window: DataWindow,
    model: &FactorModel,
    input: &FactorShockInput,
) -> Result<(FactorShockOutput, EvidenceTrace)> {
    let factor_names: Vec<String> = FACTOR_NAMES.iter().map(|s| s.to_string()).collect();

    for key in input.shocks_pct.keys() {
        if !factor_names.contains(key) {
            return Err(ComputeError::InvalidInput(format!(
                "unknown factor '{key}', expected one of {factor_names:?}"
            )));
        }
    }

    // "Computation space" values: simple (decimal) shocks directly under
    // `linear_approximation`, or log-converted shocks otherwise. Betas were
    // fit on log returns, so propagation and beta application are only
    // exact in log space; `linear_approximation` reproduces the original
    // (less correct, but simpler) checkpoint behaviour on request.
    let given_computation: BTreeMap<String, f64> = input
        .shocks_pct
        .iter()
        .map(|(k, v)| {
            let simple = v / 100.0;
            let value = if input.linear_approximation {
                simple
            } else {
                simple_to_log(simple)
            };
            (k.clone(), value)
        })
        .collect();

    let known_idx: Vec<usize> = factor_names
        .iter()
        .enumerate()
        .filter(|(_, n)| given_computation.contains_key(*n))
        .map(|(i, _)| i)
        .collect();
    let unknown_idx: Vec<usize> = (0..factor_names.len())
        .filter(|i| !known_idx.contains(i))
        .collect();
    let known_vals: Vec<f64> = known_idx
        .iter()
        .map(|i| given_computation[&factor_names[*i]])
        .collect();

    let mut full_shock = vec![0.0; factor_names.len()];
    for (i, v) in known_idx.iter().zip(known_vals.iter()) {
        full_shock[*i] = *v;
    }

    let mut implied_computation = BTreeMap::new();
    let mut conditional_coefficients_out: BTreeMap<String, BTreeMap<String, f64>> =
        BTreeMap::new();
    if input.propagate && !unknown_idx.is_empty() && !known_idx.is_empty() {
        let f = model.factor_covariance();
        let (implied, coefficients) =
            conditional_expectation(&f, &known_idx, &known_vals, &unknown_idx)?;
        for (idx, val) in unknown_idx.iter().zip(implied.iter()) {
            full_shock[*idx] = *val;
            implied_computation.insert(factor_names[*idx].clone(), *val);
        }
        for (ui, &unknown_factor_idx) in unknown_idx.iter().enumerate() {
            let mut row = BTreeMap::new();
            for (ki, &known_factor_idx) in known_idx.iter().enumerate() {
                row.insert(factor_names[known_factor_idx].clone(), coefficients[(ui, ki)]);
            }
            conditional_coefficients_out.insert(factor_names[unknown_factor_idx].clone(), row);
        }
    }

    let to_shock_value = |v: f64| -> ShockValue {
        if input.linear_approximation {
            ShockValue::from_simple(v)
        } else {
            ShockValue::from_log(v)
        }
    };
    let given_shocks: BTreeMap<String, ShockValue> = given_computation
        .iter()
        .map(|(k, v)| (k.clone(), to_shock_value(*v)))
        .collect();
    let implied_shocks: BTreeMap<String, ShockValue> = implied_computation
        .iter()
        .map(|(k, v)| (k.clone(), to_shock_value(*v)))
        .collect();

    let log_shock_at = |idx: usize| -> f64 {
        if input.linear_approximation {
            simple_to_log(full_shock[idx])
        } else {
            full_shock[idx]
        }
    };
    let gold_idx = factor_names.iter().position(|n| n == "GOLD_USD").unwrap();
    let usdinr_idx = factor_names.iter().position(|n| n == "USDINR").unwrap();
    let gold_inr_implied_move =
        ShockValue::from_log(log_shock_at(gold_idx) + log_shock_at(usdinr_idx));

    let corr = model.factor_correlation();
    let factor_correlation = CorrelationMatrix {
        factor_names: factor_names.clone(),
        rows: (0..factor_names.len())
            .map(|i| (0..factor_names.len()).map(|j| corr[(i, j)]).collect())
            .collect(),
    };

    if input.portfolio.tickers() != model.tickers {
        return Err(ComputeError::InvalidInput(
            "portfolio holdings and fitted model tickers must match 1:1, in order".to_string(),
        ));
    }

    let mut per_holding = Vec::with_capacity(model.fits.len());
    let mut factor_attribution_log_inr: BTreeMap<String, f64> =
        factor_names.iter().map(|n| (n.clone(), 0.0)).collect();
    let mut portfolio_pnl_inr = 0.0;
    let mut portfolio_log_pnl_inr = 0.0;

    for (holding, fit) in input.portfolio.holdings.iter().zip(model.fits.iter()) {
        let value_inr = holding.weight * input.portfolio.total_value_inr;
        let log_return: f64 = fit
            .betas
            .iter()
            .zip(full_shock.iter())
            .map(|(b, s)| b * s)
            .sum();
        let simple_return = if input.linear_approximation {
            log_return
        } else {
            log_to_simple(log_return)
        };
        let pnl_inr = value_inr * simple_return;
        portfolio_pnl_inr += pnl_inr;
        portfolio_log_pnl_inr += value_inr * log_return;

        for (k, name) in factor_names.iter().enumerate() {
            *factor_attribution_log_inr.get_mut(name).unwrap() +=
                value_inr * fit.betas[k] * full_shock[k];
        }

        per_holding.push(HoldingShockResult {
            ticker: holding.ticker.clone(),
            value_inr,
            log_return,
            simple_return,
            pnl_inr,
        });
    }

    let attribution_log_sum: f64 = factor_attribution_log_inr.values().sum();
    let tolerance = 1e-6 * input.portfolio.total_value_inr.abs().max(1.0);
    let mut invariants = vec![InvariantCheck::approx_eq(
        "sum(per_holding.pnl_inr) == portfolio_pnl_inr",
        per_holding.iter().map(|h| h.pnl_inr).sum(),
        portfolio_pnl_inr,
        tolerance,
    )];
    invariants.push(InvariantCheck::approx_eq(
        "sum(factor_attribution_log_inr) == portfolio_log_pnl_inr",
        attribution_log_sum,
        portfolio_log_pnl_inr,
        tolerance,
    ));
    if input.linear_approximation {
        invariants.push(InvariantCheck::approx_eq(
            "linear_approximation: portfolio_log_pnl_inr == portfolio_pnl_inr",
            portfolio_log_pnl_inr,
            portfolio_pnl_inr,
            tolerance,
        ));
    }

    let output = FactorShockOutput {
        linear_approximation: input.linear_approximation,
        given_shocks,
        implied_shocks,
        per_holding,
        portfolio_pnl_inr,
        portfolio_log_pnl_inr,
        factor_attribution_log_inr,
        factor_correlation,
        conditional_coefficients: conditional_coefficients_out,
        gold_inr_implied_move,
    };

    let note = if input.linear_approximation {
        "linear_approximation=true: shocks are treated as literal decimal returns with no \
         log conversion; P&L = value * (beta . shock) is exact and \
         portfolio_log_pnl_inr == portfolio_pnl_inr."
    } else {
        "Shocks are converted simple -> log for propagation and beta application \
         (betas are fit on log returns); per-holding/portfolio P&L converts back with \
         exp(log_return) - 1. portfolio_log_pnl_inr and factor_attribution_log_inr are an \
         exact decomposition in log-return space, not of the (convex-transformed) INR P&L. \
         Stock alpha/intercept from the OLS fit is not applied in either mode."
    };

    let trace = EvidenceTrace {
        experiment: "FactorShock".to_string(),
        inputs: serde_json::to_value(input)?,
        data_window,
        data_quality: data_quality.clone(),
        model_params: ModelParams {
            frequency: model.frequency,
            window_periods: model.window,
            factor_names: factor_names.clone(),
            shrinkage_intensity: model.shrinkage_intensity,
            annualization_factor: model.frequency.annualization_factor(),
        },
        outputs: serde_json::json!({
            "result": output,
            "note": note,
        }),
        invariants,
        engine_version: crate::trace::engine_version(),
    };

    Ok((output, trace))
}

// ---------------------------------------------------------------------
// (b) RiskDecomposition
// ---------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct RiskDecompositionInput {
    pub portfolio: Portfolio,
    #[serde(default)]
    pub frequency: Frequency,
    /// Trailing window in periods at `frequency`. Defaults to
    /// `frequency.default_window()` (252 daily, 156 weekly) when omitted.
    #[serde(default)]
    pub window: Option<usize>,
}

impl RiskDecompositionInput {
    pub fn resolved_window(&self) -> usize {
        self.window.unwrap_or_else(|| self.frequency.default_window())
    }
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct StockContribution {
    pub ticker: String,
    pub contribution: f64,
    pub fraction_of_vol: f64,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct FactorContribution {
    pub factor: String,
    pub contribution: f64,
    pub fraction_of_vol: f64,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct RiskDecompositionOutput {
    pub portfolio_vol_annualized: f64,
    pub by_stock: Vec<StockContribution>,
    pub by_factor: Vec<FactorContribution>,
    pub specific_risk_contribution: f64,
    pub specific_risk_fraction_of_vol: f64,
}

pub fn run_risk_decomposition(
    data_quality: &DataQuality,
    data_window: DataWindow,
    model: &FactorModel,
    input: &RiskDecompositionInput,
) -> Result<(RiskDecompositionOutput, EvidenceTrace)> {
    if input.portfolio.tickers() != model.tickers {
        return Err(ComputeError::InvalidInput(
            "portfolio holdings and fitted model tickers must match 1:1, in order".to_string(),
        ));
    }
    let w = input.portfolio.weights();
    let n = w.len();

    let sigma_matrix = model.stock_covariance(); // annualized B F B^T + D
    let sigma_w = &sigma_matrix * &w;
    let variance = (w.transpose() * &sigma_w)[(0, 0)];
    let vol = variance.max(0.0).sqrt();

    let by_stock: Vec<StockContribution> = (0..n)
        .map(|i| {
            let contribution = if vol > 0.0 {
                w[i] * sigma_w[i] / vol
            } else {
                0.0
            };
            StockContribution {
                ticker: model.fits[i].ticker.clone(),
                contribution,
                fraction_of_vol: if vol > 0.0 { contribution / vol } else { 0.0 },
            }
        })
        .collect();

    let b = model.beta_matrix();
    let f_annual = model.factor_covariance();
    let x = b.transpose() * &w; // factor exposure, k-vector
    let fx = &f_annual * &x;

    let factor_names: Vec<String> = FACTOR_NAMES.iter().map(|s| s.to_string()).collect();
    let by_factor: Vec<FactorContribution> = (0..factor_names.len())
        .map(|k| {
            let contribution = if vol > 0.0 { x[k] * fx[k] / vol } else { 0.0 };
            FactorContribution {
                factor: factor_names[k].clone(),
                contribution,
                fraction_of_vol: if vol > 0.0 { contribution / vol } else { 0.0 },
            }
        })
        .collect();

    let specific_variance_contribution: f64 = (0..n)
        .map(|i| {
            let d_i_annual =
                annualize_scalar(model.fits[i].residual_variance_daily, model.frequency);
            w[i] * w[i] * d_i_annual
        })
        .sum();
    let specific_risk_contribution = if vol > 0.0 {
        specific_variance_contribution / vol
    } else {
        0.0
    };

    let stock_sum: f64 = by_stock.iter().map(|s| s.contribution).sum();
    let factor_plus_specific: f64 =
        by_factor.iter().map(|f| f.contribution).sum::<f64>() + specific_risk_contribution;

    let invariants = vec![
        InvariantCheck::approx_eq("sum(by_stock.contribution) == portfolio_vol", stock_sum, vol, 1e-9),
        InvariantCheck::approx_eq(
            "sum(by_factor.contribution) + specific_risk == portfolio_vol",
            factor_plus_specific,
            vol,
            1e-9,
        ),
    ];

    let output = RiskDecompositionOutput {
        portfolio_vol_annualized: vol,
        by_stock,
        by_factor,
        specific_risk_contribution,
        specific_risk_fraction_of_vol: if vol > 0.0 {
            specific_risk_contribution / vol
        } else {
            0.0
        },
    };

    let trace = EvidenceTrace {
        experiment: "RiskDecomposition".to_string(),
        inputs: serde_json::to_value(input)?,
        data_window,
        data_quality: data_quality.clone(),
        model_params: ModelParams {
            frequency: model.frequency,
            window_periods: model.window,
            factor_names,
            shrinkage_intensity: model.shrinkage_intensity,
            annualization_factor: model.frequency.annualization_factor(),
        },
        outputs: serde_json::to_value(&output)?,
        invariants,
        engine_version: crate::trace::engine_version(),
    };

    Ok((output, trace))
}

// ---------------------------------------------------------------------
// (c) CvarRebalance — see crate::cvar.
// ---------------------------------------------------------------------

pub use crate::cvar::CvarRebalanceInput;
