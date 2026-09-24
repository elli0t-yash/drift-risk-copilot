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
    /// Shocks in percent (e.g. -12.0 for -12%), keyed by factor name (one
    /// of `FACTOR_NAMES`: MARKET, USDINR, BRENT, GOLD, RATES_PROXY).
    pub shocks_pct: BTreeMap<String, f64>,
    /// When true (default), unshocked factors receive their conditional
    /// expectation given the shocked factors, via the factor covariance.
    /// When false, unshocked factors are held at zero.
    #[serde(default = "default_true")]
    pub propagate: bool,
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

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct HoldingShockResult {
    pub ticker: String,
    pub value_inr: f64,
    /// beta . s (decimal return implied by the full shock vector).
    pub implied_return: f64,
    pub pnl_inr: f64,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct FactorShockOutput {
    /// The shocks as given by the caller (decimal, e.g. -0.12).
    pub given_shocks: BTreeMap<String, f64>,
    /// Shocks implied by conditional expectation for factors the caller
    /// did not specify (empty if `propagate` is false, or if all factors
    /// were given).
    pub implied_shocks: BTreeMap<String, f64>,
    pub per_holding: Vec<HoldingShockResult>,
    pub portfolio_pnl_inr: f64,
    /// Attribution of portfolio P&L (INR) by factor: sum_i value_i * beta_ik * s_k.
    pub factor_attribution_inr: BTreeMap<String, f64>,
}

/// Computes conditional expectations for the "unknown" factor block given
/// the "known" block, s_u = F_uk * F_kk^-1 * s_k.
fn conditional_expectation(
    f: &DMatrix<f64>,
    known_idx: &[usize],
    known_vals: &[f64],
    unknown_idx: &[usize],
) -> Result<Vec<f64>> {
    if known_idx.is_empty() || unknown_idx.is_empty() {
        return Ok(vec![0.0; unknown_idx.len()]);
    }
    let f_kk = DMatrix::from_fn(known_idx.len(), known_idx.len(), |i, j| {
        f[(known_idx[i], known_idx[j])]
    });
    let f_uk = DMatrix::from_fn(unknown_idx.len(), known_idx.len(), |i, j| {
        f[(unknown_idx[i], known_idx[j])]
    });
    let s_k = DVector::from_row_slice(known_vals);

    let f_kk_inv = f_kk.clone().try_inverse().ok_or_else(|| {
        ComputeError::Model(
            "factor covariance sub-block F_kk is singular; cannot condition".to_string(),
        )
    })?;
    let s_u = f_uk * f_kk_inv * s_k;
    Ok(s_u.iter().copied().collect())
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

    let given_shocks: BTreeMap<String, f64> = input
        .shocks_pct
        .iter()
        .map(|(k, v)| (k.clone(), v / 100.0))
        .collect();

    let known_idx: Vec<usize> = factor_names
        .iter()
        .enumerate()
        .filter(|(_, n)| given_shocks.contains_key(*n))
        .map(|(i, _)| i)
        .collect();
    let unknown_idx: Vec<usize> = (0..factor_names.len())
        .filter(|i| !known_idx.contains(i))
        .collect();
    let known_vals: Vec<f64> = known_idx
        .iter()
        .map(|i| given_shocks[&factor_names[*i]])
        .collect();

    let mut implied_shocks = BTreeMap::new();
    let mut full_shock = vec![0.0; factor_names.len()];
    for (i, v) in known_idx.iter().zip(known_vals.iter()) {
        full_shock[*i] = *v;
    }

    if input.propagate && !unknown_idx.is_empty() && !known_idx.is_empty() {
        let f = model.factor_covariance();
        let implied = conditional_expectation(&f, &known_idx, &known_vals, &unknown_idx)?;
        for (idx, val) in unknown_idx.iter().zip(implied.iter()) {
            full_shock[*idx] = *val;
            implied_shocks.insert(factor_names[*idx].clone(), *val);
        }
    }

    if input.portfolio.tickers() != model.tickers {
        return Err(ComputeError::InvalidInput(
            "portfolio holdings and fitted model tickers must match 1:1, in order".to_string(),
        ));
    }

    let mut per_holding = Vec::with_capacity(model.fits.len());
    let mut factor_attribution_inr: BTreeMap<String, f64> =
        factor_names.iter().map(|n| (n.clone(), 0.0)).collect();
    let mut portfolio_pnl_inr = 0.0;

    for (holding, fit) in input.portfolio.holdings.iter().zip(model.fits.iter()) {
        let value_inr = holding.weight * input.portfolio.total_value_inr;
        let implied_return: f64 = fit
            .betas
            .iter()
            .zip(full_shock.iter())
            .map(|(b, s)| b * s)
            .sum();
        let pnl_inr = value_inr * implied_return;
        portfolio_pnl_inr += pnl_inr;

        for (k, name) in factor_names.iter().enumerate() {
            *factor_attribution_inr.get_mut(name).unwrap() +=
                value_inr * fit.betas[k] * full_shock[k];
        }

        per_holding.push(HoldingShockResult {
            ticker: holding.ticker.clone(),
            value_inr,
            implied_return,
            pnl_inr,
        });
    }

    let attribution_sum: f64 = factor_attribution_inr.values().sum();
    let mut invariants = vec![InvariantCheck::approx_eq(
        "sum(per_holding.pnl_inr) == portfolio_pnl_inr",
        per_holding.iter().map(|h| h.pnl_inr).sum(),
        portfolio_pnl_inr,
        1e-6 * input.portfolio.total_value_inr.abs().max(1.0),
    )];
    invariants.push(InvariantCheck::approx_eq(
        "sum(factor_attribution_inr) == portfolio_pnl_inr",
        attribution_sum,
        portfolio_pnl_inr,
        1e-6 * input.portfolio.total_value_inr.abs().max(1.0),
    ));

    let output = FactorShockOutput {
        given_shocks,
        implied_shocks,
        per_holding,
        portfolio_pnl_inr,
        factor_attribution_inr,
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
            "note": "Linear, no-intercept approximation: P&L = value * (beta . shock). \
                     Stock alpha/intercept from the OLS fit is not applied.",
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
// (c) CvarRebalance — design note only, not implemented this checkpoint.
// ---------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct CvarRebalanceInput {
    pub portfolio: Portfolio,
    pub confidence_level: f64,
    pub per_name_cap: f64,
    pub turnover_limit: f64,
    pub commission_bps: f64,
}

/// Not implemented. See the design note in the checkpoint report / crate
/// README for the proposed Rockafellar-Uryasev LP formulation, scenario
/// source, solver choice, and infeasibility reporting.
pub fn run_cvar_rebalance(_input: &CvarRebalanceInput) -> Result<()> {
    Err(ComputeError::Model(
        "CvarRebalance is a design note only in this checkpoint; not implemented".to_string(),
    ))
}
