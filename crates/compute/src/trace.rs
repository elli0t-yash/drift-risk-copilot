//! Evidence Trace: the single structured record every experiment returns.
//! Rule (per spec): every number the LLM layer could later cite must exist
//! as a named field somewhere in this struct, not just be printed.

use chrono::NaiveDate;
use schemars::JsonSchema;
use serde::Serialize;

use crate::data::DataQuality;
use crate::model::Frequency;
use crate::regime::RegimeState;

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct DataWindow {
    pub frequency: Frequency,
    pub window_periods: usize,
    pub start: NaiveDate,
    pub end: NaiveDate,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct ModelParams {
    pub frequency: Frequency,
    pub window_periods: usize,
    pub factor_names: Vec<String>,
    pub shrinkage_intensity: f64,
    pub annualization_factor: f64,
    /// `Some` only when the experiment requested `regime_covariance: true`.
    pub regime_state: Option<RegimeState>,
    /// Non-empty only when `regime_state` is `Some` and at least one
    /// regime had fewer than `model::MIN_REGIME_OBSERVATIONS` observations
    /// in the fitted window (see `model::fit_factor_model_with_config`).
    pub regime_fallback_warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct InvariantCheck {
    pub name: String,
    pub passed: bool,
    pub tolerance: f64,
    pub detail: String,
}

impl InvariantCheck {
    pub fn approx_eq(name: &str, lhs: f64, rhs: f64, tolerance: f64) -> Self {
        let diff = (lhs - rhs).abs();
        InvariantCheck {
            name: name.to_string(),
            passed: diff <= tolerance,
            tolerance,
            detail: format!("lhs={lhs:.12} rhs={rhs:.12} abs_diff={diff:.3e}"),
        }
    }
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct EvidenceTrace {
    pub experiment: String,
    pub inputs: serde_json::Value,
    pub data_window: DataWindow,
    pub data_quality: DataQuality,
    pub model_params: ModelParams,
    pub outputs: serde_json::Value,
    pub invariants: Vec<InvariantCheck>,
    pub engine_version: String,
}

pub fn engine_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}
