//! Evidence Trace: the single structured record every experiment returns.
//! Rule (per spec): every number the LLM layer could later cite must exist
//! as a named field somewhere in this struct, not just be printed.

use chrono::NaiveDate;
use schemars::JsonSchema;
use serde::Serialize;

use crate::data::DataQuality;

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct DataWindow {
    pub window_days: usize,
    pub start: NaiveDate,
    pub end: NaiveDate,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct ModelParams {
    pub window_days: usize,
    pub factor_names: Vec<String>,
    pub shrinkage_intensity: f64,
    pub annualization_factor: f64,
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
