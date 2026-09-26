//! Evidence Trace: the single structured record every experiment returns.
//! Rule (per spec): every number the LLM layer could later cite must exist
//! as a named field somewhere in this struct, not just be printed.

use chrono::NaiveDate;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::data::DataQuality;
use crate::model::Frequency;
use crate::policy::PolicyResult;
use crate::regime::RegimeState;

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct DataWindow {
    pub frequency: Frequency,
    pub window_periods: usize,
    pub start: NaiveDate,
    pub end: NaiveDate,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ModelParams {
    pub frequency: Frequency,
    pub window_periods: usize,
    pub factor_names: Vec<String>,
    pub shrinkage_intensity: f64,
    pub annualization_factor: f64,
    /// Always populated (regime-conditioning is unconditional -- see
    /// `model::fit_factor_model`); `None` only for experiment types that
    /// fit no factor model at all and whose own standalone regime fit also
    /// failed (see `performance::run_portfolio_performance`).
    pub regime_state: Option<RegimeState>,
    /// Non-empty only when `regime_state` is `Some` and at least one
    /// regime had fewer than `model::MIN_REGIME_OBSERVATIONS` observations
    /// in the fitted window (see `model::fit_factor_model`).
    pub regime_fallback_warnings: Vec<String>,
    /// `Some` only for `CvarRebalance`: `"user-specified"` if the caller
    /// gave `per_name_cap`, `"server-default-0.20"` if it was defaulted
    /// (see `cvar::CvarRebalanceInput::per_name_cap`). `None` for
    /// `FactorShock`/`RiskDecomposition`, which have no such field.
    pub cap_source: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
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

/// Key params from a `RiskDrift` baseline snapshot's own trace, carried
/// alongside the current fit's `model_params` so a `RiskDrift` trace is
/// self-contained (a reader doesn't have to separately fetch the baseline
/// snapshot to know what it was fit against).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct BaselineModelParams {
    pub window_periods: usize,
    pub frequency: Frequency,
    pub shrinkage_intensity: f64,
    pub regime_label: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct EvidenceTrace {
    pub experiment: String,
    pub inputs: serde_json::Value,
    pub data_window: DataWindow,
    pub data_quality: DataQuality,
    pub model_params: ModelParams,
    pub outputs: serde_json::Value,
    pub invariants: Vec<InvariantCheck>,
    pub engine_version: String,
    /// `Some` only for `RiskDrift` (see `BaselineModelParams`); `None` for
    /// every other experiment type.
    #[serde(default)]
    pub baseline_model_params: Option<BaselineModelParams>,
    /// A passive policy check run alongside *any* experiment when the
    /// request carries a top-level `policy` field (see
    /// `dispatch::run_experiment`'s doc) -- `None` unless one was
    /// attached. `PolicyCheck`'s own result lives in `outputs.result`
    /// instead (it's the experiment, not a side effect of one), so this is
    /// always `None` for a `PolicyCheck` trace.
    #[serde(default)]
    pub policy_result: Option<PolicyResult>,
}

pub fn engine_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}
