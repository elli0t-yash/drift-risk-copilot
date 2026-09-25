use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::{header, StatusCode};
use axum::response::{Html, IntoResponse, Response};
use axum::Json;
use compute::experiments::{Experiment, Portfolio};
use compute::trace::EvidenceTrace;
use serde::{Deserialize, Serialize};
use store::{RiskSnapshot, SnapshotStore};

use crate::backend::Backend;
use crate::error::{ApiError, AppJson};
use crate::validate::validate_portfolio;

const INDEX_HTML: &str = include_str!("../static/index.html");

#[derive(Clone)]
pub struct AppState {
    pub backend: Arc<dyn Backend>,
    pub store: Arc<SnapshotStore>,
}

/// Builds the `RiskSnapshot` `SnapshotStore::insert` persists for a
/// completed experiment: `trace_json` is the full trace (so `GET
/// /report/{id}` can render a PDF from it later), and the other fields are
/// pulled out of it for fast querying without re-parsing `trace_json`.
/// `regime_label`/`smoothed_probs` come from `model_params.regime_state`,
/// which every experiment type now always populates (see `compute`'s
/// always-on regime).
fn snapshot_from_trace(trace: &EvidenceTrace, portfolio: &Portfolio) -> Result<RiskSnapshot, ApiError> {
    let holdings: Vec<(String, f64)> =
        portfolio.holdings.iter().map(|h| (h.ticker.clone(), h.weight)).collect();
    let portfolio_hash = compute::portfolio::portfolio_hash(&holdings);

    let result = trace.outputs.get("result");
    let number_at = |path: &[&str]| -> Option<f64> {
        let mut v = result?;
        for key in path {
            v = v.get(key)?;
        }
        v.as_f64()
    };
    let portfolio_vol_annualized = number_at(&["portfolio_vol_annualized"]);
    let cvar_historical = number_at(&["stats_after", "historical_cvar"])
        .or_else(|| number_at(&["stats_before", "historical_cvar"]));

    let trace_json = serde_json::to_string(trace)
        .map_err(|e| ApiError::bad_request("internal_error", format!("failed to serialize trace: {e}")))?;

    Ok(RiskSnapshot {
        id: String::new(), // SnapshotStore::insert assigns the real id
        created_at: String::new(),
        portfolio_hash,
        experiment_type: trace.experiment.clone(),
        engine_version: trace.engine_version.clone(),
        regime_label: trace.model_params.regime_state.as_ref().map(|r| r.current_label.to_string()),
        smoothed_probs: trace.model_params.regime_state.as_ref().map(|r| r.smoothed_probs),
        portfolio_vol_annualized,
        cvar_historical,
        trace_json,
        // Populated by `post_ask` after this snapshot is built (this path
        // never calls Gemini, so `post_experiment` leaves them `None`).
        narration: None,
        suggestion: None,
        grounding_warnings: None,
    })
}

#[derive(Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
    pub version: &'static str,
}

pub async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        version: env!("CARGO_PKG_VERSION"),
    })
}

#[derive(Deserialize)]
pub struct ExperimentRequest {
    pub portfolio: Portfolio,
    /// The tagged `Experiment` variant's own fields *except* `portfolio`
    /// (e.g. `{"type": "FactorShock", "shocks_pct": {...}}`); `portfolio`
    /// above is spliced in server-side before deserializing into
    /// `Experiment`, so callers never have to repeat it.
    pub experiment: serde_json::Value,
}

pub async fn post_experiment(
    State(state): State<AppState>,
    AppJson(req): AppJson<ExperimentRequest>,
) -> Result<Json<EvidenceTrace>, ApiError> {
    validate_portfolio(&req.portfolio)?;

    let mut experiment_json = req.experiment;
    if let serde_json::Value::Object(ref mut map) = experiment_json {
        map.insert(
            "portfolio".to_string(),
            serde_json::to_value(&req.portfolio).expect("Portfolio always serializes"),
        );
    } else {
        return Err(ApiError::bad_request(
            "invalid_experiment",
            "experiment must be a JSON object with a \"type\" field",
        ));
    }
    let experiment: Experiment = serde_json::from_value(experiment_json).map_err(|e| {
        ApiError::bad_request("invalid_experiment", format!("invalid experiment: {e}"))
    })?;

    let trace = state.backend.run_experiment(experiment).await?;

    let snapshot = snapshot_from_trace(&trace, &req.portfolio)?;
    state.store.insert(&snapshot).map_err(|e| {
        ApiError::bad_request("internal_error", format!("failed to persist risk snapshot: {e}"))
    })?;

    Ok(Json(trace))
}

#[derive(Deserialize)]
pub struct AskRequest {
    pub portfolio: Portfolio,
    pub message: String,
    /// Prior turns of this conversation, oldest first; empty by default.
    /// See `agent::conversation::ConversationTurn`.
    #[serde(default)]
    pub conversation_history: Vec<agent::ConversationTurn>,
}

#[derive(Serialize)]
pub struct AskResponse {
    pub experiment: Experiment,
    pub trace: EvidenceTrace,
    pub narration: String,
    pub grounding_warnings: Vec<String>,
    /// This turn's narration as an `assistant` turn, ready for the caller
    /// to append to `conversation_history` for the next request.
    pub assistant_turn: agent::ConversationTurn,
    /// One follow-up question a risk manager would naturally ask next.
    pub suggestion: String,
    /// Stored under this id in the persistent `SnapshotStore`;
    /// `GET /report/{result_id}` renders it as a PDF.
    pub result_id: String,
}

pub async fn post_ask(
    State(state): State<AppState>,
    AppJson(req): AppJson<AskRequest>,
) -> Result<Json<AskResponse>, ApiError> {
    validate_portfolio(&req.portfolio)?;

    let portfolio = req.portfolio.clone();
    let result = state
        .backend
        .run_ask(req.portfolio, req.message, req.conversation_history)
        .await?;

    let mut snapshot = snapshot_from_trace(&result.trace, &portfolio)?;
    snapshot.narration = Some(result.narration.narration.clone());
    snapshot.suggestion = Some(result.suggestion.clone());
    snapshot.grounding_warnings = if result.narration.grounding_warnings.is_empty() {
        None
    } else {
        Some(serde_json::to_string(&result.narration.grounding_warnings).map_err(|e| {
            ApiError::bad_request("internal_error", format!("failed to serialize grounding warnings: {e}"))
        })?)
    };
    let result_id = state.store.insert(&snapshot).map_err(|e| {
        ApiError::bad_request("internal_error", format!("failed to persist risk snapshot: {e}"))
    })?;

    Ok(Json(AskResponse {
        experiment: result.experiment,
        trace: result.trace,
        narration: result.narration.narration,
        grounding_warnings: result.narration.grounding_warnings,
        assistant_turn: result.assistant_turn,
        suggestion: result.suggestion,
        result_id,
    }))
}

/// Serves the embedded single-page UI for every unmatched path, so the
/// binary needs no separate static-file directory at runtime.
pub async fn static_handler() -> Html<&'static str> {
    Html(INDEX_HTML)
}

/// `GET /scenarios`: the fixed set of historical scenario presets. No
/// authentication, no portfolio needed.
pub async fn get_scenarios() -> Json<serde_json::Value> {
    Json(serde_json::to_value(compute::scenarios::all_scenarios()).expect("scenarios always serialize"))
}

/// `GET /report/{result_id}`: a one-page PDF report for a previously
/// stored `/experiment` or `/ask` result. 404 if `result_id` is unknown
/// (never stored, or the id is simply wrong).
pub async fn get_report(State(state): State<AppState>, Path(result_id): Path<String>) -> Response {
    let snapshot = match state.store.get(&result_id) {
        Ok(Some(snapshot)) => snapshot,
        Ok(None) => {
            return ApiError::not_found("result_not_found", format!("no stored result for id {result_id}"))
                .into_response();
        }
        Err(e) => {
            return ApiError::bad_request("internal_error", format!("failed to read snapshot store: {e}"))
                .into_response();
        }
    };
    let trace: EvidenceTrace = match serde_json::from_str(&snapshot.trace_json) {
        Ok(trace) => trace,
        Err(e) => {
            return ApiError::bad_request("internal_error", format!("stored trace_json is invalid: {e}"))
                .into_response();
        }
    };
    let grounding_warnings: Vec<String> = match &snapshot.grounding_warnings {
        Some(json) => match serde_json::from_str(json) {
            Ok(warnings) => warnings,
            Err(e) => {
                return ApiError::bad_request("internal_error", format!("stored grounding_warnings is invalid: {e}"))
                    .into_response();
            }
        },
        None => Vec::new(),
    };
    let bytes = crate::pdf::render_report(&trace, snapshot.narration.as_deref(), &grounding_warnings);
    (StatusCode::OK, [(header::CONTENT_TYPE, "application/pdf")], bytes).into_response()
}
