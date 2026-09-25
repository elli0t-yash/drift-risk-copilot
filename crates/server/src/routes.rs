use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::{header, StatusCode};
use axum::response::{Html, IntoResponse, Response};
use axum::Json;
use compute::experiments::{Experiment, Portfolio};
use compute::trace::EvidenceTrace;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::backend::Backend;
use crate::error::{ApiError, AppJson};
use crate::store::ResultStore;
use crate::validate::validate_portfolio;

const INDEX_HTML: &str = include_str!("../static/index.html");

#[derive(Clone)]
pub struct AppState {
    pub backend: Arc<dyn Backend>,
    pub store: Arc<ResultStore>,
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
    /// Stored under this id in the server's in-memory `ResultStore`;
    /// `GET /report/{result_id}` renders it as a PDF.
    pub result_id: Uuid,
}

pub async fn post_ask(
    State(state): State<AppState>,
    AppJson(req): AppJson<AskRequest>,
) -> Result<Json<AskResponse>, ApiError> {
    validate_portfolio(&req.portfolio)?;

    let result = state
        .backend
        .run_ask(req.portfolio, req.message, req.conversation_history)
        .await?;
    let result_id = state.store.insert(result.clone());
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
/// stored `/ask` result. 404 if `result_id` is unknown (never stored, or
/// evicted from the fixed-capacity `ResultStore`).
pub async fn get_report(State(state): State<AppState>, Path(result_id): Path<Uuid>) -> Response {
    match state.store.get(&result_id) {
        Some(result) => {
            let bytes = crate::pdf::render_report(&result);
            (
                StatusCode::OK,
                [(header::CONTENT_TYPE, "application/pdf")],
                bytes,
            )
                .into_response()
        }
        None => ApiError::not_found("result_not_found", format!("no stored result for id {result_id}"))
            .into_response(),
    }
}
