use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use chrono::NaiveDate;
use http_body_util::BodyExt;
use tower::ServiceExt;

use compute::data::DataQuality;
use compute::experiments::{Experiment, Holding, Portfolio, RiskDecompositionInput};
use compute::model::Frequency;
use compute::trace::{DataWindow, EvidenceTrace, ModelParams};

use crate::backend::{Backend, BackendError};
use crate::build_router;
use crate::routes::AppState;

fn sample_portfolio() -> Portfolio {
    Portfolio {
        holdings: vec![
            Holding {
                ticker: "RELIANCE.NS".to_string(),
                weight: 0.6,
            },
            Holding {
                ticker: "TCS.NS".to_string(),
                weight: 0.4,
            },
        ],
        total_value_inr: 1_000_000.0,
    }
}

fn sample_trace() -> EvidenceTrace {
    EvidenceTrace {
        experiment: "RiskDecomposition".to_string(),
        inputs: serde_json::json!({}),
        data_window: DataWindow {
            frequency: Frequency::Daily,
            window_periods: 252,
            start: NaiveDate::from_ymd_opt(2025, 9, 16).unwrap(),
            end: NaiveDate::from_ymd_opt(2026, 9, 24).unwrap(),
        },
        data_quality: DataQuality {
            date_range_start: NaiveDate::from_ymd_opt(2021, 9, 27).unwrap(),
            date_range_end: NaiveDate::from_ymd_opt(2026, 9, 24).unwrap(),
            trading_days: 1234,
            per_series: vec![],
        },
        model_params: ModelParams {
            frequency: Frequency::Daily,
            window_periods: 252,
            factor_names: vec!["MARKET".to_string()],
            shrinkage_intensity: 0.0374,
            annualization_factor: 252.0,
        },
        outputs: serde_json::json!({ "portfolio_vol_annualized": 0.1552 }),
        invariants: vec![],
        engine_version: "0.1.0".to_string(),
    }
}

/// A `Backend` that returns fixed responses (or a fixed error) — no
/// network access to Yahoo Finance or Gemini.
struct MockBackend {
    experiment_result: Option<EvidenceTrace>,
    ask_result: Option<agent::pipeline::PipelineResult>,
}

#[async_trait::async_trait]
impl Backend for MockBackend {
    async fn run_experiment(&self, _experiment: Experiment) -> Result<EvidenceTrace, BackendError> {
        self.experiment_result
            .clone()
            .ok_or_else(|| BackendError::Internal("no mock experiment result configured".to_string()))
    }

    async fn run_ask(
        &self,
        _portfolio: Portfolio,
        _message: String,
    ) -> Result<agent::pipeline::PipelineResult, BackendError> {
        self.ask_result
            .as_ref()
            .map(|r| agent::pipeline::PipelineResult {
                experiment: r.experiment.clone(),
                trace: r.trace.clone(),
                narration: agent::grounding::GroundedNarration {
                    narration: r.narration.narration.clone(),
                    grounding_warnings: r.narration.grounding_warnings.clone(),
                },
            })
            .ok_or_else(|| BackendError::Internal("no mock ask result configured".to_string()))
    }
}

fn app_with_backend(backend: MockBackend) -> axum::Router {
    build_router(AppState {
        backend: Arc::new(backend),
    })
}

async fn body_json(response: axum::response::Response) -> serde_json::Value {
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    serde_json::from_slice(&bytes).unwrap()
}

#[tokio::test]
async fn health_returns_200_and_expected_json() {
    let app = app_with_backend(MockBackend {
        experiment_result: None,
        ask_result: None,
    });

    let response = app
        .oneshot(Request::builder().uri("/health").body(Body::empty()).unwrap())
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;
    assert_eq!(body["status"], "ok");
    assert_eq!(body["version"], env!("CARGO_PKG_VERSION"));
}

#[tokio::test]
async fn experiment_with_valid_request_returns_a_trace() {
    let app = app_with_backend(MockBackend {
        experiment_result: Some(sample_trace()),
        ask_result: None,
    });

    let req_body = serde_json::json!({
        "portfolio": sample_portfolio(),
        "experiment": { "type": "RiskDecomposition" },
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/experiment")
                .header("content-type", "application/json")
                .body(Body::from(req_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;
    assert_eq!(body["experiment"], "RiskDecomposition");
    assert_eq!(body["outputs"]["portfolio_vol_annualized"], 0.1552);
}

#[tokio::test]
async fn experiment_with_weights_not_summing_to_one_returns_400() {
    let app = app_with_backend(MockBackend {
        experiment_result: Some(sample_trace()),
        ask_result: None,
    });

    let bad_portfolio = serde_json::json!({
        "holdings": [
            {"ticker": "RELIANCE.NS", "weight": 0.6},
            {"ticker": "TCS.NS", "weight": 0.6},
        ],
        "total_value_inr": 1000000.0,
    });
    let req_body = serde_json::json!({
        "portfolio": bad_portfolio,
        "experiment": { "type": "RiskDecomposition" },
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/experiment")
                .header("content-type", "application/json")
                .body(Body::from(req_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = body_json(response).await;
    assert_eq!(body["code"], "invalid_portfolio");
    assert!(body["error"].as_str().unwrap().contains("sum to 1.0"));
}

#[tokio::test]
async fn ask_with_mocked_pipeline_returns_grounding_warnings() {
    let pipeline_result = agent::pipeline::PipelineResult {
        experiment: Experiment::RiskDecomposition(RiskDecompositionInput {
            portfolio: sample_portfolio(),
            frequency: Frequency::Daily,
            window: None,
        }),
        trace: sample_trace(),
        narration: agent::grounding::GroundedNarration {
            narration: "Vol is 99% (unverified).".to_string(),
            grounding_warnings: vec![
                "unverified number '99%' at byte position 7 in the narration".to_string(),
            ],
        },
    };
    let app = app_with_backend(MockBackend {
        experiment_result: None,
        ask_result: Some(pipeline_result),
    });

    let req_body = serde_json::json!({
        "portfolio": sample_portfolio(),
        "message": "what's my portfolio risk?",
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/ask")
                .header("content-type", "application/json")
                .body(Body::from(req_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;
    assert_eq!(body["narration"], "Vol is 99% (unverified).");
    assert_eq!(body["grounding_warnings"].as_array().unwrap().len(), 1);
    assert_eq!(body["experiment"]["type"], "RiskDecomposition");
}

#[tokio::test]
async fn static_route_returns_200_and_html_content_type() {
    let app = app_with_backend(MockBackend {
        experiment_result: None,
        ask_result: None,
    });

    let response = app
        .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let content_type = response
        .headers()
        .get("content-type")
        .unwrap()
        .to_str()
        .unwrap();
    assert!(content_type.starts_with("text/html"));
}
