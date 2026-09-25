use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use chrono::NaiveDate;
use http_body_util::BodyExt;
use tower::ServiceExt;

use compute::data::DataQuality;
use compute::experiments::{Experiment, Holding, Portfolio, RiskDecompositionInput};
use compute::model::Frequency;
use compute::regime::RegimeState;
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

/// A fixed `RegimeState`, standing in for a real HMM fit -- regime is
/// always-on now (see `compute`'s always-on regime), so every mock trace
/// used by these tests carries one, matching what a real experiment
/// response always has.
fn sample_regime_state() -> RegimeState {
    RegimeState {
        current_regime: 0,
        current_label: "Bull",
        smoothed_probs: [0.7, 0.2, 0.1],
        viterbi_sequence: vec![0; 252],
        obs_count_per_regime: [200, 40, 12],
        log_likelihood: -123.45,
        n_iter: 12,
        smoothing_note: "full-history smoothed, not suitable for live trading signals",
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
            regime_state: Some(sample_regime_state()),
            regime_fallback_warnings: vec![],
            cap_source: None,
        },
        outputs: serde_json::json!({ "result": { "portfolio_vol_annualized": 0.1552 } }),
        invariants: vec![],
        engine_version: "0.1.0".to_string(),
    }
}

/// A `Backend` that returns fixed responses (or a fixed error) — no
/// network access to Yahoo Finance or Gemini.
struct MockBackend {
    experiment_result: Option<EvidenceTrace>,
    ask_result: Option<agent::pipeline::PipelineResult>,
    received_conversation_history: Mutex<Option<Vec<agent::ConversationTurn>>>,
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
        conversation_history: Vec<agent::ConversationTurn>,
    ) -> Result<agent::pipeline::PipelineResult, BackendError> {
        *self.received_conversation_history.lock().unwrap() = Some(conversation_history);
        self.ask_result
            .as_ref()
            .map(|r| agent::pipeline::PipelineResult {
                experiment: r.experiment.clone(),
                trace: r.trace.clone(),
                narration: agent::grounding::GroundedNarration {
                    narration: r.narration.narration.clone(),
                    grounding_warnings: r.narration.grounding_warnings.clone(),
                },
                assistant_turn: r.assistant_turn.clone(),
                suggestion: r.suggestion.clone(),
            })
            .ok_or_else(|| BackendError::Internal("no mock ask result configured".to_string()))
    }
}

fn app_with_backend(backend: MockBackend) -> axum::Router {
    app_with_backend_and_store(backend).0
}

/// Like `app_with_backend`, but also hands back the `SnapshotStore` handle
/// so a test can inspect what got persisted directly (rather than only
/// through the HTTP responses), e.g. after `POST /experiment`, which never
/// echoes a `result_id` back to the caller.
fn app_with_backend_and_store(backend: MockBackend) -> (axum::Router, Arc<store::SnapshotStore>) {
    let store = Arc::new(store::SnapshotStore::open(":memory:").unwrap());
    let app = build_router(AppState {
        backend: Arc::new(backend),
        store: store.clone(),
    });
    (app, store)
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
        received_conversation_history: Mutex::new(None),
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
        received_conversation_history: Mutex::new(None),
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
    assert_eq!(body["outputs"]["result"]["portfolio_vol_annualized"], 0.1552);
}

#[tokio::test]
async fn experiment_with_weights_not_summing_to_one_returns_400() {
    let app = app_with_backend(MockBackend {
        experiment_result: Some(sample_trace()),
        ask_result: None,
        received_conversation_history: Mutex::new(None),
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
        assistant_turn: agent::ConversationTurn::assistant("Vol is 99% (unverified)."),
        suggestion: "What if I reduce my turnover to 20%?".to_string(),
    };
    let app = app_with_backend(MockBackend {
        experiment_result: None,
        ask_result: Some(pipeline_result),
        received_conversation_history: Mutex::new(None),
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
    assert_eq!(body["assistant_turn"]["role"], "assistant");
    assert_eq!(body["assistant_turn"]["content"], "Vol is 99% (unverified).");
    assert_eq!(body["suggestion"], "What if I reduce my turnover to 20%?");
    assert!(body["result_id"].is_string(), "expected a result_id field, got {body:?}");
}

#[tokio::test]
async fn ask_with_non_empty_conversation_history_forwards_it_to_the_backend() {
    let pipeline_result = agent::pipeline::PipelineResult {
        experiment: Experiment::RiskDecomposition(RiskDecompositionInput {
            portfolio: sample_portfolio(),
            frequency: Frequency::Daily,
            window: None,
        }),
        trace: sample_trace(),
        narration: agent::grounding::GroundedNarration {
            narration: "Vol is 15.5% annualised.".to_string(),
            grounding_warnings: vec![],
        },
        assistant_turn: agent::ConversationTurn::assistant("Vol is 15.5% annualised."),
        suggestion: "Now reduce my tail risk with 20% turnover?".to_string(),
    };
    let backend = Arc::new(MockBackend {
        experiment_result: None,
        ask_result: Some(pipeline_result),
        received_conversation_history: Mutex::new(None),
    });
    let app = build_router(AppState {
        backend: backend.clone(),
        store: Arc::new(store::SnapshotStore::open(":memory:").unwrap()),
    });

    let req_body = serde_json::json!({
        "portfolio": sample_portfolio(),
        "message": "now reduce my tail risk, I can tolerate 20% turnover",
        "conversation_history": [
            {"role": "user", "content": "where is my risk concentrated?"},
            {"role": "assistant", "content": "Vol is 15.5% annualised."},
        ],
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
    let received = backend.received_conversation_history.lock().unwrap();
    let history = received.as_ref().expect("run_ask should have been called");
    assert_eq!(history.len(), 2);
    assert_eq!(history[0].role, "user");
    assert_eq!(history[1].role, "assistant");
    assert_eq!(history[1].content, "Vol is 15.5% annualised.");
}

#[tokio::test]
async fn report_route_returns_pdf_for_a_result_stored_by_a_prior_ask() {
    let pipeline_result = agent::pipeline::PipelineResult {
        experiment: Experiment::RiskDecomposition(RiskDecompositionInput {
            portfolio: sample_portfolio(),
            frequency: Frequency::Daily,
            window: None,
        }),
        trace: sample_trace(),
        narration: agent::grounding::GroundedNarration {
            narration: "Vol is 15.5% annualised.".to_string(),
            grounding_warnings: vec![],
        },
        assistant_turn: agent::ConversationTurn::assistant("Vol is 15.5% annualised."),
        suggestion: "Now reduce my tail risk?".to_string(),
    };
    let app = app_with_backend(MockBackend {
        experiment_result: None,
        ask_result: Some(pipeline_result),
        received_conversation_history: Mutex::new(None),
    });

    let ask_body = serde_json::json!({
        "portfolio": sample_portfolio(),
        "message": "where is my risk concentrated?",
    });
    let ask_response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/ask")
                .header("content-type", "application/json")
                .body(Body::from(ask_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(ask_response.status(), StatusCode::OK);
    let ask_json = body_json(ask_response).await;
    let result_id = ask_json["result_id"].as_str().expect("result_id should be a string").to_string();

    let report_response = app
        .oneshot(
            Request::builder()
                .uri(format!("/report/{result_id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(report_response.status(), StatusCode::OK);
    let content_type = report_response
        .headers()
        .get("content-type")
        .unwrap()
        .to_str()
        .unwrap();
    assert_eq!(content_type, "application/pdf");
    let bytes = report_response.into_body().collect().await.unwrap().to_bytes();
    assert!(bytes.starts_with(b"%PDF"), "expected a PDF file signature");
}

#[tokio::test]
async fn report_route_returns_404_for_an_unknown_id() {
    let app = app_with_backend(MockBackend {
        experiment_result: None,
        ask_result: None,
        received_conversation_history: Mutex::new(None),
    });

    let unknown_id = uuid::Uuid::new_v4();
    let response = app
        .oneshot(
            Request::builder()
                .uri(format!("/report/{unknown_id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn scenarios_route_returns_three_scenarios_with_expected_fields() {
    let app = app_with_backend(MockBackend {
        experiment_result: None,
        ask_result: None,
        received_conversation_history: Mutex::new(None),
    });

    let response = app
        .oneshot(Request::builder().uri("/scenarios").body(Body::empty()).unwrap())
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;
    let scenarios = body.as_array().expect("expected a JSON array");
    assert_eq!(scenarios.len(), 3);
    for scenario in scenarios {
        assert!(scenario["id"].is_string());
        assert!(scenario["name"].is_string());
        assert!(scenario["shocks_pct"].is_object());
    }
}

#[tokio::test]
async fn static_route_returns_200_and_html_content_type() {
    let app = app_with_backend(MockBackend {
        experiment_result: None,
        ask_result: None,
        received_conversation_history: Mutex::new(None),
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

#[tokio::test]
async fn experiment_still_works_end_to_end_with_snapshot_store() {
    let (app, store) = app_with_backend_and_store(MockBackend {
        experiment_result: Some(sample_trace()),
        ask_result: None,
        received_conversation_history: Mutex::new(None),
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

    let recent = store.list_recent(10).unwrap();
    assert_eq!(recent.len(), 1, "POST /experiment should have inserted exactly one snapshot");
    let snapshot = &recent[0];
    assert_eq!(snapshot.experiment_type, "RiskDecomposition");
    assert_eq!(snapshot.portfolio_vol_annualized, Some(0.1552));
    assert_eq!(snapshot.regime_label.as_deref(), Some("Bull"));
    assert!(snapshot.smoothed_probs.is_some());

    let fetched = store.get(&snapshot.id).unwrap();
    assert!(fetched.is_some(), "the just-inserted snapshot should be retrievable by id");
}

#[tokio::test]
async fn report_route_retrieves_an_experiment_originated_snapshot() {
    let (app, store) = app_with_backend_and_store(MockBackend {
        experiment_result: Some(sample_trace()),
        ask_result: None,
        received_conversation_history: Mutex::new(None),
    });

    let req_body = serde_json::json!({
        "portfolio": sample_portfolio(),
        "experiment": { "type": "RiskDecomposition" },
    });
    app.clone()
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

    let id = store.list_recent(1).unwrap()[0].id.clone();
    let response = app
        .oneshot(Request::builder().uri(format!("/report/{id}")).body(Body::empty()).unwrap())
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    assert!(bytes.starts_with(b"%PDF"), "expected a PDF file signature");
}

#[tokio::test]
async fn regime_state_is_non_null_in_every_experiment_response() {
    let app = app_with_backend(MockBackend {
        experiment_result: Some(sample_trace()),
        ask_result: None,
        received_conversation_history: Mutex::new(None),
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
    assert!(!body["model_params"]["regime_state"].is_null(), "regime_state should always be present, got {body:?}");
    assert_eq!(body["model_params"]["regime_state"]["current_label"], "Bull");
}

/// Builds a `multipart/form-data` body with a single "file" field, the way
/// a browser's `<input type="file">` upload would.
fn multipart_body(filename: &str, content_type: &str, bytes: &[u8]) -> (String, Vec<u8>) {
    let boundary = "----driftriskcopilotboundary";
    let mut body = Vec::new();
    body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
    body.extend_from_slice(
        format!("Content-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n").as_bytes(),
    );
    body.extend_from_slice(format!("Content-Type: {content_type}\r\n\r\n").as_bytes());
    body.extend_from_slice(bytes);
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
    (format!("multipart/form-data; boundary={boundary}"), body)
}

async fn upload(app: axum::Router, filename: &str, content_type: &str, bytes: &[u8]) -> axum::response::Response {
    let (content_type_header, body) = multipart_body(filename, content_type, bytes);
    app.oneshot(
        Request::builder()
            .method("POST")
            .uri("/portfolio/upload")
            .header("content-type", content_type_header)
            .body(Body::from(body))
            .unwrap(),
    )
    .await
    .unwrap()
}

fn empty_backend_app() -> axum::Router {
    app_with_backend(MockBackend {
        experiment_result: None,
        ask_result: None,
        received_conversation_history: Mutex::new(None),
    })
}

#[tokio::test]
async fn upload_weight_based_csv_returns_200_and_correct_portfolio() {
    let csv = "ticker,weight\nRELIANCE.NS,0.6\nTCS.NS,0.4\n";
    let response = upload(empty_backend_app(), "portfolio.csv", "text/csv", csv.as_bytes()).await;

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;
    assert_eq!(body["layout_detected"], "weight");
    assert_eq!(body["row_count"], 2);
    let holdings = body["portfolio"]["holdings"].as_array().unwrap();
    assert_eq!(holdings.len(), 2);
    assert_eq!(holdings[0]["ticker"], "RELIANCE.NS");
    assert_eq!(holdings[0]["weight"], 0.6);
    assert_eq!(holdings[1]["ticker"], "TCS.NS");
    assert_eq!(holdings[1]["weight"], 0.4);
}

#[tokio::test]
async fn upload_value_based_csv_returns_200_and_weights_sum_to_one() {
    let csv = "ticker,shares,avg_price_inr\nRELIANCE.NS,10,2850.00\nHDFCBANK.NS,25,1640.00\n";
    let response = upload(empty_backend_app(), "portfolio.csv", "text/csv", csv.as_bytes()).await;

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;
    assert_eq!(body["layout_detected"], "value");
    let holdings = body["portfolio"]["holdings"].as_array().unwrap();
    let sum: f64 = holdings.iter().map(|h| h["weight"].as_f64().unwrap()).sum();
    assert!((sum - 1.0).abs() < 1e-6, "weights should sum to 1.0 within 1e-6, got {sum}");
}

#[tokio::test]
async fn upload_xlsx_returns_200() {
    let bytes = include_bytes!("../tests/fixtures/sample_portfolio.xlsx");
    let response = upload(
        empty_backend_app(),
        "portfolio.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        bytes,
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;
    assert_eq!(body["layout_detected"], "weight");
    assert_eq!(body["row_count"], 2);
}

#[tokio::test]
async fn upload_unsupported_file_type_returns_400() {
    let response = upload(empty_backend_app(), "portfolio.txt", "text/plain", b"not a real portfolio file").await;

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = body_json(response).await;
    assert_eq!(body["code"], "unsupported_file_type");
}

#[tokio::test]
async fn upload_weights_not_summing_to_one_returns_400() {
    let csv = "ticker,weight\nRELIANCE.NS,0.6\nTCS.NS,0.6\n";
    let response = upload(empty_backend_app(), "portfolio.csv", "text/csv", csv.as_bytes()).await;

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = body_json(response).await;
    assert_eq!(body["code"], "invalid_portfolio");
}

#[tokio::test]
async fn upload_single_holding_returns_400() {
    let csv = "ticker,weight\nRELIANCE.NS,1.0\n";
    let response = upload(empty_backend_app(), "portfolio.csv", "text/csv", csv.as_bytes()).await;

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = body_json(response).await;
    assert_eq!(body["code"], "invalid_portfolio");
}
