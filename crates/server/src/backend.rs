//! The `Backend` trait separates route handlers from the concrete
//! compute/Gemini calls, so routes can be tested against a mock backend
//! with no network access (Yahoo Finance or Gemini).

use agent::gemini::GeminiError;
use agent::pipeline::PipelineResult;
use agent::ConversationTurn;
use compute::experiments::{Experiment, Portfolio};
use compute::trace::EvidenceTrace;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum BackendError {
    /// The NL request didn't map to a supported experiment; the model's
    /// one-sentence explanation. Maps to 422.
    #[error("{0}")]
    Unrecognised(String),
    /// Any error from the compute layer (data fetch, model fit, LP solve,
    /// invalid input). Maps to 500.
    #[error("compute error: {0}")]
    Compute(String),
    /// Gemini returned an error (after its own internal retries) or a
    /// malformed response. Maps to 503.
    #[error("gemini unavailable: {0}")]
    GeminiUnavailable(String),
    /// Anything else unexpected. Maps to 500.
    #[error("internal error: {0}")]
    Internal(String),
}

impl From<GeminiError> for BackendError {
    fn from(err: GeminiError) -> Self {
        match err {
            // A missing API key is a startup-time misconfiguration (main()
            // already refuses to start without one); if it's somehow hit
            // at request time, it's ours to fix, not a transient upstream
            // failure.
            GeminiError::MissingApiKey => BackendError::Internal(err.to_string()),
            _ => BackendError::GeminiUnavailable(err.to_string()),
        }
    }
}

impl From<agent::parse::ParseError> for BackendError {
    fn from(err: agent::parse::ParseError) -> Self {
        match err {
            agent::parse::ParseError::Unrecognised(text) => BackendError::Unrecognised(text),
            agent::parse::ParseError::Gemini(g) => BackendError::from(g),
            other => BackendError::Internal(other.to_string()),
        }
    }
}

impl From<agent::narrate::NarrateError> for BackendError {
    fn from(err: agent::narrate::NarrateError) -> Self {
        match err {
            agent::narrate::NarrateError::Gemini(g) => BackendError::from(g),
            other => BackendError::Internal(other.to_string()),
        }
    }
}

impl From<agent::suggest::SuggestError> for BackendError {
    fn from(err: agent::suggest::SuggestError) -> Self {
        match err {
            agent::suggest::SuggestError::Gemini(g) => BackendError::from(g),
            other => BackendError::Internal(other.to_string()),
        }
    }
}

impl From<agent::pipeline::PipelineError> for BackendError {
    fn from(err: agent::pipeline::PipelineError) -> Self {
        match err {
            agent::pipeline::PipelineError::Parse(p) => BackendError::from(p),
            agent::pipeline::PipelineError::Compute(c) => BackendError::Compute(c.to_string()),
            agent::pipeline::PipelineError::Narrate(n) => BackendError::from(n),
            agent::pipeline::PipelineError::Suggest(s) => BackendError::from(s),
        }
    }
}

impl From<compute::ComputeError> for BackendError {
    fn from(err: compute::ComputeError) -> Self {
        BackendError::Compute(err.to_string())
    }
}

/// What the route handlers depend on. `RealBackend` wraps live
/// compute + Gemini calls; tests use a `MockBackend` (see
/// `tests/support`) with canned responses.
#[async_trait::async_trait]
pub trait Backend: Send + Sync {
    async fn run_experiment(&self, experiment: Experiment) -> Result<EvidenceTrace, BackendError>;
    async fn run_ask(
        &self,
        portfolio: Portfolio,
        message: String,
        conversation_history: Vec<ConversationTurn>,
    ) -> Result<PipelineResult, BackendError>;
}

pub struct RealBackend {
    gemini: agent::gemini::HttpGeminiClient,
}

impl RealBackend {
    pub fn new(gemini: agent::gemini::HttpGeminiClient) -> Self {
        RealBackend { gemini }
    }
}

#[async_trait::async_trait]
impl Backend for RealBackend {
    async fn run_experiment(&self, experiment: Experiment) -> Result<EvidenceTrace, BackendError> {
        let trace = tokio::task::spawn_blocking(move || agent::pipeline::compute_trace(&experiment))
            .await
            .map_err(|e| BackendError::Internal(format!("compute task panicked: {e}")))??;
        Ok(trace)
    }

    async fn run_ask(
        &self,
        portfolio: Portfolio,
        message: String,
        conversation_history: Vec<ConversationTurn>,
    ) -> Result<PipelineResult, BackendError> {
        let result =
            agent::pipeline::run(&self.gemini, &message, portfolio, &conversation_history).await?;
        Ok(result)
    }
}
