//! A minimal async Gemini client: request/response types for the
//! `generateContent` endpoint, function-calling support, and a retrying
//! HTTP transport. `GeminiClient` is a trait so `parse`/`narrate` can be
//! tested against a mock implementation with no network access.

use std::time::Duration;

use serde::{Deserialize, Serialize};
use thiserror::Error;

const API_BASE: &str = "https://generativelanguage.googleapis.com/v1beta/models";
// The checkpoint spec that introduced this client named gemini-2.0-flash.
// Confirmed live (2026-09-24) that model has been retired: the API now
// returns 404 NOT_FOUND for it, with the response body itself pointing to
// gemini-3.8-flash as the replacement ("This model models/gemini-2.0-flash
// is no longer available... use models/gemini-3.8-flash").
const MODEL: &str = "gemini-3.8-flash";
const MAX_ATTEMPTS: u32 = 3;

#[derive(Debug, Error)]
pub enum GeminiError {
    #[error("GEMINI_API_KEY environment variable is not set")]
    MissingApiKey,
    #[error("http request failed: {0}")]
    Http(#[from] reqwest::Error),
    #[error("gemini returned status {status}: {body}")]
    Status { status: u16, body: String },
    #[error("failed to parse gemini response as json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("gemini response had no candidates")]
    NoCandidates,
}

/// One turn's content: a role (absent for `system_instruction`) plus parts.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Content {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    pub parts: Vec<Part>,
}

/// A single part of a `Content`. Gemini's wire format has at most one of
/// these fields set per part, so this mirrors that as optional fields
/// rather than a tagged enum (which would not round-trip Gemini's actual
/// JSON shape).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Part {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_call: Option<FunctionCall>,
}

impl Part {
    pub fn text(text: impl Into<String>) -> Self {
        Part {
            text: Some(text.into()),
            function_call: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    #[serde(default)]
    pub args: serde_json::Value,
}

#[derive(Debug, Clone, Serialize)]
pub struct FunctionDeclaration {
    pub name: String,
    pub description: String,
    /// JSON Schema (as accepted by Gemini's function-calling subset) for
    /// the function's arguments.
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Tool {
    pub function_declarations: Vec<FunctionDeclaration>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GeminiRequest {
    pub contents: Vec<Content>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_instruction: Option<Content>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
}

impl GeminiRequest {
    /// A single user-turn request with a system instruction and no tools
    /// (the narration call shape).
    pub fn user_turn(system_prompt: impl Into<String>, user_text: impl Into<String>) -> Self {
        GeminiRequest {
            contents: vec![Content {
                role: Some("user".to_string()),
                parts: vec![Part::text(user_text)],
            }],
            system_instruction: Some(Content {
                role: None,
                parts: vec![Part::text(system_prompt)],
            }),
            tools: None,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Candidate {
    pub content: Content,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GeminiResponse {
    #[serde(default)]
    pub candidates: Vec<Candidate>,
}

impl GeminiResponse {
    /// The first candidate's first part, whichever of text/function_call it has.
    pub fn first_part(&self) -> Option<&Part> {
        self.candidates.first()?.content.parts.first()
    }
}

/// Anything that can execute a Gemini `generateContent` call. A trait so
/// tests can supply a mock implementation with canned responses instead of
/// making network calls.
#[async_trait::async_trait]
pub trait GeminiClient: Send + Sync {
    async fn generate(&self, request: &GeminiRequest) -> Result<GeminiResponse, GeminiError>;
}

/// The real client: POSTs to Gemini's `generateContent` endpoint, retrying
/// up to `MAX_ATTEMPTS` times with exponential backoff on 429/503.
pub struct HttpGeminiClient {
    http: reqwest::Client,
    api_key: String,
    model: String,
}

impl HttpGeminiClient {
    /// Reads `GEMINI_API_KEY` from the environment.
    pub fn new() -> Result<Self, GeminiError> {
        let api_key = std::env::var("GEMINI_API_KEY").map_err(|_| GeminiError::MissingApiKey)?;
        Ok(HttpGeminiClient {
            http: reqwest::Client::new(),
            api_key,
            model: MODEL.to_string(),
        })
    }
}

#[async_trait::async_trait]
impl GeminiClient for HttpGeminiClient {
    async fn generate(&self, request: &GeminiRequest) -> Result<GeminiResponse, GeminiError> {
        let url = format!("{API_BASE}/{}:generateContent", self.model);
        let mut attempt = 0u32;
        loop {
            attempt += 1;
            let resp = self
                .http
                .post(&url)
                .query(&[("key", self.api_key.as_str())])
                .json(request)
                .send()
                .await?;
            let status = resp.status();
            if status.is_success() {
                let text = resp.text().await?;
                let body: GeminiResponse = serde_json::from_str(&text)?;
                return Ok(body);
            }
            let retryable = status.as_u16() == 429 || status.as_u16() == 503;
            if retryable && attempt < MAX_ATTEMPTS {
                let backoff = Duration::from_millis(250 * 2u64.pow(attempt - 1));
                tokio::time::sleep(backoff).await;
                continue;
            }
            let body = resp.text().await.unwrap_or_default();
            return Err(GeminiError::Status {
                status: status.as_u16(),
                body,
            });
        }
    }
}

/// Calls `client.generate`, a thin free function matching the spec's
/// `generate(client, request) -> Result<GeminiResponse, GeminiError>` shape.
pub async fn generate<C: GeminiClient>(
    client: &C,
    request: &GeminiRequest,
) -> Result<GeminiResponse, GeminiError> {
    client.generate(request).await
}
