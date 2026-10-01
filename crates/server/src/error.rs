//! JSON error responses. Every error path returns `{ "error": string,
//! "code": string }` with the appropriate HTTP status, per spec.

use axum::extract::{FromRequest, Request};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::de::DeserializeOwned;

use crate::backend::BackendError;

#[derive(Debug)]
pub struct ApiError {
    pub status: StatusCode,
    pub code: &'static str,
    pub message: String,
    /// Extra top-level fields merged into the `{error, code}` response
    /// object (e.g. `portfolio/upload`'s `columns_found`/`missing`) --
    /// `None` for every error that only ever needed the base two fields.
    pub extra: Option<serde_json::Value>,
}

impl ApiError {
    pub fn bad_request(code: &'static str, message: impl Into<String>) -> Self {
        ApiError {
            status: StatusCode::BAD_REQUEST,
            code,
            message: message.into(),
            extra: None,
        }
    }

    pub fn not_found(code: &'static str, message: impl Into<String>) -> Self {
        ApiError {
            status: StatusCode::NOT_FOUND,
            code,
            message: message.into(),
            extra: None,
        }
    }

    pub fn new(status: StatusCode, code: &'static str, message: impl Into<String>) -> Self {
        ApiError {
            status,
            code,
            message: message.into(),
            extra: None,
        }
    }

    /// Like `bad_request`, but with extra fields merged into the response
    /// JSON alongside `error`/`code`. `extra` must serialize to a JSON
    /// object -- its keys become top-level response fields.
    pub fn bad_request_with_extra(code: &'static str, message: impl Into<String>, extra: serde_json::Value) -> Self {
        ApiError {
            status: StatusCode::BAD_REQUEST,
            code,
            message: message.into(),
            extra: Some(extra),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let mut body = serde_json::json!({
            "error": self.message,
            "code": self.code,
        });
        if let Some(extra) = self.extra {
            if let (Some(base), Some(extra)) = (body.as_object_mut(), extra.as_object()) {
                base.extend(extra.clone());
            }
        }
        (self.status, Json(body)).into_response()
    }
}

impl From<BackendError> for ApiError {
    fn from(err: BackendError) -> Self {
        match err {
            BackendError::Unrecognised(message) => ApiError {
                status: StatusCode::UNPROCESSABLE_ENTITY,
                code: "unrecognised_request",
                message,
                extra: None,
            },
            BackendError::Compute(message) => ApiError {
                status: StatusCode::INTERNAL_SERVER_ERROR,
                code: "compute_error",
                message,
                extra: None,
            },
            BackendError::GeminiUnavailable(message) => ApiError {
                status: StatusCode::SERVICE_UNAVAILABLE,
                code: "gemini_unavailable",
                message,
                extra: None,
            },
            BackendError::Internal(message) => ApiError {
                status: StatusCode::INTERNAL_SERVER_ERROR,
                code: "internal_error",
                message,
                extra: None,
            },
        }
    }
}

/// A `Json<T>` extractor whose rejection (malformed JSON, wrong content
/// type, missing/mistyped fields) is reported in the same `{error, code}`
/// shape as every other error response, instead of axum's default plain
/// text.
pub struct AppJson<T>(pub T);

#[async_trait::async_trait]
impl<T, S> FromRequest<S> for AppJson<T>
where
    T: DeserializeOwned,
    S: Send + Sync,
{
    type Rejection = ApiError;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        match Json::<T>::from_request(req, state).await {
            Ok(Json(value)) => Ok(AppJson(value)),
            Err(rejection) => Err(ApiError {
                status: rejection.status(),
                code: "invalid_json",
                message: rejection.body_text(),
                extra: None,
            }),
        }
    }
}
