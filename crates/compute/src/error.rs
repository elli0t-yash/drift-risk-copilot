use thiserror::Error;

#[derive(Debug, Error)]
pub enum ComputeError {
    #[error("http request failed: {0}")]
    Http(#[from] reqwest::Error),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("csv error: {0}")]
    Csv(#[from] csv::Error),

    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("data error: {0}")]
    Data(String),

    #[error("model error: {0}")]
    Model(String),

    #[error("invalid input: {0}")]
    InvalidInput(String),
}

pub type Result<T> = std::result::Result<T, ComputeError>;
