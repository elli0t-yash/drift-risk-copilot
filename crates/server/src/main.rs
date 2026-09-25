mod backend;
mod error;
mod logging;
mod pdf;
mod routes;
mod store;
mod validate;

use std::sync::Arc;

use axum::routing::{get, post};
use axum::Router;

use backend::{Backend, RealBackend};
use routes::AppState;
use store::ResultStore;

#[tokio::main]
async fn main() {
    init_tracing();

    let gemini = match agent::gemini::HttpGeminiClient::new() {
        Ok(client) => client,
        Err(err) => {
            tracing::error!(error = %err, "failed to start: {err}");
            std::process::exit(1);
        }
    };
    let backend: Arc<dyn Backend> = Arc::new(RealBackend::new(gemini));
    let store = Arc::new(ResultStore::new());
    let state = AppState { backend, store };

    let app = build_router(state);

    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8080);
    let addr = format!("0.0.0.0:{port}");

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .unwrap_or_else(|err| panic!("failed to bind {addr}: {err}"));
    tracing::info!(%addr, "listening");

    axum::serve(listener, app)
        .await
        .expect("server error");
}

fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(routes::health))
        .route("/scenarios", get(routes::get_scenarios))
        .route("/experiment", post(routes::post_experiment))
        .route("/ask", post(routes::post_ask))
        .route("/report/:result_id", get(routes::get_report))
        .fallback(routes::static_handler)
        .layer(axum::middleware::from_fn(logging::log_requests))
        .with_state(state)
}

fn init_tracing() {
    use tracing_subscriber::EnvFilter;

    tracing_subscriber::fmt()
        .json()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
        .init();
}

#[cfg(test)]
mod tests;
