//! The single entry point the `server` crate calls: NL message + portfolio
//! in, a parsed `Experiment` + computed `EvidenceTrace` + grounded
//! narration out.

use std::path::Path;

use compute::experiments::{Experiment, Portfolio};
use compute::trace::{DataWindow, EvidenceTrace};
use thiserror::Error;

use crate::conversation::ConversationTurn;
use crate::gemini::GeminiClient;
use crate::grounding::GroundedNarration;
use crate::narrate::NarrateError;
use crate::parse::{parse_experiment, ParseError};
use crate::suggest::suggest_follow_up;

const CACHE_DIR: &str = "data/cache";

#[derive(Debug, Error)]
pub enum PipelineError {
    #[error("could not parse the request: {0}")]
    Parse(#[from] ParseError),
    #[error("compute error: {0}")]
    Compute(#[from] compute::ComputeError),
    #[error("narration error: {0}")]
    Narrate(#[from] NarrateError),
    #[error("suggestion error: {0}")]
    Suggest(#[from] crate::suggest::SuggestError),
}

#[derive(Clone)]
pub struct PipelineResult {
    pub experiment: Experiment,
    pub trace: EvidenceTrace,
    pub narration: GroundedNarration,
    /// This turn's narration, as an `assistant` `ConversationTurn` ready
    /// for the caller to append to its `conversation_history` before the
    /// next request.
    pub assistant_turn: ConversationTurn,
    /// One follow-up question a risk manager would naturally ask next
    /// (see `suggest`). Not grounding-checked -- it's a question, not a
    /// factual claim about the trace.
    pub suggestion: String,
}

/// parse -> compute -> narrate (with grounding check) -> suggest a
/// follow-up. `conversation_history` (if any) is threaded through both the
/// parse and narrate Gemini calls so multi-turn references resolve
/// correctly; an empty history behaves exactly as before it existed.
pub async fn run<C: GeminiClient>(
    client: &C,
    user_message: &str,
    portfolio: Portfolio,
    conversation_history: &[ConversationTurn],
) -> Result<PipelineResult, PipelineError> {
    let experiment = parse_experiment(client, user_message, portfolio, conversation_history).await?;

    // The compute layer's data/model-fitting path is synchronous
    // (blocking HTTP + CPU-bound linear algebra); run it on a blocking
    // thread so it doesn't stall the async executor running the Gemini
    // calls.
    let experiment_for_compute = experiment.clone();
    let trace = tokio::task::spawn_blocking(move || compute_trace(&experiment_for_compute))
        .await
        .expect("compute_trace task panicked")?;

    // narrate and suggest are both independent Gemini calls that only
    // depend on `trace` (not on each other's output -- suggest takes a
    // short summary built from the trace directly, not the narration), so
    // run them concurrently rather than back-to-back.
    let summary = experiment_summary(&trace);
    let (narration_result, suggestion_result) = tokio::join!(
        crate::grounding::grounded_narrate(client, &trace, conversation_history),
        suggest_follow_up(client, &summary),
    );
    let narration = narration_result?;
    let suggestion = suggestion_result?;
    let assistant_turn = ConversationTurn::assistant(narration.narration.clone());

    Ok(PipelineResult {
        experiment,
        trace,
        narration,
        assistant_turn,
        suggestion,
    })
}

/// A short plain-text summary of `trace` (experiment type + one key output
/// number), used as `suggest_follow_up`'s input instead of the narration --
/// the two Gemini calls run concurrently (see `run`), so suggest can't wait
/// on narrate's output.
fn experiment_summary(trace: &EvidenceTrace) -> String {
    let result = trace.outputs.get("result");
    let number = |path: &[&str]| -> Option<f64> {
        let mut v = result?;
        for key in path {
            v = v.get(key)?;
        }
        v.as_f64()
    };
    match trace.experiment.as_str() {
        "FactorShock" => match number(&["portfolio_pnl_inr"]) {
            Some(pnl) => format!("FactorShock experiment result: portfolio P&L is {pnl:.2} INR."),
            None => "FactorShock experiment result.".to_string(),
        },
        "RiskDecomposition" => match number(&["portfolio_vol_annualized"]) {
            Some(vol) => {
                format!("RiskDecomposition experiment result: annualised portfolio vol is {vol:.6}.")
            }
            None => "RiskDecomposition experiment result.".to_string(),
        },
        "CvarRebalance" => {
            let before = number(&["stats_before", "historical_cvar"]);
            let after = number(&["stats_after", "historical_cvar"]);
            match (before, after) {
                (Some(b), Some(a)) => format!(
                    "CvarRebalance experiment result: historical CVaR went from {b:.6} to {a:.6}."
                ),
                _ => "CvarRebalance experiment result.".to_string(),
            }
        }
        "PortfolioPerformance" => match number(&["total_return"]) {
            Some(r) => format!(
                "PortfolioPerformance experiment result: total return over the window is {r:.4}."
            ),
            None => "PortfolioPerformance experiment result.".to_string(),
        },
        other => format!("{other} experiment result."),
    }
}

fn build_data_window(
    data: &compute::data::MarketData,
    window: usize,
    frequency: compute::model::Frequency,
) -> DataWindow {
    DataWindow {
        frequency,
        window_periods: window,
        start: data.dates[data.dates.len() - window],
        end: *data.dates.last().unwrap(),
    }
}

/// Runs the compute-layer path (data fetch + model fit + the appropriate
/// experiment) for `experiment`. Synchronous/blocking (see the module-level
/// note in `run`'s body about `spawn_blocking`); exposed so `server`'s
/// direct `/experiment` route can reuse it without a Gemini round trip.
pub fn compute_trace(experiment: &Experiment) -> compute::Result<EvidenceTrace> {
    let cache_dir = Path::new(CACHE_DIR);
    match experiment {
        Experiment::FactorShock(input) => {
            let tickers = input.portfolio.tickers();
            let window = input.resolved_window();
            let data =
                compute::data::load_market_data(cache_dir, &tickers, false, input.frequency)?;
            let model = compute::model::fit_factor_model_with_config(
                &data,
                &tickers,
                compute::model::ModelConfig::new(window, input.frequency)
                    .with_regime_covariance(input.regime_covariance),
            )?;
            let data_window = build_data_window(&data, window, input.frequency);
            let (_, trace) =
                compute::experiments::run_factor_shock(&data.quality, data_window, &model, input)?;
            Ok(trace)
        }
        Experiment::RiskDecomposition(input) => {
            let tickers = input.portfolio.tickers();
            let window = input.resolved_window();
            let data =
                compute::data::load_market_data(cache_dir, &tickers, false, input.frequency)?;
            let model = compute::model::fit_factor_model_with_config(
                &data,
                &tickers,
                compute::model::ModelConfig::new(window, input.frequency)
                    .with_regime_covariance(input.regime_covariance),
            )?;
            let data_window = build_data_window(&data, window, input.frequency);
            let (_, trace) = compute::experiments::run_risk_decomposition(
                &data.quality,
                data_window,
                &model,
                input,
            )?;
            Ok(trace)
        }
        Experiment::CvarRebalance(input) => {
            let tickers = input.portfolio.tickers();
            let data =
                compute::data::load_market_data(cache_dir, &tickers, false, input.frequency)?;
            let (_, trace) = compute::cvar::run_cvar_rebalance(&data.quality, &data, input)?;
            Ok(trace)
        }
        Experiment::PortfolioPerformance(input) => {
            let tickers = input.portfolio.tickers();
            let window = input.resolved_window();
            let data =
                compute::data::load_market_data(cache_dir, &tickers, false, input.frequency)?;
            let data_window = build_data_window(&data, window, input.frequency);
            let (_, trace) = compute::performance::run_portfolio_performance(
                &data.quality,
                data_window,
                &data,
                input,
            )?;
            Ok(trace)
        }
    }
}
