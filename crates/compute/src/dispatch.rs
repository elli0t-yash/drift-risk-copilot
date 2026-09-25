//! The single entry point that runs any `Experiment` end to end: data
//! fetch, model fit, and the appropriate experiment function, returning its
//! `EvidenceTrace`. Moved here (from `agent::pipeline::compute_trace`,
//! which is now a thin wrapper around this) because `RiskDrift` needs read
//! access to the snapshot store to resolve its baseline, and the `store`
//! crate is a dependency of `compute`, not of the (Gemini-facing) `agent`
//! crate's narrower concerns.

use std::path::Path;

use crate::context::ExperimentContext;
use crate::data::MarketData;
use crate::experiments::Experiment;
use crate::model::Frequency;
use crate::trace::{DataWindow, EvidenceTrace};
use crate::Result;

const CACHE_DIR: &str = "data/cache";

fn build_data_window(data: &MarketData, window: usize, frequency: Frequency) -> DataWindow {
    DataWindow {
        frequency,
        window_periods: window,
        start: data.dates[data.dates.len() - window],
        end: *data.dates.last().unwrap(),
    }
}

/// Runs `experiment` (data fetch + model fit + the appropriate experiment
/// function) and returns its `EvidenceTrace`. Synchronous/blocking --
/// callers on an async executor (e.g. `agent::pipeline::run`) should run it
/// via `tokio::task::spawn_blocking`.
///
/// `portfolio` is `experiment`'s portfolio, always -- for every variant
/// except `RiskDrift`, this is the same portfolio already embedded in the
/// variant's own input (`FactorShockInput::portfolio` etc.), so it's
/// slightly redundant there; `RiskDriftInput` carries no `portfolio` field
/// of its own (it operates on "the current portfolio" vs. a *stored*
/// baseline, not two portfolios given inline), so this is its only source
/// of one. `ctx` is likewise only used by `RiskDrift`; every other variant
/// receives it unused (reserved for a later session's policy checks).
pub fn run_experiment(
    experiment: &Experiment,
    portfolio: &crate::experiments::Portfolio,
    ctx: &ExperimentContext,
) -> Result<EvidenceTrace> {
    let cache_dir = Path::new(CACHE_DIR);
    match experiment {
        Experiment::FactorShock(input) => {
            let tickers = input.portfolio.tickers();
            let window = input.resolved_window();
            let data = crate::data::load_market_data(cache_dir, &tickers, false, input.frequency)?;
            let model = crate::model::fit_factor_model(
                &data,
                &tickers,
                crate::model::ModelConfig::new(window, input.frequency),
            )?;
            let data_window = build_data_window(&data, window, input.frequency);
            let (_, trace) = crate::experiments::run_factor_shock(&data.quality, data_window, &model, input)?;
            Ok(trace)
        }
        Experiment::RiskDecomposition(input) => {
            let tickers = input.portfolio.tickers();
            let window = input.resolved_window();
            let data = crate::data::load_market_data(cache_dir, &tickers, false, input.frequency)?;
            let model = crate::model::fit_factor_model(
                &data,
                &tickers,
                crate::model::ModelConfig::new(window, input.frequency),
            )?;
            let data_window = build_data_window(&data, window, input.frequency);
            let (_, trace) =
                crate::experiments::run_risk_decomposition(&data.quality, data_window, &model, input)?;
            Ok(trace)
        }
        Experiment::CvarRebalance(input) => {
            let tickers = input.portfolio.tickers();
            let data = crate::data::load_market_data(cache_dir, &tickers, false, input.frequency)?;
            let (_, trace) = crate::cvar::run_cvar_rebalance(&data.quality, &data, input)?;
            Ok(trace)
        }
        Experiment::PortfolioPerformance(input) => {
            let tickers = input.portfolio.tickers();
            let window = input.resolved_window();
            let data = crate::data::load_market_data(cache_dir, &tickers, false, input.frequency)?;
            let data_window = build_data_window(&data, window, input.frequency);
            let (_, trace) =
                crate::performance::run_portfolio_performance(&data.quality, data_window, &data, input)?;
            Ok(trace)
        }
        Experiment::RiskDrift(input) => {
            let (_, trace) = crate::drift::run_risk_drift(cache_dir, portfolio, input, ctx)?;
            Ok(trace)
        }
    }
}
