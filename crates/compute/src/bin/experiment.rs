use std::path::PathBuf;

use clap::Parser;
use compute::data::load_market_data;
use compute::experiments::{
    run_factor_shock, run_risk_decomposition, Experiment,
};
use compute::model::fit_factor_model;
use compute::trace::DataWindow;

/// Runs a single experiment (FactorShock or RiskDecomposition) described by
/// a JSON file and prints its Evidence Trace to stdout.
#[derive(Parser, Debug)]
struct Args {
    /// Path to a JSON file containing a tagged `Experiment` value.
    input: PathBuf,

    /// Directory used to cache fetched price series.
    #[arg(long, default_value = "data/cache")]
    cache_dir: PathBuf,

    /// Refetch price series instead of reading from cache.
    #[arg(long)]
    refresh: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let raw = std::fs::read_to_string(&args.input)?;
    let experiment: Experiment = serde_json::from_str(&raw)?;

    let (portfolio, window) = match &experiment {
        Experiment::FactorShock(i) => (&i.portfolio, i.window),
        Experiment::RiskDecomposition(i) => (&i.portfolio, i.window),
        Experiment::CvarRebalance(_) => {
            eprintln!("CvarRebalance is a design note only in this checkpoint; not implemented.");
            std::process::exit(1);
        }
    };

    let tickers = portfolio.tickers();
    let data = load_market_data(&args.cache_dir, &tickers, args.refresh)?;
    let model = fit_factor_model(&data, &tickers, window)?;

    let data_window = DataWindow {
        window_days: window,
        start: data.dates[data.dates.len() - window],
        end: *data.dates.last().unwrap(),
    };

    let trace = match &experiment {
        Experiment::FactorShock(input) => {
            let (_, trace) = run_factor_shock(&data.quality, data_window, &model, input)?;
            trace
        }
        Experiment::RiskDecomposition(input) => {
            let (_, trace) = run_risk_decomposition(&data.quality, data_window, &model, input)?;
            trace
        }
        Experiment::CvarRebalance(_) => unreachable!(),
    };

    println!("{}", serde_json::to_string_pretty(&trace)?);
    Ok(())
}
