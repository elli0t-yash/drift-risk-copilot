mod support;

use agent::parse::{parse_experiment, ParseError};
use agent::schema::{CVAR_REBALANCE_FUNCTION, FACTOR_SHOCK_FUNCTION, RISK_DECOMPOSITION_FUNCTION};
use compute::experiments::{Experiment, Holding, Portfolio};
use support::{function_call_response, text_response, MockGeminiClient};

fn two_stock_portfolio() -> Portfolio {
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

#[tokio::test]
async fn parses_factor_shock_function_call() {
    let args = serde_json::json!({
        "shocks_pct": { "MARKET": -12.0, "BRENT": 20.0 },
        "propagate": true,
    });
    let client = MockGeminiClient::new(vec![function_call_response(FACTOR_SHOCK_FUNCTION, args)]);

    let portfolio = two_stock_portfolio();
    let experiment = parse_experiment(&client, "what if the market drops 12%", portfolio.clone())
        .await
        .unwrap();

    match experiment {
        Experiment::FactorShock(input) => {
            assert_eq!(input.shocks_pct.get("MARKET"), Some(&-12.0));
            assert_eq!(input.shocks_pct.get("BRENT"), Some(&20.0));
            assert!(input.propagate);
            assert_eq!(input.portfolio.tickers(), portfolio.tickers());
        }
        other => panic!("expected FactorShock, got {other:?}"),
    }
}

#[tokio::test]
async fn parses_risk_decomposition_function_call() {
    let args = serde_json::json!({});
    let client =
        MockGeminiClient::new(vec![function_call_response(RISK_DECOMPOSITION_FUNCTION, args)]);

    let portfolio = two_stock_portfolio();
    let experiment = parse_experiment(&client, "what's my portfolio risk?", portfolio.clone())
        .await
        .unwrap();

    match experiment {
        Experiment::RiskDecomposition(input) => {
            assert_eq!(input.portfolio.tickers(), portfolio.tickers());
        }
        other => panic!("expected RiskDecomposition, got {other:?}"),
    }
}

#[tokio::test]
async fn parses_cvar_rebalance_function_call() {
    let args = serde_json::json!({
        "per_name_cap": 0.2,
        "turnover_limit": 0.3,
        "confidence_level": 0.95,
    });
    let client = MockGeminiClient::new(vec![function_call_response(CVAR_REBALANCE_FUNCTION, args)]);

    let portfolio = two_stock_portfolio();
    let experiment = parse_experiment(&client, "rebalance to cut tail risk", portfolio.clone())
        .await
        .unwrap();

    match experiment {
        Experiment::CvarRebalance(input) => {
            assert_eq!(input.per_name_cap, 0.2);
            assert_eq!(input.turnover_limit, 0.3);
            assert_eq!(input.portfolio.tickers(), portfolio.tickers());
        }
        other => panic!("expected CvarRebalance, got {other:?}"),
    }
}

/// The caller's portfolio always wins over whatever (if anything) Gemini
/// put in the function-call args' portfolio field, since Gemini is never
/// given real holdings/weights to extract from.
#[tokio::test]
async fn caller_portfolio_overrides_any_portfolio_in_the_function_call_args() {
    let args = serde_json::json!({
        "portfolio": { "holdings": [{"ticker": "MADE_UP.NS", "weight": 1.0}], "total_value_inr": 1.0 },
    });
    let client =
        MockGeminiClient::new(vec![function_call_response(RISK_DECOMPOSITION_FUNCTION, args)]);

    let portfolio = two_stock_portfolio();
    let experiment = parse_experiment(&client, "risk please", portfolio.clone())
        .await
        .unwrap();

    match experiment {
        Experiment::RiskDecomposition(input) => {
            assert_eq!(input.portfolio.tickers(), portfolio.tickers());
        }
        other => panic!("expected RiskDecomposition, got {other:?}"),
    }
}

#[tokio::test]
async fn text_response_is_unrecognised() {
    let client = MockGeminiClient::new(vec![text_response(
        "I cannot map this message to a supported experiment.",
    )]);

    let err = parse_experiment(&client, "what's the weather today?", two_stock_portfolio())
        .await
        .unwrap_err();

    match err {
        ParseError::Unrecognised(text) => {
            assert_eq!(text, "I cannot map this message to a supported experiment.");
        }
        other => panic!("expected Unrecognised, got {other:?}"),
    }
}
