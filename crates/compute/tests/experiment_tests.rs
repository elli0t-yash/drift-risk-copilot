mod common;

use std::collections::BTreeMap;

use compute::experiments::{
    run_factor_shock, run_risk_decomposition, FactorShockInput, Holding, Portfolio,
    RiskDecompositionInput,
};
use compute::model::fit_factor_model;
use compute::trace::DataWindow;

fn two_stock_model() -> (compute::data::MarketData, compute::model::FactorModel) {
    let stocks = [
        ("AAA", 0.0001, [0.9, 0.1, 0.0, 0.2, 0.3]),
        ("BBB", -0.0002, [0.4, -0.3, 0.5, 0.0, -0.1]),
    ];
    let data = common::synthetic_multi_stock(&stocks, 300, 0.0008, 123);
    let tickers = vec!["AAA".to_string(), "BBB".to_string()];
    let model = fit_factor_model(&data, &tickers, 252).unwrap();
    (data, model)
}

fn data_window(data: &compute::data::MarketData, window: usize) -> DataWindow {
    DataWindow {
        window_days: window,
        start: data.dates[data.dates.len() - window],
        end: *data.dates.last().unwrap(),
    }
}

#[test]
fn euler_contributions_sum_to_portfolio_vol_stock_and_factor_views() {
    let (data, model) = two_stock_model();
    let portfolio = Portfolio {
        holdings: vec![
            Holding {
                ticker: "AAA".to_string(),
                weight: 0.6,
            },
            Holding {
                ticker: "BBB".to_string(),
                weight: 0.4,
            },
        ],
        total_value_inr: 1_000_000.0,
    };
    let input = RiskDecompositionInput {
        portfolio,
        window: 252,
    };

    let (output, trace) =
        run_risk_decomposition(&data.quality, data_window(&data, 252), &model, &input).unwrap();

    for inv in &trace.invariants {
        assert!(inv.passed, "invariant failed: {} ({})", inv.name, inv.detail);
    }

    let stock_sum: f64 = output.by_stock.iter().map(|s| s.contribution).sum();
    assert!((stock_sum - output.portfolio_vol_annualized).abs() < 1e-9);

    let factor_plus_specific: f64 = output.by_factor.iter().map(|f| f.contribution).sum::<f64>()
        + output.specific_risk_contribution;
    assert!((factor_plus_specific - output.portfolio_vol_annualized).abs() < 1e-9);
}

#[test]
fn factor_shock_pnl_is_linear_in_shock_size() {
    let (data, model) = two_stock_model();
    let portfolio = Portfolio {
        holdings: vec![
            Holding {
                ticker: "AAA".to_string(),
                weight: 0.6,
            },
            Holding {
                ticker: "BBB".to_string(),
                weight: 0.4,
            },
        ],
        total_value_inr: 1_000_000.0,
    };

    let mut shocks = BTreeMap::new();
    shocks.insert("MARKET".to_string(), -12.0);

    let input1 = FactorShockInput {
        portfolio: portfolio.clone(),
        shocks_pct: shocks.clone(),
        propagate: false,
        window: 252,
    };
    let mut shocks2 = shocks.clone();
    *shocks2.get_mut("MARKET").unwrap() *= 2.0;
    let input2 = FactorShockInput {
        portfolio,
        shocks_pct: shocks2,
        propagate: false,
        window: 252,
    };

    let (out1, trace1) =
        run_factor_shock(&data.quality, data_window(&data, 252), &model, &input1).unwrap();
    let (out2, _) =
        run_factor_shock(&data.quality, data_window(&data, 252), &model, &input2).unwrap();

    for inv in &trace1.invariants {
        assert!(inv.passed, "invariant failed: {} ({})", inv.name, inv.detail);
    }

    assert!(
        (out2.portfolio_pnl_inr - 2.0 * out1.portfolio_pnl_inr).abs()
            < 1e-6 * out1.portfolio_pnl_inr.abs().max(1.0),
        "doubling the shock should double P&L: {} vs 2x{}",
        out2.portfolio_pnl_inr,
        out1.portfolio_pnl_inr
    );
}

#[test]
fn conditional_propagation_is_noop_when_all_factors_given() {
    let (data, model) = two_stock_model();
    let portfolio = Portfolio {
        holdings: vec![
            Holding {
                ticker: "AAA".to_string(),
                weight: 0.6,
            },
            Holding {
                ticker: "BBB".to_string(),
                weight: 0.4,
            },
        ],
        total_value_inr: 1_000_000.0,
    };

    let mut shocks = BTreeMap::new();
    shocks.insert("MARKET".to_string(), -12.0);
    shocks.insert("USDINR".to_string(), 2.0);
    shocks.insert("BRENT".to_string(), 20.0);
    shocks.insert("GOLD".to_string(), 5.0);
    shocks.insert("RATES_PROXY".to_string(), 1.0);

    let input = FactorShockInput {
        portfolio,
        shocks_pct: shocks,
        propagate: true,
        window: 252,
    };

    let (output, _) =
        run_factor_shock(&data.quality, data_window(&data, 252), &model, &input).unwrap();

    assert!(
        output.implied_shocks.is_empty(),
        "no factors should need propagation when all five are specified"
    );
}
