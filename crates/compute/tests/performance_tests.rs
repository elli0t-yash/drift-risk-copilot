mod common;

use compute::experiments::{Holding, Portfolio};
use compute::model::Frequency;
use compute::performance::{run_portfolio_performance, PortfolioPerformanceInput};
use compute::trace::DataWindow;

fn window_and_trace_helper(
    data: &compute::data::MarketData,
    input: &PortfolioPerformanceInput,
) -> (
    compute::performance::PortfolioPerformanceOutput,
    compute::trace::EvidenceTrace,
) {
    let window = input.resolved_window();
    let data_window = DataWindow {
        frequency: input.frequency,
        window_periods: window,
        start: data.dates[data.dates.len() - window],
        end: *data.dates.last().unwrap(),
    };
    run_portfolio_performance(&data.quality, data_window, data, input).unwrap()
}

/// A single stock with a constant +1% daily log return should compound to
/// a known, exactly-checkable total/annualized return, zero volatility,
/// and zero drawdown -- the simplest possible correctness check.
#[test]
fn constant_positive_return_compounds_correctly_with_zero_vol_and_drawdown() {
    let n = 100;
    let log_r = 0.01_f64;
    let data = common::market_data_from_log_returns(&[("STEADY", vec![log_r; n])]);

    let input = PortfolioPerformanceInput {
        portfolio: Portfolio {
            holdings: vec![Holding { ticker: "STEADY".to_string(), weight: 1.0 }],
            total_value_inr: 1_000_000.0,
        },
        frequency: Frequency::Daily,
        window: Some(n),
    };

    let (output, trace) = window_and_trace_helper(&data, &input);

    let simple_r = log_r.exp() - 1.0;
    let expected_total_return = (1.0 + simple_r).powi(n as i32) - 1.0;
    assert!(
        (output.total_return - expected_total_return).abs() < 1e-9,
        "total_return {} != expected {}",
        output.total_return,
        expected_total_return
    );
    assert!(output.annualized_vol_realized.abs() < 1e-9, "expected ~zero vol, got {}", output.annualized_vol_realized);
    assert!(output.max_drawdown.abs() < 1e-9, "expected zero drawdown for a monotonically rising series");
    assert!(output.end_value_inr > output.start_value_inr);

    for inv in &trace.invariants {
        assert!(inv.passed, "invariant failed: {} ({})", inv.name, inv.detail);
    }
}

/// A portfolio that falls 50% then fully recovers should show a real
/// max_drawdown of about -50% even though total_return ends near zero.
#[test]
fn drawdown_is_captured_even_when_total_return_recovers_to_near_zero() {
    let mut returns = vec![(0.5_f64).ln()]; // -50% in one day (as a log return)
    returns.extend(vec![(2.0_f64).ln()]); // +100% the next day: back to ~1.0x
    returns.extend(vec![0.0; 8]); // pad to a reasonable window

    let data = common::market_data_from_log_returns(&[("VSHAPE", returns)]);

    let input = PortfolioPerformanceInput {
        portfolio: Portfolio {
            holdings: vec![Holding { ticker: "VSHAPE".to_string(), weight: 1.0 }],
            total_value_inr: 1_000_000.0,
        },
        frequency: Frequency::Daily,
        window: Some(10),
    };

    let (output, _trace) = window_and_trace_helper(&data, &input);

    assert!(
        output.max_drawdown < -0.45,
        "expected a deep drawdown from the -50% day, got {}",
        output.max_drawdown
    );
    assert!(
        output.total_return.abs() < 1e-6,
        "expected the portfolio to have round-tripped back to ~0 total return, got {}",
        output.total_return
    );
}

/// Two-stock portfolio: the invariant (end_value == start_value * (1 +
/// total_return)) should hold regardless of the per-holding weight split.
#[test]
fn two_holding_portfolio_satisfies_its_own_value_invariant() {
    let n = 50;
    let mut rng = common::Rng::new(11);
    let a: Vec<f64> = (0..n).map(|_| rng.next_signed() * 0.01).collect();
    let b: Vec<f64> = (0..n).map(|_| rng.next_signed() * 0.015).collect();
    let data = common::market_data_from_log_returns(&[("A", a), ("B", b)]);

    let input = PortfolioPerformanceInput {
        portfolio: Portfolio {
            holdings: vec![
                Holding { ticker: "A".to_string(), weight: 0.3 },
                Holding { ticker: "B".to_string(), weight: 0.7 },
            ],
            total_value_inr: 2_500_000.0,
        },
        frequency: Frequency::Daily,
        window: Some(n),
    };

    let (output, trace) = window_and_trace_helper(&data, &input);

    assert!(
        (output.end_value_inr - output.start_value_inr * (1.0 + output.total_return)).abs() < 1.0,
        "end_value_inr should match start_value_inr * (1 + total_return)"
    );
    for inv in &trace.invariants {
        assert!(inv.passed, "invariant failed: {} ({})", inv.name, inv.detail);
    }
    assert_eq!(trace.experiment, "PortfolioPerformance");
    assert!(trace.model_params.regime_state.is_none(), "this experiment never fits a factor model");
}
