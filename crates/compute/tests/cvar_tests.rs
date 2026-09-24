mod common;

use compute::cvar::{run_cvar_rebalance, CvarRebalanceInput};
use compute::experiments::{Holding, Portfolio};

/// CRASH has 95 calm scenarios (+/-1% log return) and 5 severe crash
/// scenarios (-40% log return); SAFE is calm throughout. With a 95%
/// confidence level, k = round(100 * 0.05) = 5 exactly matches the crash
/// count, so any weight on CRASH pulls the entire CVaR tail down with it.
/// A CVaR-minimizing, adequately-capitalized rebalance should cut CRASH to
/// (near) zero.
fn crash_vs_safe_returns() -> (Vec<f64>, Vec<f64>) {
    let n = 100;
    let mut crash = Vec::with_capacity(n);
    let mut safe = Vec::with_capacity(n);
    for i in 0..n {
        // Deterministic small alternating noise, no RNG needed.
        let noise = if i % 2 == 0 { 0.01 } else { -0.01 };
        safe.push(noise * 0.5);
        if i < 5 {
            crash.push(-0.40);
        } else {
            crash.push(noise);
        }
    }
    (crash, safe)
}

#[test]
fn heavy_tail_asset_is_cut_to_near_zero_when_turnover_allows() {
    let (crash, safe) = crash_vs_safe_returns();
    let data = common::market_data_from_log_returns(&[("CRASH", crash), ("SAFE", safe)]);

    let input = CvarRebalanceInput {
        portfolio: Portfolio {
            holdings: vec![
                Holding {
                    ticker: "CRASH".to_string(),
                    weight: 0.5,
                },
                Holding {
                    ticker: "SAFE".to_string(),
                    weight: 0.5,
                },
            ],
            total_value_inr: 1_000_000.0,
        },
        confidence_level: 0.95,
        // 1.0 (not < 1.0): a tighter per-name cap on both names would force
        // some residual weight onto CRASH just to make weights sum to 1,
        // which is a real cap-driven effect, not what this test is after.
        per_name_cap: 1.0,
        turnover_limit: 1.5,
        commission_bps: 10.0,
        frequency: compute::model::Frequency::Daily,
        window: None,
    };

    let (output, trace) = run_cvar_rebalance(&data.quality, &data, &input).unwrap();
    assert_eq!(output.status, "optimal", "diagnostics: {:?}", output.diagnostics);

    for inv in &trace.invariants {
        assert!(inv.passed, "invariant failed: {} ({})", inv.name, inv.detail);
    }

    let weights_after = output.weights_after.expect("optimal solve has weights_after");
    assert!(
        weights_after["CRASH"] < 0.05,
        "CVaR minimization should cut the heavy-tail asset close to zero, got {}",
        weights_after["CRASH"]
    );
    assert!(
        (weights_after["SAFE"] - (1.0 - weights_after["CRASH"])).abs() < 1e-6,
        "weights should still sum to 1"
    );

    let stats_after = output.stats_after.unwrap();
    let stats_before = output.stats_before;
    assert!(
        stats_after.historical_cvar < stats_before.historical_cvar,
        "rebalanced CVaR ({}) should be lower than the starting CVaR ({})",
        stats_after.historical_cvar,
        stats_before.historical_cvar
    );
}

#[test]
fn zero_turnover_limit_returns_starting_weights_unchanged() {
    let (crash, safe) = crash_vs_safe_returns();
    let data = common::market_data_from_log_returns(&[("CRASH", crash), ("SAFE", safe)]);

    let input = CvarRebalanceInput {
        portfolio: Portfolio {
            holdings: vec![
                Holding {
                    ticker: "CRASH".to_string(),
                    weight: 0.5,
                },
                Holding {
                    ticker: "SAFE".to_string(),
                    weight: 0.5,
                },
            ],
            total_value_inr: 1_000_000.0,
        },
        confidence_level: 0.95,
        per_name_cap: 0.9, // w0 already satisfies the cap, so tau=0 is feasible
        turnover_limit: 0.0,
        commission_bps: 10.0,
        frequency: compute::model::Frequency::Daily,
        window: None,
    };

    let (output, _trace) = run_cvar_rebalance(&data.quality, &data, &input).unwrap();
    assert_eq!(output.status, "optimal", "diagnostics: {:?}", output.diagnostics);

    let weights_after = output.weights_after.unwrap();
    assert!((weights_after["CRASH"] - 0.5).abs() < 1e-6);
    assert!((weights_after["SAFE"] - 0.5).abs() < 1e-6);
    assert!((output.turnover.unwrap()).abs() < 1e-6);
}

#[test]
fn cap_too_tight_for_full_investment_is_reported_as_infeasible() {
    let (crash, safe) = crash_vs_safe_returns();
    let data = common::market_data_from_log_returns(&[("CRASH", crash), ("SAFE", safe)]);

    let input = CvarRebalanceInput {
        portfolio: Portfolio {
            holdings: vec![
                Holding {
                    ticker: "CRASH".to_string(),
                    weight: 0.5,
                },
                Holding {
                    ticker: "SAFE".to_string(),
                    weight: 0.5,
                },
            ],
            total_value_inr: 1_000_000.0,
        },
        confidence_level: 0.95,
        // 2 names * 0.2 cap = 0.4 < 1: can never reach full investment.
        per_name_cap: 0.2,
        turnover_limit: 1.5,
        commission_bps: 10.0,
        frequency: compute::model::Frequency::Daily,
        window: None,
    };

    let (output, trace) = run_cvar_rebalance(&data.quality, &data, &input).unwrap();
    assert_eq!(output.status, "infeasible");
    assert!(output.diagnostics.is_some());
    assert!(output.weights_after.is_none());
    assert!(trace.invariants.iter().any(|i| !i.passed));
}
