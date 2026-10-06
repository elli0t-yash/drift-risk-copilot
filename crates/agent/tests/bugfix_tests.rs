//! Tests for the routing / narration / grounding bug-fix batch: stock-vs-
//! macro routing, entity isolation, focus holding, directional and
//! recommendation grounding. The Gemini mock returns canned text, so
//! tests of *model* behaviour pin the prompt wording and the deterministic
//! checks around it; the live behaviour is covered by the deployed checks.

mod support;

use agent::gemini::GeminiRequest;
use agent::grounding::grounded_narrate_many_with;
use agent::narrate::{narrate_tools_with_options, NarrationOptions, NARRATE_SYSTEM_PROMPT};
use agent::orchestrator::{apply_focus_holding, plan_tools, resolve_focus_holding, OrchestratorError, PLANNING_SYSTEM_PROMPT as P};
use agent::ConversationTurn;
use compute::experiments::{Holding, Portfolio};
use compute::trace::EvidenceTrace;
use support::{text_response, MockGeminiClient};

fn portfolio() -> Portfolio {
    let h = |t: &str, w: f64| Holding { ticker: t.to_string(), weight: w };
    Portfolio {
        holdings: vec![
            h("RELIANCE.NS", 0.3),
            h("RATNAVEER.NS", 0.1),
            h("NSIL.BO", 0.1),
            h("ZYDUSWELL.NS", 0.1),
            h("GOLDCASE.NS", 0.2),
            h("KIOCL.NS", 0.1),
            h("BLS.NS", 0.05),
            h("DYCL.BO", 0.05),
        ],
        total_value_inr: 1_000_000.0,
    }
}

fn trace(experiment: &str, result: serde_json::Value) -> EvidenceTrace {
    let mut t: EvidenceTrace = serde_json::from_value(serde_json::json!({
        "experiment": experiment, "inputs": {"portfolio": portfolio()},
        "data_window": {"frequency": "Daily", "window_periods": 1, "start": "2025-01-01", "end": "2025-01-02"},
        "data_quality": {"date_range_start": "2025-01-01", "date_range_end": "2025-01-02", "trading_days": 2, "per_series": []},
        "model_params": {"frequency": "Daily", "window_periods": 1, "factor_names": [], "shrinkage_intensity": 0.0, "annualization_factor": 252.0, "regime_state": null, "regime_fallback_warnings": [], "cap_source": null},
        "outputs": {}, "invariants": [], "engine_version": "t"
    }))
    .unwrap();
    t.outputs = serde_json::json!({ "result": result });
    t
}

/// A shock trace where GOLDCASE, KIOCL and BLS lose the most and RELIANCE gains.
fn shock_trace() -> EvidenceTrace {
    trace(
        "FactorShock",
        serde_json::json!({"per_holding": [
            {"ticker": "GOLDCASE.NS", "pnl_inr": -9000.0},
            {"ticker": "KIOCL.NS", "pnl_inr": -5000.0},
            {"ticker": "BLS.NS", "pnl_inr": -4000.0},
            {"ticker": "DYCL.BO", "pnl_inr": -300.0},
            {"ticker": "RELIANCE.NS", "pnl_inr": 2000.0},
            {"ticker": "RATNAVEER.NS", "pnl_inr": 100.0},
        ]}),
    )
}

fn system_prompt(req: &GeminiRequest) -> String {
    req.system_instruction.as_ref().unwrap().parts[0].text.clone().unwrap()
}

fn content_text(req: &GeminiRequest, i: usize) -> String {
    req.contents[i].parts[0].text.clone().unwrap()
}

// ---- Bug 1 / 6: routing and scope --------------------------------------

#[tokio::test]
async fn a_declined_stock_price_question_surfaces_as_declined_with_the_redirect_text() {
    let redirect = "I cannot model individual stock price movements -- the system analyses macro factor risks. Which would be most useful?";
    for msg in ["What if Ratnaveer increases 10%?", "What if NSIL goes up 5%?"] {
        let client = MockGeminiClient::new(vec![text_response(format!(
            r#"[{{"tool": "decline", "params": {{}}, "reason": "{redirect}"}}]"#
        ))]);
        match plan_tools(&client, msg, &[]).await.unwrap_err() {
            OrchestratorError::Declined(text) => assert_eq!(text, redirect),
            other => panic!("expected Declined, got {other:?}"),
        }
    }
}

#[tokio::test]
async fn macro_questions_plan_a_tool_not_a_decline() {
    let client = MockGeminiClient::new(vec![text_response(
        r#"[{"tool": "factor_shock", "params": {"shocks_pct": {"MARKET": -10.0}}, "reason": "Nifty falls"}]"#,
    )]);
    let (plans, _) = plan_tools(&client, "What if Nifty falls 10%?", &[]).await.unwrap();
    assert_eq!(plans[0].tool, "factor_shock");
    assert_eq!(plans[0].params["shocks_pct"]["MARKET"], -10.0);
}

#[test]
fn the_planning_prompt_pins_rule_0_and_every_scope_boundary() {
    assert!(P.starts_with("You are a planning engine") || P.contains("ROUTING RULE 0"));
    // Rule 0 comes before the general tool rules.
    assert!(P.find("ROUTING RULE 0").unwrap() < P.find("Rules for tool selection:").unwrap());
    for example in ["Ratnaveer", "NSIL", "NEVER route to factor_shock or historical_stress"] {
        assert!(P.contains(example), "{example}");
    }
    assert!(P.contains("I cannot model individual stock price movements"));
    assert!(P.contains("Which would be most useful?"));
    // Historical scenarios only when the *current* message names one.
    assert!(P.contains("NEVER use historical_stress unless the CURRENT user message explicitly names"));
    assert!(P.contains("crude oil after an IL&FS question is a factor_shock"));
    // Out-of-scope messages.
    assert!(P.contains("Screener.in or Tickertape"));
    assert!(P.contains("I cannot forecast stock prices"));
    assert!(P.contains("I don't have real-time news"));
    assert!(P.contains("I can only help with portfolio risk analysis."));
    // Macro stays in scope.
    assert!(P.contains("'market crash', 'Nifty falls', 'bear market' -> factor_shock MARKET: -20.0"));
    assert!(P.contains("'Nifty falls 15%' -> MARKET: -15.0"));
}

#[tokio::test]
async fn an_explicit_historical_event_still_plans_historical_stress() {
    let client = MockGeminiClient::new(vec![text_response(
        r#"[{"tool": "historical_stress", "params": {"scenario_id": "ilfs_contagion"}, "reason": "IL&FS named"}]"#,
    )]);
    let (plans, _) = plan_tools(&client, "What is the IL&FS impact?", &[]).await.unwrap();
    assert_eq!(plans[0].params["scenario_id"], "ilfs_contagion");
}

#[tokio::test]
async fn the_planner_sees_prior_turns_but_a_crude_followup_plans_a_factor_shock() {
    let client = MockGeminiClient::new(vec![text_response(
        r#"[{"tool": "factor_shock", "params": {"shocks_pct": {"BRENT": 20.0}}, "reason": "crude spike"}]"#,
    )]);
    let history = vec![
        ConversationTurn::user("What would the IL&FS scenario do to my portfolio?"),
        ConversationTurn::assistant("The IL&FS scenario would cost you about 12%."),
    ];
    let (plans, _) = plan_tools(&client, "What if crude spikes?", &history).await.unwrap();
    assert_eq!(plans[0].tool, "factor_shock");
    assert_eq!(plans[0].params["shocks_pct"]["BRENT"], 20.0);
    // History is passed to the planner, followed by the current message.
    let req = client.last_request();
    assert_eq!(req.contents.len(), 3);
    assert_eq!(content_text(&req, 2), "What if crude spikes?");
}

// ---- Bug 3: focus holding ----------------------------------------------

#[test]
fn focus_holding_resolves_to_a_real_holding_even_from_a_slightly_wrong_symbol() {
    let p = portfolio();
    assert_eq!(resolve_focus_holding("ZYDUSWELL.NS", &p).as_deref(), Some("ZYDUSWELL.NS"));
    assert_eq!(resolve_focus_holding("zyduswell", &p).as_deref(), Some("ZYDUSWELL.NS"));
    // The planner's guess had a stray P; the holding is RATNAVEER.NS.
    assert_eq!(resolve_focus_holding("RATNAVEERP.NS", &p).as_deref(), Some("RATNAVEER.NS"));
    // A holding listed only on BSE still resolves from an .NS guess.
    assert_eq!(resolve_focus_holding("NSIL.NS", &p).as_deref(), Some("NSIL.BO"));
    assert_eq!(resolve_focus_holding("TCS.NS", &p), None);
}

#[tokio::test]
async fn how_is_zydus_doing_plans_portfolio_performance_with_the_focus_holding_pinned() {
    let client = MockGeminiClient::new(vec![text_response(
        r#"[{"tool": "portfolio_performance", "params": {"focus_holding": "ZYDUSWELL.NS"}, "reason": "specific stock"}]"#,
    )]);
    let (mut plans, _) = plan_tools(&client, "How is Zydus Wellness doing in my portfolio?", &[]).await.unwrap();
    assert_eq!(plans[0].tool, "portfolio_performance");
    assert_eq!(apply_focus_holding(&mut plans, &portfolio()).as_deref(), Some("ZYDUSWELL.NS"));
    assert_eq!(plans[0].params["focus_holding"], "ZYDUSWELL.NS");

    // An unknown stock is dropped rather than narrated about.
    let mut plans = vec![agent::ToolPlan {
        tool: "portfolio_performance".into(),
        params: serde_json::json!({"focus_holding": "TCS.NS"}),
        reason: String::new(),
    }];
    assert_eq!(apply_focus_holding(&mut plans, &portfolio()), None);
    assert!(plans[0].params.get("focus_holding").is_none());
}

#[tokio::test]
async fn the_focus_holding_is_prepended_to_the_trace_sent_for_narration() {
    let client = MockGeminiClient::new(vec![text_response("Zyduswell is up.")]);
    let t = trace("PortfolioPerformance", serde_json::json!({"holding_returns": {}}));
    let opts = NarrationOptions { focus_holding: Some("ZYDUSWELL.NS".to_string()) };
    narrate_tools_with_options(&client, &[t], None, &[], &opts).await.unwrap();
    let req = client.last_request();
    let payload = content_text(&req, req.contents.len() - 1);
    assert!(payload.starts_with("USER QUESTION IS SPECIFICALLY ABOUT: ZYDUSWELL.NS\n"));
    assert!(payload.contains("Do NOT give a generic portfolio summary."));
    assert!(payload.contains("TRACE:\n["));

    // Without a focus the payload is just the trace JSON.
    let client = MockGeminiClient::new(vec![text_response("ok")]);
    narrate_tools_with_options(&client, &[shock_trace()], None, &[], &NarrationOptions::default()).await.unwrap();
    let req = client.last_request();
    assert!(content_text(&req, req.contents.len() - 1).starts_with('['));
}

// ---- Bug 2: entity isolation -------------------------------------------

#[tokio::test]
async fn prior_turns_are_followed_by_a_separator_exchange_before_the_current_trace() {
    let client = MockGeminiClient::new(vec![text_response("ok")]);
    let history = vec![
        ConversationTurn::user("How is Ratnaveer doing?"),
        ConversationTurn::assistant("Ratnaveer is down."),
    ];
    narrate_tools_with_options(&client, &[shock_trace()], None, &history, &NarrationOptions::default())
        .await
        .unwrap();
    let req = client.last_request();
    let roles: Vec<&str> = req.contents.iter().map(|c| c.role.as_deref().unwrap()).collect();
    assert_eq!(roles, vec!["user", "model", "user", "model", "user"]);
    assert!(content_text(&req, 2).starts_with("--- PRIOR CONVERSATION ENDS HERE ---"));
    assert!(content_text(&req, 2).contains("Narrate ONLY this."));
    assert_eq!(content_text(&req, 3), "Understood. I will narrate only the current experiment result.");
    assert!(content_text(&req, 4).starts_with('['), "the trace comes last");
}

#[tokio::test]
async fn no_history_means_no_separator() {
    let client = MockGeminiClient::new(vec![text_response("ok")]);
    narrate_tools_with_options(&client, &[shock_trace()], None, &[], &NarrationOptions::default()).await.unwrap();
    assert_eq!(client.last_request().contents.len(), 1);
}

#[test]
fn the_narration_prompt_opens_with_rule_0_and_carries_the_recommendation_rule() {
    assert!(NARRATE_SYSTEM_PROMPT.starts_with("RULE 0 \u{2014} ENTITY ISOLATION"));
    assert!(NARRATE_SYSTEM_PROMPT.contains("ZERO mentions of Ratnaveer"));
    assert!(NARRATE_SYSTEM_PROMPT.contains("The CURRENT EXPERIMENT TRACE is the only source of truth"));
    assert!(NARRATE_SYSTEM_PROMPT.contains("RECOMMENDATION RULE:"));
    assert!(NARRATE_SYSTEM_PROMPT.contains("Reduce KIOCL.NS"));
    // The old instruction to reference prior findings would contradict Rule 0.
    assert!(!NARRATE_SYSTEM_PROMPT.contains("Reference prior findings naturally"));
}

#[tokio::test]
async fn a_narration_that_drags_in_a_prior_turns_stock_is_retried_and_cleaned() {
    let history = vec![
        ConversationTurn::user("How is Ratnaveer doing?"),
        ConversationTurn::assistant("Ratnaveer has fallen sharply."),
    ];
    let client = MockGeminiClient::new(vec![
        text_response("A crude spike hurts KIOCL most, while you are fixated on Ratnaveer."),
        text_response("A crude spike hurts KIOCL most."),
    ]);
    let (n, retries) =
        grounded_narrate_many_with(&client, &[shock_trace()], &history, &NarrationOptions::default()).await.unwrap();
    assert_eq!(retries, 1);
    assert!(!n.narration.to_lowercase().contains("ratnaveer"));
    assert!(n.grounding_warnings.is_empty());
    assert!(system_prompt(&client.last_request()).contains("CRITICAL CORRECTION REQUIRED"));
    assert!(system_prompt(&client.last_request()).contains("RATNAVEER.NS was discussed in an earlier turn"));
}

#[tokio::test]
async fn a_prior_turn_stock_that_is_a_current_top_mover_may_be_mentioned() {
    let history = vec![ConversationTurn::user("How is KIOCL doing?"), ConversationTurn::assistant("KIOCL is weak.")];
    let client = MockGeminiClient::new(vec![text_response("KIOCL is again one of the biggest losers here.")]);
    let (n, retries) =
        grounded_narrate_many_with(&client, &[shock_trace()], &history, &NarrationOptions::default()).await.unwrap();
    assert_eq!(retries, 0);
    assert!(n.directional_checks.is_empty());
}

#[tokio::test]
async fn the_stock_the_user_asks_about_now_is_always_allowed() {
    let history = vec![ConversationTurn::user("How is Ratnaveer doing?"), ConversationTurn::assistant("Ratnaveer is down.")];
    let client = MockGeminiClient::new(vec![text_response("Ratnaveer is barely moving.")]);
    let opts = NarrationOptions { focus_holding: Some("RATNAVEER.NS".to_string()) };
    let (_, retries) = grounded_narrate_many_with(&client, &[shock_trace()], &history, &opts).await.unwrap();
    assert_eq!(retries, 0);
}

// ---- Bug 4: directional grounding --------------------------------------

#[tokio::test]
async fn a_wrong_direction_triggers_a_retry_and_the_corrected_narration_is_accepted() {
    let client = MockGeminiClient::new(vec![
        text_response("Gold is the primary stabiliser in this selloff."),
        text_response("GOLDCASE.NS was the largest drag in this selloff."),
    ]);
    let (n, retries) =
        grounded_narrate_many_with(&client, &[shock_trace()], &[], &NarrationOptions::default()).await.unwrap();
    assert_eq!(retries, 1);
    assert_eq!(n.narration, "GOLDCASE.NS was the largest drag in this selloff.");
    assert!(n.grounding_warnings.is_empty());
    assert!(n.directional_checks.is_empty() && n.directional_warnings.is_empty());
    let retry_prompt = system_prompt(&client.last_request());
    assert!(retry_prompt.contains("CRITICAL CORRECTION REQUIRED"));
    assert!(retry_prompt.contains("described positively but the trace shows it as a negative contributor"));
}

#[tokio::test]
async fn an_uncorrected_wrong_direction_is_reported_in_grounding_warnings() {
    let client = MockGeminiClient::new(vec![
        text_response("Gold is the primary stabiliser."),
        text_response("Gold remains a stabiliser here."),
        text_response("Gold is still the stabiliser."),
    ]);
    let (n, retries) =
        grounded_narrate_many_with(&client, &[shock_trace()], &[], &NarrationOptions::default()).await.unwrap();
    assert_eq!(retries, 2);
    assert_eq!(n.directional_checks.len(), 1);
    assert_eq!(n.directional_warnings.len(), 1);
    assert!(n.grounding_warnings.iter().any(|w| w.contains("described positively")));
}

#[tokio::test]
async fn consistent_direction_does_not_retry() {
    let client = MockGeminiClient::new(vec![text_response("Reliance gained while GOLDCASE.NS dragged the portfolio down.")]);
    let (_, retries) =
        grounded_narrate_many_with(&client, &[shock_trace()], &[], &NarrationOptions::default()).await.unwrap();
    assert_eq!(retries, 0);
    assert_eq!(client.call_count(), 1);
}

// ---- Bug 5: recommendations --------------------------------------------

#[tokio::test]
async fn a_recommendation_for_a_holding_the_trace_does_not_single_out_is_corrected() {
    let perf = trace(
        "PortfolioPerformance",
        serde_json::json!({"holding_returns": {
            "KIOCL.NS": {"total_return_pct": -40.0, "max_drawdown_pct": -55.0},
            "DYCL.BO": {"total_return_pct": -30.0, "max_drawdown_pct": -45.0},
            "BLS.NS": {"total_return_pct": -20.0, "max_drawdown_pct": -35.0},
            "ELECON.BO": {"total_return_pct": -10.0, "max_drawdown_pct": -25.0},
            "RELIANCE.NS": {"total_return_pct": 8.0, "max_drawdown_pct": -5.0},
        }}),
    );
    let client = MockGeminiClient::new(vec![
        text_response("Consider reducing Reliance because of its high beta."),
        text_response("Consider reducing KIOCL.NS, which has the deepest drawdown."),
    ]);
    let (n, retries) = grounded_narrate_many_with(&client, &[perf], &[], &NarrationOptions::default()).await.unwrap();
    assert_eq!(retries, 1);
    assert!(n.narration.contains("KIOCL.NS") && !n.narration.contains("Reliance"));
    assert!(system_prompt(&client.last_request()).contains("not among the trace's worst contributors"));
}

// ---- AskResponse / pipeline: a decline runs nothing ---------------------

#[tokio::test]
async fn the_pipeline_surfaces_a_decline_before_running_any_experiment() {
    let client = MockGeminiClient::new(vec![text_response(
        r#"[{"tool": "decline", "params": {}, "reason": "I cannot forecast stock prices."}]"#,
    )]);
    let ctx = compute::context::ExperimentContext {
        store: std::sync::Arc::new(store::SnapshotStore::open(":memory:").unwrap()),
        portfolio_hash: String::new(),
        policy: None,
    };
    let err = match agent::pipeline::run(&client, "Will Reliance reach 3000?", portfolio(), &[], &ctx).await {
        Err(e) => e,
        Ok(_) => panic!("a declined question must not produce a result"),
    };
    match err {
        agent::PipelineError::Orchestrator(OrchestratorError::Declined(t)) => assert_eq!(t, "I cannot forecast stock prices."),
        other => panic!("unexpected {other:?}"),
    }
    assert_eq!(client.call_count(), 1, "only the planning call ran");
}
