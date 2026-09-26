//! Plain-language narration of a completed `EvidenceTrace` via Gemini.
//! Grounding (verifying every number Gemini states actually appears in the
//! trace) lives in `crate::grounding`; this module only makes the call.

use compute::trace::EvidenceTrace;
use thiserror::Error;

use crate::conversation::{turn_to_content, ConversationTurn};
use crate::gemini::{Content, GeminiClient, GeminiError, GeminiRequest, Part, MODEL_NARRATE};

/// Verbatim per spec; do not paraphrase or reorder.
pub const NARRATE_SYSTEM_PROMPT: &str = "You are a portfolio risk analyst explaining results to an investment professional. You have access to deterministic model outputs. Your job is to translate numbers into meaning, not to list field values.

Formatting rules (apply to every response):
0. Express all monetary values in Indian notation:
   - Below \u{20b9}1L: '\u{20b9}X,XXX'
   - \u{20b9}1L to \u{20b9}1Cr: '\u{20b9}X.XL' (e.g. \u{20b9}37.8L)
   - Above \u{20b9}1Cr: '\u{20b9}X.XCr' (e.g. \u{20b9}1.2Cr)
   Never write 7-digit raw numbers. Never write more than 2 decimal places for rupee amounts.
   Express percentages to 1 decimal place (e.g. 15.5%, not 0.15497 or 15.497%).
   Never use field names from the trace (portfolio_log_pnl_inr, factor_attribution_log_inr, smoothed_probs, etc.).
   Never state log-space quantities \u{2014} these are internal model values, not user-facing results.

Experiment-specific rules:

1. For FactorShock:
   Lead with the portfolio outcome: 'Your portfolio would lose approximately \u{20b9}X' using Indian notation.
   Distinguish given shocks from model-estimated implied shocks in plain English \u{2014} one sentence each, no field names.
   Explain the loss by factor share in %, not raw INR attribution.
   Name the one or two factors that explain the majority of the loss. Do not list all five factors.
   If crisis_comparison is present and the current regime is not Crisis: add one sentence comparing the crisis-regime loss to the current-regime loss.

2. For RiskDecomposition:
   State annualised portfolio volatility first as a %.
   Name the top two risk contributors by factor with their share in %. Describe what this means in plain English (e.g. 'your portfolio moves almost entirely with the broader market').
   State specific risk share. If it is below 20%, note that diversification within the factor model is limited.
   State the current regime in one sentence.

3. For CvarRebalance:
   State before and after CVaR as % of portfolio.
   State turnover used and commission cost in \u{20b9}.
   If policy is present: state how many breaches were resolved and name any that remain \u{2014} one sentence.
   Do not describe factor attribution.

4. For ReverseStress:
   Lead with the severity label and what it means ('a within-1\u{3c3} event \u{2014} well within normal market moves').
   Describe the shock vector in plain English, not as a list of numbers (e.g. 'a Nifty fall of about 5%, accompanied by modest INR weakness').
   State the portfolio loss in Indian notation.
   Name the most vulnerable holdings \u{2014} one sentence.
   If linearisation_error_inr exceeds 5% of the threshold: add 'Note: the linear approximation may understate the true shock \u{2014} treat this as indicative.'

5. For PolicyCheck:
   Lead with a clear verdict: 'Your portfolio passes all X checks' or 'Your portfolio breaches X of Y checks.'
   For each breach: one sentence naming the rule, the actual value, and the limit \u{2014} plain English, no field names.
   If all pass: briefly name all checks in one sentence.

6. For RiskDrift:
   Lead with the direction and magnitude of vol change.
   Name the factor with the largest contribution increase.
   State whether the regime changed \u{2014} if regime_worsened is true, flag it explicitly.
   State the time elapsed between snapshots.
   Do not narrate every factor \u{2014} focus on the two or three most material changes.

7. For PortfolioPerformance:
   Lead with total return and whether it is positive or negative.
   State annualised return and vol side by side.
   State max drawdown \u{2014} put it in context ('the portfolio fell as much as X% from its peak').
   State current regime in one sentence.

8. For multi-tool results (when several experiments were run):
   Open with a one-sentence summary of what was investigated.
   Then address each tool result in the order it was run, using the single-experiment rules above but condensed to 2-3 sentences each.
   Close with one sentence connecting the findings (e.g. 'Together these suggest your tail risk is elevated and concentrated in market exposure').

Grounding rule (always applies):
   Every number you state must appear in the evidence provided.
   Do not round beyond what the trace shows. Do not derive new numbers.";

#[derive(Debug, Error)]
pub enum NarrateError {
    #[error("gemini error: {0}")]
    Gemini(#[from] GeminiError),
    #[error("failed to serialize the evidence trace: {0}")]
    Serialize(#[from] serde_json::Error),
    #[error("gemini response had no candidates")]
    NoCandidates,
    #[error("gemini response had no text part")]
    NoText,
}

/// Narrates `trace`, optionally appending `extra_instructions` to the
/// system prompt (used by `grounding::grounded_narrate` to ask for a
/// grounding-corrected rewrite). `conversation_history` (if any) is
/// included as prior turns before the trace-injection turn, so the
/// narration can refer back to earlier results ("compared to the previous
/// scenario..."); the grounding check itself still only validates this
/// turn's narration against this call's trace. The trace is injected as a
/// JSON user-turn message after the system prompt and any prior turns, per
/// spec.
pub async fn narrate_with_instructions<C: GeminiClient>(
    client: &C,
    trace: &EvidenceTrace,
    extra_instructions: Option<&str>,
    conversation_history: &[ConversationTurn],
) -> Result<String, NarrateError> {
    let mut system_prompt = NARRATE_SYSTEM_PROMPT.to_string();
    if let Some(extra) = extra_instructions {
        system_prompt.push_str("\n\n");
        system_prompt.push_str(extra);
    }
    let trace_json = serde_json::to_string(trace)?;

    let mut contents: Vec<Content> = conversation_history.iter().map(turn_to_content).collect();
    contents.push(Content {
        role: Some("user".to_string()),
        parts: vec![Part::text(trace_json)],
    });

    let request = GeminiRequest {
        contents,
        system_instruction: Some(Content {
            role: None,
            parts: vec![Part::text(system_prompt)],
        }),
        tools: None,
    };

    let response = client.generate(MODEL_NARRATE, &request).await?;
    let part = response.first_part().ok_or(NarrateError::NoCandidates)?;
    part.text.clone().ok_or(NarrateError::NoText)
}

/// Narrates `trace` with the base system prompt only (no grounding retry,
/// no conversation history).
pub async fn narrate<C: GeminiClient>(
    client: &C,
    trace: &EvidenceTrace,
) -> Result<String, NarrateError> {
    narrate_with_instructions(client, trace, None, &[]).await
}

/// Multi-tool variant of `narrate_with_instructions`: injects every trace
/// in `traces` as a single JSON array user-turn message (rule 8 in
/// `NARRATE_SYSTEM_PROMPT` covers this shape), rather than one trace
/// object. Used by the orchestrator whenever more than one tool ran for a
/// single `/ask` request; a one-trace slice produces the same prose a
/// direct `narrate_with_instructions` call would, since rule 8 only
/// changes behaviour when "several experiments were run".
pub async fn narrate_tools_with_instructions<C: GeminiClient>(
    client: &C,
    traces: &[EvidenceTrace],
    extra_instructions: Option<&str>,
    conversation_history: &[ConversationTurn],
) -> Result<String, NarrateError> {
    let mut system_prompt = NARRATE_SYSTEM_PROMPT.to_string();
    if let Some(extra) = extra_instructions {
        system_prompt.push_str("\n\n");
        system_prompt.push_str(extra);
    }
    let traces_json = serde_json::to_string(traces)?;

    let mut contents: Vec<Content> = conversation_history.iter().map(turn_to_content).collect();
    contents.push(Content {
        role: Some("user".to_string()),
        parts: vec![Part::text(traces_json)],
    });

    let request = GeminiRequest {
        contents,
        system_instruction: Some(Content {
            role: None,
            parts: vec![Part::text(system_prompt)],
        }),
        tools: None,
    };

    let response = client.generate(MODEL_NARRATE, &request).await?;
    let part = response.first_part().ok_or(NarrateError::NoCandidates)?;
    part.text.clone().ok_or(NarrateError::NoText)
}

/// Multi-tool variant of `narrate` (base system prompt only, no grounding
/// retry, no conversation history).
pub async fn narrate_tools<C: GeminiClient>(
    client: &C,
    traces: &[EvidenceTrace],
) -> Result<String, NarrateError> {
    narrate_tools_with_instructions(client, traces, None, &[]).await
}
