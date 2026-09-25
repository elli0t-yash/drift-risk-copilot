//! Plain-language narration of a completed `EvidenceTrace` via Gemini.
//! Grounding (verifying every number Gemini states actually appears in the
//! trace) lives in `crate::grounding`; this module only makes the call.

use compute::trace::EvidenceTrace;
use thiserror::Error;

use crate::conversation::{turn_to_content, ConversationTurn};
use crate::gemini::{Content, GeminiClient, GeminiError, GeminiRequest, Part, MODEL_NARRATE};

/// Verbatim per spec; do not paraphrase.
pub const NARRATE_SYSTEM_PROMPT: &str = "You are a portfolio risk analyst. Explain the following risk experiment result to an investment professional in 3\u{2013}5 sentences. Rules you must follow exactly:
1. Every number you state must appear verbatim in the evidence trace provided. Do not round, restate in different units, or derive new numbers.
2. For FactorShock: name every implied shock separately from given shocks, and note that implied moves are model-estimated from this portfolio's return history.
3. For RiskDecomposition: state the annualised portfolio vol first, then the top two risk contributors by factor, then specific risk.
4. For CvarRebalance: state before and after CVaR, turnover used, and commission cost. Do not describe factor attribution \u{2014} it is not applicable here.
5. For PortfolioPerformance: state total return and annualized return first, then annualized volatility and max drawdown. Do not describe factor attribution or a hypothetical shock \u{2014} this experiment reports realized historical performance only.
6. Do not use the phrase 'based on the evidence trace' or any meta-reference to the trace.";

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
