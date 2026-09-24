//! Plain-language narration of a completed `EvidenceTrace` via Gemini.
//! Grounding (verifying every number Gemini states actually appears in the
//! trace) lives in `crate::grounding`; this module only makes the call.

use compute::trace::EvidenceTrace;
use thiserror::Error;

use crate::gemini::{GeminiClient, GeminiError, GeminiRequest};

/// Verbatim per spec; do not paraphrase.
pub const NARRATE_SYSTEM_PROMPT: &str = "You are a portfolio risk analyst. Explain the following risk experiment result to an investment professional in 3\u{2013}5 sentences. Rules you must follow exactly:
1. Every number you state must appear verbatim in the evidence trace provided. Do not round, restate in different units, or derive new numbers.
2. For FactorShock: name every implied shock separately from given shocks, and note that implied moves are model-estimated from this portfolio's return history.
3. For RiskDecomposition: state the annualised portfolio vol first, then the top two risk contributors by factor, then specific risk.
4. For CvarRebalance: state before and after CVaR, turnover used, and commission cost. Do not describe factor attribution \u{2014} it is not applicable here.
5. Do not use the phrase 'based on the evidence trace' or any meta-reference to the trace.";

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
/// grounding-corrected rewrite). The trace is injected as a JSON user-turn
/// message after the system prompt, per spec.
pub async fn narrate_with_instructions<C: GeminiClient>(
    client: &C,
    trace: &EvidenceTrace,
    extra_instructions: Option<&str>,
) -> Result<String, NarrateError> {
    let mut system_prompt = NARRATE_SYSTEM_PROMPT.to_string();
    if let Some(extra) = extra_instructions {
        system_prompt.push_str("\n\n");
        system_prompt.push_str(extra);
    }
    let trace_json = serde_json::to_string(trace)?;
    let request = GeminiRequest::user_turn(system_prompt, trace_json);

    let response = client.generate(&request).await?;
    let part = response.first_part().ok_or(NarrateError::NoCandidates)?;
    part.text.clone().ok_or(NarrateError::NoText)
}

/// Narrates `trace` with the base system prompt only (no grounding retry).
pub async fn narrate<C: GeminiClient>(
    client: &C,
    trace: &EvidenceTrace,
) -> Result<String, NarrateError> {
    narrate_with_instructions(client, trace, None).await
}
