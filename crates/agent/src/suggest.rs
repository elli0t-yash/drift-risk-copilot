//! Proactive follow-up suggestion: after a completed experiment, asks
//! Gemini for one actionable follow-up question a risk manager would
//! naturally ask next. Plain text, no function calling, no grounding check
//! (a question isn't a factual claim to verify against the trace).

use thiserror::Error;

use crate::gemini::{GeminiClient, GeminiError, GeminiRequest};

/// Verbatim per spec; do not paraphrase.
pub const SUGGEST_SYSTEM_PROMPT: &str = "You are a portfolio risk analyst. Given this experiment result, generate exactly one follow-up question that a risk manager would naturally ask next. The question must be directly actionable as a follow-up experiment on the same portfolio. It must be one sentence, under 20 words, phrased as something the user would type. Return only the question, no preamble, no punctuation other than the question mark.";

#[derive(Debug, Error)]
pub enum SuggestError {
    #[error("gemini error: {0}")]
    Gemini(#[from] GeminiError),
    #[error("gemini response had no candidates")]
    NoCandidates,
    #[error("gemini response had no text part")]
    NoText,
}

/// Asks Gemini for one follow-up question given `narration` (the completed
/// experiment's grounded narration). Returns the response text as-is,
/// including an empty string if that's what Gemini returned.
pub async fn suggest_follow_up<C: GeminiClient>(
    client: &C,
    narration: &str,
) -> Result<String, SuggestError> {
    let request = GeminiRequest::user_turn(SUGGEST_SYSTEM_PROMPT, narration);
    let response = client.generate(&request).await?;
    let part = response.first_part().ok_or(SuggestError::NoCandidates)?;
    part.text.clone().ok_or(SuggestError::NoText)
}
