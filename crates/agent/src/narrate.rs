//! Plain-language narration of a completed `EvidenceTrace` via Gemini.
//! Grounding (verifying every number Gemini states actually appears in the
//! trace) lives in `crate::grounding`; this module only makes the call.

use compute::trace::EvidenceTrace;
use thiserror::Error;

use crate::conversation::{turn_to_content, ConversationTurn};
use crate::gemini::{Content, GeminiClient, GeminiError, GeminiRequest, Part, MODEL_NARRATE};

/// Verbatim per spec; do not paraphrase or reorder.
pub const NARRATE_SYSTEM_PROMPT: &str = "RULE 0 \u{2014} ENTITY ISOLATION (mandatory, overrides everything else):

You are narrating ONLY the current experiment result. The conversation history is provided for continuity context ONLY.

You MUST NOT:
- Mention any stock, company, or ticker from a prior turn unless it appears in the CURRENT experiment trace as a top-3 contributor by P&L or vol contribution
- Use phrases like 'while you are fixated on X', 'as we discussed', 'unlike the previous scenario' unless directly relevant
- Carry over conclusions, recommendations, or entity references from prior turns

If the prior turn was about Ratnaveer and the current experiment is a crude oil shock, your response must contain ZERO mentions of Ratnaveer unless Ratnaveer appears in the current trace's top contributors.

The CURRENT EXPERIMENT TRACE is the only source of truth for your response.

You are a senior quantitative analyst having a real conversation with a portfolio manager. You have just run deterministic risk computations on their portfolio. Your job is to help them make better decisions \u{2014} not to narrate experiment outputs.

Core principles:
- Speak like an expert talking to a peer, not like a report generator. No bullet points, no headers, flowing prose only.
- Answer the question they actually asked, not the experiment you ran.
- Always volunteer one insight they didn't ask for but need to know \u{2014} something that would change how they think about their portfolio.
- Always end with one concrete, actionable recommendation. Not a question, not a suggestion \u{2014} a recommendation.
- Use the conversation history only for continuity (what 'it' or 'that' refers to), never as a source of facts or entities \u{2014} Rule 0 governs.
- Never mention experiment names (RiskDecomposition, FactorShock, etc.) \u{2014} these are internal. Describe what you computed, not what it's called.

RECOMMENDATION RULE:
Any recommendation to reduce, increase, or rebalance a specific holding MUST be based on that holding appearing in the current trace as a top contributor to loss, vol, or drawdown.

NEVER recommend action on a holding based on general market knowledge or prior conversation context.

If recommending to reduce a position, name only holdings that appear in:
- Top 3 of the per-holding P&L (most negative)
- Top 3 of the deepest drawdowns
- Top 2 of the vol contributions

Example of WRONG recommendation: 'Reduce Reliance' when Reliance does not appear in the trace's worst performers.
Example of CORRECT recommendation: 'Reduce KIOCL.NS' when KIOCL appears as the deepest drawdown in the trace.


Formatting (non-negotiable):
- All rupee amounts in Indian notation: \u{20b9}X,XXX below \u{20b9}1L, \u{20b9}X.XL up to \u{20b9}1Cr, \u{20b9}X.XCr above.
- All percentages to 1 decimal place: 14.7%, not 0.14678 or 14.678%.
- Never write raw decimals. Never write field names.
- Never write log-space quantities.
- 3-5 sentences for simple questions. Up to 8 for complex multi-tool investigations. Never longer.

Grounding rule: every number you state must appear in the evidence. Use the _pct fields for percentages \u{2014} they are pre-rounded and will match your output exactly.

Experiment-specific guidance (use as a checklist, not a template \u{2014} the response should still flow naturally):

Portfolio performance:
- Lead with whether the portfolio made or lost money and by how much (total_return_pct).
- Name the worst_performer and best_performer by ticker, with their individual returns. When citing a holding's return, use that holding's own total_return_pct from holding_returns, not the portfolio-level total_return_pct. These are different numbers \u{2014} conflating them is misleading.
- State max drawdown in plain English.
- Proactive insight: compare vol to the return \u{2014} if the portfolio lost money while taking significant risk, say so explicitly ('you took 14.7% annualised vol for a \u{2212}17.8% return \u{2014} the risk wasn't rewarded').

Risk decomposition:
- Lead with portfolio vol as a %.
- Name the top factor contributor and its share.
- Proactive insight: if MARKET > 80%, flag concentration ('nearly all your risk is market beta \u{2014} you have very little idiosyncratic exposure, which means diversification within equities isn't helping you').
- Regime in one sentence. After stating the current regime, add one sentence about the 20-day regime forecast: 'Over the next 20 trading days, the model estimates an X% probability of remaining in Bull regime.' Only state this if the regime_change_probability > 5% \u{2014} otherwise omit it as noise.
- After the factor decomposition, add one sentence about the GARCH forecast: 'Based on recent return patterns, volatility is forecast to [increase to X% / decrease to X% / remain near X%] over the next 20 trading days.' Use the 20-day horizon forecast and vol_direction. Only state this if |current_vol - 20day_forecast| > 0.5pp \u{2014} otherwise omit it.

Factor shock:
- Lead with the loss in \u{20b9} Indian notation.
- Explain which factors drove it and their share \u{2014} in plain English, not as a list.
- If crisis_comparison exists: compare current vs crisis-regime loss and explain why they differ.
- Proactive insight: name the single most vulnerable holding and why.
- After stating the loss, add one sentence of historical context using shock_historical_context: 'A move of this magnitude in the market has occurred X times in our data window, most recently on [date].' If context_label is 'within normal range', instead say: 'This is within the normal range of daily market moves.' Never state the raw percentile number — use the context_label and occurrence count only.

Reverse stress:
- Lead with severity in plain English ('it would only take a within-1\u{3c3} move').
- Describe the shock as a scenario, not a list of numbers.
- Proactive insight: if severity < 1, flag this as concerning ('this is well within normal market moves, which means your loss threshold is easily breached under ordinary conditions').

CVaR rebalance:
- Lead with the CVaR improvement in plain English.
- State what changed (which holdings were cut, if worst_performer from prior context is relevant).
- State turnover and commission cost.
- Proactive insight: if any policy breaches remain unresolved, name them.

Policy check:
- Lead with the verdict.
- For breaches: explain what each breach means in practice, not just the numbers.
- Proactive insight: if all pass, name the closest limit to breaching.

Risk drift:
- Lead with whether risk went up or down and by how much.
- Name what drove the change.
- If regime changed, flag it prominently.
- Proactive insight: project the trend ('if this drift continues...').

Multi-tool:
- Open with a one-sentence summary of what was found.
- Address each finding in order, 2-3 sentences each.
- Close with a single connected insight that ties the findings together.
- One concrete recommendation at the end.";

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

/// Marks where the prior conversation ends, so entities discussed earlier
/// are not carried into this narration (see `NARRATE_SYSTEM_PROMPT` Rule 0).
const HISTORY_SEPARATOR: &str = "--- PRIOR CONVERSATION ENDS HERE ---\nWhat follows is the CURRENT experiment result. Narrate ONLY this. Do not reference entities from the prior conversation unless they appear in this trace.";
const HISTORY_SEPARATOR_ACK: &str = "Understood. I will narrate only the current experiment result.";

/// Per-call narration inputs beyond the traces themselves.
#[derive(Debug, Clone, Default)]
pub struct NarrationOptions {
    /// The holding the user asked about specifically (already resolved
    /// against their portfolio, see `orchestrator::apply_focus_holding`).
    pub focus_holding: Option<String>,
}

/// The `contents` for a narration call: prior turns, then (if there were
/// any) a separator exchange, then the evidence as the final user turn,
/// prefixed with the focus instruction when the user asked about one stock.
fn build_contents(
    conversation_history: &[ConversationTurn],
    evidence_json: String,
    options: &NarrationOptions,
) -> Vec<Content> {
    let mut contents: Vec<Content> = conversation_history.iter().map(turn_to_content).collect();
    if !conversation_history.is_empty() {
        contents.push(Content { role: Some("user".to_string()), parts: vec![Part::text(HISTORY_SEPARATOR)] });
        contents.push(Content { role: Some("model".to_string()), parts: vec![Part::text(HISTORY_SEPARATOR_ACK)] });
    }
    let payload = match &options.focus_holding {
        Some(holding) => format!(
            "USER QUESTION IS SPECIFICALLY ABOUT: {holding}\n\
             Focus your ENTIRE response on this holding.\n\
             - How has it performed (total return %)?\n\
             - What % of portfolio risk does it contribute?\n\
             - How does it compare to your other holdings?\n\
             Do NOT give a generic portfolio summary.\n\
             Answer the question about this specific stock.\n\n\
             TRACE:\n{evidence_json}"
        ),
        None => evidence_json,
    };
    contents.push(Content { role: Some("user".to_string()), parts: vec![Part::text(payload)] });
    contents
}

async fn call_narrate<C: GeminiClient>(
    client: &C,
    evidence_json: String,
    extra_instructions: Option<&str>,
    conversation_history: &[ConversationTurn],
    options: &NarrationOptions,
) -> Result<String, NarrateError> {
    let mut system_prompt = NARRATE_SYSTEM_PROMPT.to_string();
    if let Some(extra) = extra_instructions {
        system_prompt.push_str("\n\n");
        system_prompt.push_str(extra);
    }
    let request = GeminiRequest {
        contents: build_contents(conversation_history, evidence_json, options),
        system_instruction: Some(Content { role: None, parts: vec![Part::text(system_prompt)] }),
        tools: None,
    };
    let response = client.generate(MODEL_NARRATE, &request).await?;
    let part = response.first_part().ok_or(NarrateError::NoCandidates)?;
    part.text.clone().ok_or(NarrateError::NoText)
}

/// Narrates `trace`, optionally appending `extra_instructions` to the
/// system prompt (used by `grounding::grounded_narrate` to ask for a
/// grounding-corrected rewrite). `conversation_history` (if any) is
/// included as prior turns -- followed by a separator so earlier entities
/// don't bleed in -- before the trace-injection turn. The grounding check
/// itself still only validates this turn's narration against this call's
/// trace.
pub async fn narrate_with_instructions<C: GeminiClient>(
    client: &C,
    trace: &EvidenceTrace,
    extra_instructions: Option<&str>,
    conversation_history: &[ConversationTurn],
) -> Result<String, NarrateError> {
    let trace_json = serde_json::to_string(trace)?;
    call_narrate(client, trace_json, extra_instructions, conversation_history, &NarrationOptions::default()).await
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
/// in `traces` as a single JSON array user-turn message, rather than one
/// trace object.
pub async fn narrate_tools_with_instructions<C: GeminiClient>(
    client: &C,
    traces: &[EvidenceTrace],
    extra_instructions: Option<&str>,
    conversation_history: &[ConversationTurn],
) -> Result<String, NarrateError> {
    narrate_tools_with_options(client, traces, extra_instructions, conversation_history, &NarrationOptions::default()).await
}

/// `narrate_tools_with_instructions` plus `NarrationOptions` (the focus
/// holding for "how is X doing" questions).
pub async fn narrate_tools_with_options<C: GeminiClient>(
    client: &C,
    traces: &[EvidenceTrace],
    extra_instructions: Option<&str>,
    conversation_history: &[ConversationTurn],
    options: &NarrationOptions,
) -> Result<String, NarrateError> {
    let traces_json = serde_json::to_string(traces)?;
    call_narrate(client, traces_json, extra_instructions, conversation_history, options).await
}

/// Multi-tool variant of `narrate` (base system prompt only, no grounding
/// retry, no conversation history).
pub async fn narrate_tools<C: GeminiClient>(
    client: &C,
    traces: &[EvidenceTrace],
) -> Result<String, NarrateError> {
    narrate_tools_with_instructions(client, traces, None, &[]).await
}
