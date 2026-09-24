//! Natural-language -> `Experiment` extraction via a single Gemini
//! function-calling turn.

use compute::experiments::{Experiment, Portfolio};
use thiserror::Error;

use crate::gemini::{Content, GeminiClient, GeminiError, GeminiRequest, Part, Tool};
use crate::schema::experiment_function_declaration;

/// Verbatim per spec; do not paraphrase.
pub const PARSE_SYSTEM_PROMPT: &str = "You are a parameter extraction engine. Your only job is \
to call run_experiment with the correct experiment type and parameters extracted from the \
user's message. Do not add explanation. Do not ask clarifying questions. If the user's intent \
clearly maps to one of the three experiment types, call the function. If it does not, return a \
text response with one sentence explaining what you cannot extract.";

#[derive(Debug, Error)]
pub enum ParseError {
    #[error("gemini error: {0}")]
    Gemini(#[from] GeminiError),
    /// Gemini responded with text instead of calling `run_experiment`: the
    /// user's message didn't clearly map to one of the three experiments.
    #[error("could not extract an experiment from the message: {0}")]
    Unrecognised(String),
    #[error("gemini returned no candidates")]
    NoCandidates,
    #[error("gemini returned neither a function call nor text")]
    EmptyResponse,
    #[error("gemini called an unexpected function: {0}")]
    UnexpectedFunction(String),
    #[error("failed to deserialize run_experiment args into an Experiment: {0}")]
    InvalidArgs(#[from] serde_json::Error),
}

/// Sends `user_message` to Gemini with the `run_experiment` function
/// declaration. On a function-call response, deserializes the args into an
/// `Experiment` and overwrites its `portfolio` field with the caller's
/// `portfolio` (Gemini is never given portfolio data — see
/// `schema::CALLER_SUPPLIED_FIELDS` — so this is always the source of
/// truth, whether or not Gemini's args happened to include one). On a text
/// response, returns `ParseError::Unrecognised`.
pub async fn parse_experiment<C: GeminiClient>(
    client: &C,
    user_message: &str,
    portfolio: Portfolio,
) -> Result<Experiment, ParseError> {
    let request = GeminiRequest {
        contents: vec![Content {
            role: Some("user".to_string()),
            parts: vec![Part::text(user_message)],
        }],
        system_instruction: Some(Content {
            role: None,
            parts: vec![Part::text(PARSE_SYSTEM_PROMPT)],
        }),
        tools: Some(vec![Tool {
            function_declarations: vec![experiment_function_declaration()],
        }]),
    };

    let response = client.generate(&request).await?;
    let candidate = response.candidates.first().ok_or(ParseError::NoCandidates)?;

    for part in &candidate.content.parts {
        if let Some(call) = &part.function_call {
            if call.name != "run_experiment" {
                return Err(ParseError::UnexpectedFunction(call.name.clone()));
            }
            let mut args = call.args.clone();
            if let serde_json::Value::Object(ref mut map) = args {
                map.insert("portfolio".to_string(), serde_json::to_value(&portfolio)?);
            }
            let experiment: Experiment = serde_json::from_value(args)?;
            return Ok(experiment);
        }
        if let Some(text) = &part.text {
            return Err(ParseError::Unrecognised(text.clone()));
        }
    }

    Err(ParseError::EmptyResponse)
}
