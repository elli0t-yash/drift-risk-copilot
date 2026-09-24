//! The LLM-facing agent layer: NL request -> `compute::Experiment` ->
//! `EvidenceTrace` -> grounded plain-language narration.

pub mod gemini;
pub mod grounding;
pub mod narrate;
pub mod parse;
pub mod pipeline;
pub mod schema;

pub use gemini::{GeminiClient, GeminiError, GeminiRequest, GeminiResponse, HttpGeminiClient};
pub use grounding::{grounded_narrate, GroundedNarration};
pub use narrate::{narrate, NarrateError};
pub use parse::{parse_experiment, ParseError};
pub use pipeline::{run, PipelineError, PipelineResult};
pub use schema::experiment_function_declarations;
