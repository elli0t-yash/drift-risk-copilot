//! Produces the `run_experiment` function declaration Gemini uses for
//! structured extraction, derived from `compute::experiments::Experiment`'s
//! `schemars` JSON Schema.

use compute::experiments::Experiment;

use crate::gemini::FunctionDeclaration;

const FUNCTION_NAME: &str = "run_experiment";
const FUNCTION_DESCRIPTION: &str = "Parse the user's request into a structured experiment. \
    Portfolio holdings and weights are provided separately; only extract the experiment type \
    and its parameters from the user's text.";

/// Fields the caller supplies out-of-band (portfolio holdings/weights) and
/// that are therefore stripped from the schema shown to Gemini, so a small
/// model isn't nudged into inventing a `portfolio` object from prose that
/// never mentions tickers or weights. `parse::parse_experiment` always
/// overwrites this field with the caller's real portfolio before
/// deserializing the function-call args into an `Experiment`, regardless of
/// whether Gemini included it.
const CALLER_SUPPLIED_FIELDS: &[&str] = &["portfolio"];

/// The JSON Schema for `Experiment`, with caller-supplied fields removed
/// from every variant's `properties`/`required`.
pub fn experiment_json_schema() -> serde_json::Value {
    let schema = schemars::schema_for!(Experiment);
    let mut value = serde_json::to_value(schema).expect("schemars output is always valid JSON");
    strip_fields(&mut value, CALLER_SUPPLIED_FIELDS);
    value
}

/// Recursively removes `fields` from every `properties` map and `required`
/// array found anywhere in a JSON Schema value (including inside
/// `definitions`/`$defs`, reached by the generic recursion below).
fn strip_fields(value: &mut serde_json::Value, fields: &[&str]) {
    match value {
        serde_json::Value::Object(map) => {
            if let Some(serde_json::Value::Object(properties)) = map.get_mut("properties") {
                for field in fields {
                    properties.remove(*field);
                }
            }
            if let Some(serde_json::Value::Array(required)) = map.get_mut("required") {
                required.retain(|v| !v.as_str().is_some_and(|s| fields.contains(&s)));
            }
            for child in map.values_mut() {
                strip_fields(child, fields);
            }
        }
        serde_json::Value::Array(items) => {
            for item in items {
                strip_fields(item, fields);
            }
        }
        _ => {}
    }
}

/// The single function declaration passed to Gemini for NL -> Experiment
/// parsing.
pub fn experiment_function_declaration() -> FunctionDeclaration {
    FunctionDeclaration {
        name: FUNCTION_NAME.to_string(),
        description: FUNCTION_DESCRIPTION.to_string(),
        parameters: experiment_json_schema(),
    }
}
