//! Produces the `run_experiment` function declaration Gemini uses for
//! structured extraction, derived from `compute::experiments::Experiment`'s
//! `schemars` JSON Schema.

use std::collections::HashMap;

use compute::experiments::Experiment;
use serde_json::Value;

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
/// from every variant's `properties`/`required`, and reshaped to fit
/// Gemini's function-calling schema subset (see `sanitize_for_gemini`).
pub fn experiment_json_schema() -> serde_json::Value {
    let schema = schemars::schema_for!(Experiment);
    let mut value = serde_json::to_value(schema).expect("schemars output is always valid JSON");
    strip_fields(&mut value, CALLER_SUPPLIED_FIELDS);
    sanitize_for_gemini(value)
}

/// Reshapes a `schemars`-generated JSON Schema into what Gemini's
/// `generateContent` function-calling API actually accepts. Verified
/// against a live deployment: without this, Gemini rejects the schema
/// outright (400 `INVALID_ARGUMENT`) for using `$schema`/`definitions`/
/// `$ref` (draft-07 features Gemini's schema proto doesn't have),
/// `additionalProperties` (used by schemars for `BTreeMap<String, f64>`
/// fields like `shocks_pct` — not supported; degrades to an unconstrained
/// `object`, which is fine here since the description already says what
/// keys are valid), and `"type": ["integer", "null"]` for `Option<T>`
/// fields (Gemini's proto field is a single enum value, not a list — the
/// `null` branch is dropped, relying on `required` to encode optionality
/// instead of an explicit nullable type).
fn sanitize_for_gemini(mut schema: Value) -> Value {
    let definitions: HashMap<String, Value> = match &mut schema {
        Value::Object(map) => {
            let defs = map.remove("definitions").or_else(|| map.remove("$defs"));
            map.remove("$schema");
            match defs {
                Some(Value::Object(defs_map)) => defs_map.into_iter().collect(),
                _ => HashMap::new(),
            }
        }
        _ => HashMap::new(),
    };
    resolve_and_clean(&mut schema, &definitions);
    schema
}

fn resolve_and_clean(value: &mut Value, defs: &HashMap<String, Value>) {
    match value {
        Value::Object(map) => {
            // Inline `$ref` pointers into `definitions`/`$defs` (Gemini has
            // no concept of a schema reference); replaces this node
            // entirely, so nothing else in this branch applies afterward.
            if let Some(Value::String(r)) = map.get("$ref") {
                if let Some(name) = r
                    .strip_prefix("#/definitions/")
                    .or_else(|| r.strip_prefix("#/$defs/"))
                {
                    if let Some(resolved) = defs.get(name) {
                        let mut resolved = resolved.clone();
                        resolve_and_clean(&mut resolved, defs);
                        *value = resolved;
                        return;
                    }
                }
            }

            // Collapse `allOf: [single_schema]` (schemars' encoding of a
            // `$ref` with sibling keywords, e.g. `{"allOf": [{"$ref": ...}],
            // "default": ...}`) by merging the resolved schema's keys in
            // alongside the siblings already present.
            if let Some(Value::Array(items)) = map.remove("allOf") {
                for mut item in items {
                    resolve_and_clean(&mut item, defs);
                    if let Value::Object(inner) = item {
                        for (k, v) in inner {
                            map.entry(k).or_insert(v);
                        }
                    }
                }
            }

            // Gemini's schema proto has no open-map equivalent; drop it and
            // fall back to an unconstrained `object` (the field's
            // `description` carries the "what goes in here" guidance).
            map.remove("additionalProperties");

            // `Option<T>` fields serialize as `"type": ["T", "null"]`;
            // Gemini's `type` is a single scalar. Optionality is still
            // conveyed via `required` (or its absence), so just drop the
            // `null` branch.
            if let Some(Value::Array(types)) = map.get("type").cloned() {
                match types.into_iter().find(|t| t.as_str() != Some("null")) {
                    Some(t) => {
                        map.insert("type".to_string(), t);
                    }
                    None => {
                        map.remove("type");
                    }
                }
            }

            for child in map.values_mut() {
                resolve_and_clean(child, defs);
            }
        }
        Value::Array(items) => {
            for item in items {
                resolve_and_clean(item, defs);
            }
        }
        _ => {}
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression test for a live-caught bug: Gemini's `generateContent`
    /// rejected the unsanitized schemars output with a 400 (`$schema`,
    /// `definitions`, `$ref`, `additionalProperties` unrecognized; `type`
    /// as a list not accepted). Asserts none of those survive anywhere in
    /// the schema actually sent to Gemini.
    #[test]
    fn sanitized_schema_has_no_gemini_incompatible_keywords() {
        let schema = experiment_json_schema();
        assert_no_incompatible_keywords(&schema);
    }

    fn assert_no_incompatible_keywords(value: &Value) {
        match value {
            Value::Object(map) => {
                assert!(!map.contains_key("$schema"), "found $schema: {value}");
                assert!(!map.contains_key("$ref"), "found $ref: {value}");
                assert!(!map.contains_key("definitions"), "found definitions: {value}");
                assert!(!map.contains_key("$defs"), "found $defs: {value}");
                assert!(
                    !map.contains_key("additionalProperties"),
                    "found additionalProperties: {value}"
                );
                assert!(!map.contains_key("allOf"), "found allOf: {value}");
                if let Some(t) = map.get("type") {
                    assert!(!t.is_array(), "found array-valued \"type\": {value}");
                }
                for child in map.values() {
                    assert_no_incompatible_keywords(child);
                }
            }
            Value::Array(items) => {
                for item in items {
                    assert_no_incompatible_keywords(item);
                }
            }
            _ => {}
        }
    }
}
