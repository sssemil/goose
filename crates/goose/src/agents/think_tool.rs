//! Think tool handler for the goose agent.
//!
//! A no-op scratchpad: the model passes a thought, we log it, and we return
//! a trivial acknowledgment. No state changes, no external calls. Following
//! Anthropic's "think tool" recommendation, system-prompt guidance lives in
//! `prompts/system.md`.

use rmcp::model::{Content, ErrorCode, ErrorData};

use super::Agent;
use crate::mcp_utils::ToolResult;

impl Agent {
    pub async fn handle_think(
        &self,
        arguments: serde_json::Value,
    ) -> ToolResult<Vec<Content>> {
        let thought = arguments
            .get("thought")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                ErrorData::new(
                    ErrorCode::INVALID_PARAMS,
                    "Missing 'thought' parameter".to_string(),
                    None,
                )
            })?;

        if thought.trim().is_empty() {
            return Err(ErrorData::new(
                ErrorCode::INVALID_PARAMS,
                "'thought' must be a non-empty string".to_string(),
                None,
            ));
        }

        tracing::info!(target: "goose::think", thought = %thought, "agent thought");

        Ok(vec![Content::text(
            "Thought logged. This was internal reasoning only — now continue and produce the user-facing answer or take the next concrete action.",
        )])
    }
}

#[cfg(test)]
mod tests {
    use crate::agents::platform_tools::{think_tool, PLATFORM_THINK_TOOL_NAME};

    #[test]
    fn think_tool_schema_matches_spec() {
        let tool = think_tool();
        assert_eq!(tool.name, PLATFORM_THINK_TOOL_NAME);
        let schema = serde_json::to_value(&tool.input_schema).unwrap();
        assert_eq!(
            schema["required"],
            serde_json::json!(["thought"]),
            "schema must require `thought`"
        );
        assert_eq!(
            schema["properties"]["thought"]["type"],
            serde_json::json!("string"),
            "`thought` must be a string"
        );
    }
}
