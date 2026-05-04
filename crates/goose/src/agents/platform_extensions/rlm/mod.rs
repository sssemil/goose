//! `rlm` platform extension — exposes Recursive Language Model primitives
//! as MCP tools backed by a per-session [`RlmStore`].
//!
//! Tools (all unprefixed → `rlm__*`):
//! - `rlm__list_contexts`
//! - `rlm__search`
//! - `rlm__get_chunk`
//! - `rlm__sub_query`
//! - `rlm__batch_sub_query`
//! - `rlm__store`
//! - `rlm__retrieve`
//! - `rlm__list_keys`
//!
//! Termination is handled via the existing `final_output` pathway, so no
//! `rlm__finalize` is needed — the model just returns its final text.

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use indoc::indoc;
use rmcp::model::{
    CallToolResult, Content, Implementation, InitializeResult, JsonObject, ListToolsResult,
    ServerCapabilities, Tool, ToolAnnotations,
};
use schemars::{schema_for, JsonSchema};
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

use crate::agents::extension::PlatformExtensionContext;
use crate::agents::mcp_client::{Error, McpClientTrait};
use crate::agents::rlm::{ChunkData, RlmStore, SearchHit, SearchMode};
use crate::agents::tool_execution::ToolCallContext;
use crate::conversation::message::Message;
use crate::providers::base::Provider;

pub static EXTENSION_NAME: &str = "rlm";

const TOOL_RESPONSE_PREVIEW_CHARS: usize = 2_000;
const SUB_QUERY_MAX_INLINE_CHARS: usize = 12_000;
const BATCH_CONCURRENCY: usize = 8;

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct SearchParams {
    /// Name of the loaded context to search.
    context: String,
    /// Search query (free text for bm25, literal for substring, regex for regex).
    query: String,
    #[serde(default = "default_k")]
    k: usize,
    /// One of `bm25` (default), `substring`, `regex`.
    #[serde(default)]
    mode: Option<String>,
}

fn default_k() -> usize {
    10
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct GetChunkParams {
    context: String,
    /// Chunk id (e.g. `doc:0042`) — preferred.
    #[serde(default)]
    chunk_id: Option<String>,
    /// `[start, end]` byte range into the context blob.
    #[serde(default)]
    byte_range: Option<[usize; 2]>,
    /// Substring of a section heading path to fetch.
    #[serde(default)]
    section: Option<String>,
    #[serde(default = "default_get_chunk_max_chars")]
    max_chars: usize,
}

fn default_get_chunk_max_chars() -> usize {
    8_000
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ContextRef {
    context: String,
    #[serde(default)]
    chunk_ids: Option<Vec<String>>,
    #[serde(default)]
    section: Option<String>,
    #[serde(default)]
    byte_range: Option<[usize; 2]>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct SubQueryParams {
    /// What you want the sub-LLM to do with the referenced slices.
    prompt: String,
    /// References into already-loaded contexts.
    #[serde(default)]
    context_refs: Vec<ContextRef>,
    /// Optional inline content (used when slicing isn't appropriate).
    #[serde(default)]
    inline_content: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct BatchSubQueryParams {
    queries: Vec<SubQueryParams>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct StoreParams {
    key: String,
    value: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct RetrieveParams {
    key: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ListContextsParams {}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ListKeysParams {}

pub struct RlmClient {
    info: InitializeResult,
    context: PlatformExtensionContext,
}

impl RlmClient {
    pub fn new(context: PlatformExtensionContext) -> Result<Self> {
        let info = InitializeResult::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(
                Implementation::new(EXTENSION_NAME.to_string(), "1.0.0".to_string())
                    .with_title("Recursive Language Model"),
            )
            .with_instructions(
                indoc! {r#"
                You are operating in Recursive Language Model (RLM) mode. Long inputs
                live OUTSIDE your message history, in named contexts. NEVER ask the user
                to paste content already loaded in a context — use the rlm__* tools.

                Workflow for a long-context task:
                1. `rlm__list_contexts` — see what's loaded.
                2. `rlm__search` — find relevant chunks (BM25 by default).
                3. `rlm__get_chunk` — pull only what you need.
                4. `rlm__sub_query` — delegate per-chunk analysis to a fresh sub-LLM call.
                5. `rlm__store` / `rlm__retrieve` — persist intermediate findings between turns.
                6. Reply with your final answer (no special finalize tool — just answer).

                Keep your message history small: most tool responses are deliberately
                truncated. Full content stays in the store and you can re-fetch by id.
                "#}
                .to_string(),
            );

        Ok(Self { info, context })
    }

    fn store(&self) -> Arc<RlmStore> {
        self.context.rlm_store.clone()
    }

    async fn provider(&self) -> Option<Arc<dyn Provider>> {
        let weak = self.context.extension_manager.as_ref()?;
        let em = weak.upgrade()?;
        let guard = em.get_provider().lock().await;
        guard.clone()
    }

    fn parse_args<T: for<'de> Deserialize<'de>>(args: Option<JsonObject>) -> Result<T, String> {
        let v = serde_json::Value::Object(args.unwrap_or_default());
        serde_json::from_value(v).map_err(|e| format!("invalid arguments: {e}"))
    }

    async fn handle_list_contexts(&self) -> Result<Vec<Content>, String> {
        let contexts = self.store().list_contexts();
        Ok(vec![Content::text(
            serde_json::to_string_pretty(&contexts).unwrap_or_else(|_| "[]".into()),
        )])
    }

    async fn handle_list_keys(&self) -> Result<Vec<Content>, String> {
        let keys = self.store().list_memory_keys();
        Ok(vec![Content::text(
            serde_json::to_string(&keys).unwrap_or_else(|_| "[]".into()),
        )])
    }

    async fn handle_search(&self, args: Option<JsonObject>) -> Result<Vec<Content>, String> {
        let p: SearchParams = Self::parse_args(args)?;
        let mode = match p.mode.as_deref().unwrap_or("bm25") {
            "bm25" => SearchMode::Bm25,
            "substring" => SearchMode::Substring,
            "regex" => SearchMode::Regex,
            other => return Err(format!("unknown search mode: {other}")),
        };
        let ctx = self
            .store()
            .get(&p.context)
            .ok_or_else(|| format!("no such context: {}", p.context))?;
        let hits: Vec<SearchHit> = ctx.search(&p.query, p.k, mode);
        let body = serde_json::to_string_pretty(&hits).unwrap_or_else(|_| "[]".into());
        Ok(vec![Content::text(truncate_for_history(body))])
    }

    async fn handle_get_chunk(&self, args: Option<JsonObject>) -> Result<Vec<Content>, String> {
        let p: GetChunkParams = Self::parse_args(args)?;
        let ctx = self
            .store()
            .get(&p.context)
            .ok_or_else(|| format!("no such context: {}", p.context))?;
        let result: serde_json::Value = if let Some(id) = p.chunk_id {
            match ctx.get_chunk_by_id(&id, p.max_chars) {
                Some(c) => serde_json::to_value(c).unwrap_or(serde_json::Value::Null),
                None => return Err(format!("chunk not found: {id}")),
            }
        } else if let Some(range) = p.byte_range {
            match ctx.get_chunk_by_range(range[0], range[1], p.max_chars) {
                Some(c) => serde_json::to_value(c).unwrap_or(serde_json::Value::Null),
                None => return Err("byte_range out of bounds".into()),
            }
        } else if let Some(section) = p.section {
            let chunks: Vec<ChunkData> = ctx.get_chunks_by_section(&section, p.max_chars);
            serde_json::to_value(chunks).unwrap_or(serde_json::Value::Null)
        } else {
            return Err("must provide chunk_id, byte_range, or section".into());
        };
        let body = serde_json::to_string(&result).unwrap_or_default();
        Ok(vec![Content::text(truncate_for_history(body))])
    }

    fn assemble_sub_query_content(&self, p: &SubQueryParams) -> Result<String, String> {
        let mut buf = String::new();
        let mut total = 0usize;
        for r in &p.context_refs {
            let ctx = self
                .store()
                .get(&r.context)
                .ok_or_else(|| format!("no such context: {}", r.context))?;
            if let Some(ids) = &r.chunk_ids {
                for id in ids {
                    if let Some(c) = ctx.get_chunk_by_id(id, SUB_QUERY_MAX_INLINE_CHARS) {
                        append_capped(&mut buf, &mut total, &c);
                    }
                }
            } else if let Some(range) = r.byte_range {
                if let Some(c) =
                    ctx.get_chunk_by_range(range[0], range[1], SUB_QUERY_MAX_INLINE_CHARS)
                {
                    append_capped(&mut buf, &mut total, &c);
                }
            } else if let Some(section) = &r.section {
                for c in ctx.get_chunks_by_section(section, SUB_QUERY_MAX_INLINE_CHARS) {
                    append_capped(&mut buf, &mut total, &c);
                }
            }
        }
        if let Some(inline) = &p.inline_content {
            buf.push_str("\n\n");
            buf.push_str(inline);
        }
        Ok(buf)
    }

    async fn handle_sub_query(
        &self,
        args: Option<JsonObject>,
        cancel: CancellationToken,
    ) -> Result<Vec<Content>, String> {
        let p: SubQueryParams = Self::parse_args(args)?;
        let provider = self
            .provider()
            .await
            .ok_or_else(|| "no provider configured for sub_query".to_string())?;
        let assembled = self.assemble_sub_query_content(&p)?;
        let result = run_leaf_sub_query(provider, &p.prompt, &assembled, cancel).await?;
        Ok(vec![Content::text(truncate_for_history(result))])
    }

    async fn handle_batch_sub_query(
        &self,
        args: Option<JsonObject>,
        cancel: CancellationToken,
    ) -> Result<Vec<Content>, String> {
        let p: BatchSubQueryParams = Self::parse_args(args)?;
        let provider = self
            .provider()
            .await
            .ok_or_else(|| "no provider configured for sub_query".to_string())?;
        let semaphore = Arc::new(tokio::sync::Semaphore::new(BATCH_CONCURRENCY));
        let store_for_assembly = self.context.rlm_store.clone();
        let mut handles = Vec::with_capacity(p.queries.len());
        for q in p.queries {
            let provider = provider.clone();
            let cancel = cancel.clone();
            let semaphore = semaphore.clone();
            let store = store_for_assembly.clone();
            handles.push(tokio::spawn(async move {
                let _permit = semaphore.acquire_owned().await.ok();
                let assembled = assemble_with_store(&store, &q)?;
                run_leaf_sub_query(provider, &q.prompt, &assembled, cancel).await
            }));
        }
        let mut answers = Vec::with_capacity(handles.len());
        for h in handles {
            match h.await {
                Ok(Ok(s)) => answers.push(s),
                Ok(Err(e)) => answers.push(format!("[error] {e}")),
                Err(e) => answers.push(format!("[join error] {e}")),
            }
        }
        let body = serde_json::to_string_pretty(&answers).unwrap_or_default();
        Ok(vec![Content::text(truncate_for_history(body))])
    }

    async fn handle_store(&self, args: Option<JsonObject>) -> Result<Vec<Content>, String> {
        let p: StoreParams = Self::parse_args(args)?;
        self.store().store_memory(p.key.clone(), p.value);
        Ok(vec![Content::text(format!("stored: {}", p.key))])
    }

    async fn handle_retrieve(&self, args: Option<JsonObject>) -> Result<Vec<Content>, String> {
        let p: RetrieveParams = Self::parse_args(args)?;
        let v = self
            .store()
            .retrieve_memory(&p.key)
            .ok_or_else(|| format!("no such key: {}", p.key))?;
        let body = serde_json::to_string(&*v).unwrap_or_default();
        Ok(vec![Content::text(truncate_for_history(body))])
    }

    fn get_tools() -> Vec<Tool> {
        let s = |params: serde_json::Value| params.as_object().cloned().unwrap_or_default();
        vec![
            Tool::new(
                "list_contexts",
                "List all loaded contexts (name, source, size, chunk count).",
                s(serde_json::to_value(schema_for!(ListContextsParams)).unwrap()),
            )
            .annotate(ToolAnnotations::from_raw(
                Some("List Contexts".into()),
                Some(true),
                Some(false),
                Some(false),
                Some(false),
            )),
            Tool::new(
                "search",
                "Search a loaded context by query. Modes: bm25 (default), substring, regex. Returns top-k matches with previews — never the full chunk.",
                s(serde_json::to_value(schema_for!(SearchParams)).unwrap()),
            )
            .annotate(ToolAnnotations::from_raw(
                Some("RLM Search".into()),
                Some(true),
                Some(false),
                Some(false),
                Some(false),
            )),
            Tool::new(
                "get_chunk",
                "Fetch a specific chunk (by chunk_id, byte_range, or section heading). Always capped by max_chars; the full content stays in the store.",
                s(serde_json::to_value(schema_for!(GetChunkParams)).unwrap()),
            )
            .annotate(ToolAnnotations::from_raw(
                Some("RLM Get Chunk".into()),
                Some(true),
                Some(false),
                Some(false),
                Some(false),
            )),
            Tool::new(
                "sub_query",
                "Delegate analysis of one or more referenced slices to a fresh sub-LLM call. Returns just the answer (no history).",
                s(serde_json::to_value(schema_for!(SubQueryParams)).unwrap()),
            )
            .annotate(ToolAnnotations::from_raw(
                Some("RLM Sub Query".into()),
                Some(false),
                Some(false),
                Some(false),
                Some(false),
            )),
            Tool::new(
                "batch_sub_query",
                "Run multiple sub-queries in parallel (cap 8 concurrent).",
                s(serde_json::to_value(schema_for!(BatchSubQueryParams)).unwrap()),
            )
            .annotate(ToolAnnotations::from_raw(
                Some("RLM Batch Sub Query".into()),
                Some(false),
                Some(false),
                Some(false),
                Some(false),
            )),
            Tool::new(
                "store",
                "Persist a key/value finding to session-scoped memory.",
                s(serde_json::to_value(schema_for!(StoreParams)).unwrap()),
            )
            .annotate(ToolAnnotations::from_raw(
                Some("RLM Store".into()),
                Some(false),
                Some(true),
                Some(false),
                Some(false),
            )),
            Tool::new(
                "retrieve",
                "Pull a previously stored value from session memory.",
                s(serde_json::to_value(schema_for!(RetrieveParams)).unwrap()),
            )
            .annotate(ToolAnnotations::from_raw(
                Some("RLM Retrieve".into()),
                Some(true),
                Some(false),
                Some(false),
                Some(false),
            )),
            Tool::new(
                "list_keys",
                "List all keys currently in session memory.",
                s(serde_json::to_value(schema_for!(ListKeysParams)).unwrap()),
            )
            .annotate(ToolAnnotations::from_raw(
                Some("RLM List Keys".into()),
                Some(true),
                Some(false),
                Some(false),
                Some(false),
            )),
        ]
    }
}

fn append_capped(buf: &mut String, total: &mut usize, chunk: &ChunkData) {
    if *total >= SUB_QUERY_MAX_INLINE_CHARS {
        return;
    }
    let remaining = SUB_QUERY_MAX_INLINE_CHARS.saturating_sub(*total);
    let header = match (
        chunk.section_path.as_deref(),
        chunk
            .source_file
            .as_deref()
            .map(|p| p.display().to_string()),
    ) {
        (Some(s), Some(f)) => format!("\n--- {f} :: {s}\n"),
        (Some(s), None) => format!("\n--- {s}\n"),
        (None, Some(f)) => format!("\n--- {f}\n"),
        (None, None) => "\n---\n".to_string(),
    };
    buf.push_str(&header);
    let take: String = chunk.text.chars().take(remaining).collect();
    *total += take.chars().count() + header.chars().count();
    buf.push_str(&take);
}

fn assemble_with_store(store: &Arc<RlmStore>, p: &SubQueryParams) -> Result<String, String> {
    let mut buf = String::new();
    let mut total = 0usize;
    for r in &p.context_refs {
        let ctx = store
            .get(&r.context)
            .ok_or_else(|| format!("no such context: {}", r.context))?;
        if let Some(ids) = &r.chunk_ids {
            for id in ids {
                if let Some(c) = ctx.get_chunk_by_id(id, SUB_QUERY_MAX_INLINE_CHARS) {
                    append_capped(&mut buf, &mut total, &c);
                }
            }
        } else if let Some(range) = r.byte_range {
            if let Some(c) = ctx.get_chunk_by_range(range[0], range[1], SUB_QUERY_MAX_INLINE_CHARS)
            {
                append_capped(&mut buf, &mut total, &c);
            }
        } else if let Some(section) = &r.section {
            for c in ctx.get_chunks_by_section(section, SUB_QUERY_MAX_INLINE_CHARS) {
                append_capped(&mut buf, &mut total, &c);
            }
        }
    }
    if let Some(inline) = &p.inline_content {
        buf.push_str("\n\n");
        buf.push_str(inline);
    }
    Ok(buf)
}

async fn run_leaf_sub_query(
    provider: Arc<dyn Provider>,
    prompt: &str,
    content: &str,
    _cancel: CancellationToken,
) -> Result<String, String> {
    let system = "You are a focused sub-LLM in a Recursive Language Model. \
                  Answer the user prompt using the provided content. Be concise.";
    let user_text = if content.is_empty() {
        prompt.to_string()
    } else {
        format!("{prompt}\n\n--- content ---\n{content}")
    };
    let msg = Message::user().with_text(user_text);
    let model_config = provider.get_model_config();
    let session_id = format!("rlm-sub-{}", uuid::Uuid::new_v4());
    let (response, _usage) = provider
        .complete(&model_config, &session_id, system, &[msg], &[])
        .await
        .map_err(|e| format!("sub_query provider error: {e}"))?;
    Ok(response.as_concat_text())
}

fn truncate_for_history(s: String) -> String {
    if s.chars().count() <= TOOL_RESPONSE_PREVIEW_CHARS {
        return s;
    }
    let head: String = s.chars().take(TOOL_RESPONSE_PREVIEW_CHARS).collect();
    let total = s.chars().count();
    format!(
        "{head}\n\n[truncated for history: {} of {total} chars shown]",
        TOOL_RESPONSE_PREVIEW_CHARS
    )
}

#[async_trait]
impl McpClientTrait for RlmClient {
    async fn list_tools(
        &self,
        _session_id: &str,
        _next_cursor: Option<String>,
        _cancellation_token: CancellationToken,
    ) -> Result<ListToolsResult, Error> {
        Ok(ListToolsResult {
            tools: Self::get_tools(),
            next_cursor: None,
            meta: None,
        })
    }

    async fn call_tool(
        &self,
        _ctx: &ToolCallContext,
        name: &str,
        arguments: Option<JsonObject>,
        cancellation_token: CancellationToken,
    ) -> Result<CallToolResult, Error> {
        let result = match name {
            "list_contexts" => self.handle_list_contexts().await,
            "search" => self.handle_search(arguments).await,
            "get_chunk" => self.handle_get_chunk(arguments).await,
            "sub_query" => self.handle_sub_query(arguments, cancellation_token).await,
            "batch_sub_query" => {
                self.handle_batch_sub_query(arguments, cancellation_token)
                    .await
            }
            "store" => self.handle_store(arguments).await,
            "retrieve" => self.handle_retrieve(arguments).await,
            "list_keys" => self.handle_list_keys().await,
            other => Err(format!("unknown tool: {other}")),
        };
        match result {
            Ok(content) => Ok(CallToolResult::success(content)),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "Error: {e}"
            ))])),
        }
    }

    fn get_info(&self) -> Option<&InitializeResult> {
        Some(&self.info)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agents::rlm::RlmStore;
    use std::io::Write;

    fn ctx_with_store(store: Arc<RlmStore>) -> PlatformExtensionContext {
        PlatformExtensionContext {
            extension_manager: None,
            session_manager: Arc::new(crate::session::SessionManager::new(std::env::temp_dir())),
            session: None,
            rlm_store: store,
        }
    }

    fn extract_text(result: &CallToolResult) -> String {
        result
            .content
            .iter()
            .filter_map(|c| c.as_text().map(|t| t.text.clone()))
            .collect::<Vec<_>>()
            .join("")
    }

    #[tokio::test]
    async fn list_contexts_after_load() {
        let store = Arc::new(RlmStore::new());
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(b"# H\nhello apple banana\n").unwrap();
        store.load_file(f.path(), "doc").unwrap();

        let client = RlmClient::new(ctx_with_store(store)).unwrap();
        let tools = client
            .list_tools("s", None, CancellationToken::new())
            .await
            .unwrap();
        assert!(tools.tools.iter().any(|t| t.name == "search"));
        let r = client
            .call_tool(
                &ToolCallContext::new("s".into(), None, None),
                "list_contexts",
                None,
                CancellationToken::new(),
            )
            .await
            .unwrap();
        let body = extract_text(&r);
        assert!(body.contains("\"name\""));
        assert!(body.contains("doc"));
    }

    #[tokio::test]
    async fn search_and_get_chunk_round_trip() {
        let store = Arc::new(RlmStore::new());
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(
            b"# Intro\nhi\n# Secret\nthe magic phrase is BLUE_TOKEN_91 here\n# Outro\nbye\n",
        )
        .unwrap();
        store.load_file(f.path(), "doc").unwrap();
        let client = RlmClient::new(ctx_with_store(store)).unwrap();

        let search = client
            .call_tool(
                &ToolCallContext::new("s".into(), None, None),
                "search",
                Some(
                    serde_json::json!({"context":"doc","query":"magic phrase","mode":"bm25","k":3})
                        .as_object()
                        .unwrap()
                        .clone(),
                ),
                CancellationToken::new(),
            )
            .await
            .unwrap();
        let body = extract_text(&search);
        assert!(body.contains("chunk_id"));

        let value: serde_json::Value = serde_json::from_str(&body).unwrap();
        let id = value[0]["chunk_id"].as_str().unwrap().to_string();

        let chunk = client
            .call_tool(
                &ToolCallContext::new("s".into(), None, None),
                "get_chunk",
                Some(
                    serde_json::json!({"context":"doc","chunk_id":id,"max_chars":4096})
                        .as_object()
                        .unwrap()
                        .clone(),
                ),
                CancellationToken::new(),
            )
            .await
            .unwrap();
        assert!(extract_text(&chunk).contains("BLUE_TOKEN_91"));
    }

    #[tokio::test]
    async fn store_retrieve_round_trip() {
        let store = Arc::new(RlmStore::new());
        let client = RlmClient::new(ctx_with_store(store)).unwrap();
        let _ = client
            .call_tool(
                &ToolCallContext::new("s".into(), None, None),
                "store",
                Some(
                    serde_json::json!({"key":"finding","value":{"n":3}})
                        .as_object()
                        .unwrap()
                        .clone(),
                ),
                CancellationToken::new(),
            )
            .await
            .unwrap();
        let r = client
            .call_tool(
                &ToolCallContext::new("s".into(), None, None),
                "retrieve",
                Some(
                    serde_json::json!({"key":"finding"})
                        .as_object()
                        .unwrap()
                        .clone(),
                ),
                CancellationToken::new(),
            )
            .await
            .unwrap();
        assert!(extract_text(&r).contains("\"n\":3"));
    }

    #[tokio::test]
    async fn sub_query_without_provider_errors_gracefully() {
        let store = Arc::new(RlmStore::new());
        let client = RlmClient::new(ctx_with_store(store)).unwrap();
        let r = client
            .call_tool(
                &ToolCallContext::new("s".into(), None, None),
                "sub_query",
                Some(
                    serde_json::json!({"prompt":"hi","context_refs":[]})
                        .as_object()
                        .unwrap()
                        .clone(),
                ),
                CancellationToken::new(),
            )
            .await
            .unwrap();
        assert!(extract_text(&r).contains("no provider"));
    }
}
