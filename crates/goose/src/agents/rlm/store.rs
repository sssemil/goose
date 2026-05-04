//! Per-session lock-free RLM store.
//!
//! Holds named contexts (loaded files / directories) as immutable blobs
//! plus a chunk index, and a key/value memory map. The root LLM never
//! sees the blobs directly — it interacts via the `rlm__*` tools, which
//! return small metadata stubs and on-demand chunk slices.
//!
//! Concurrency model:
//! - `contexts` and `memory` are `DashMap`s (sharded locks).
//! - `RlmContext` is wrapped in `Arc` and is **immutable after load**, so
//!   concurrent reads (search / get_chunk / sub_query) never block each other.
//! - Writers to `memory` only contend at the shard level.
//! - `bytes_ingested` is an `AtomicU64` for the size cap.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use bm25::{Language, SearchEngine, SearchEngineBuilder};
use dashmap::DashMap;
use ignore::WalkBuilder;
use regex::RegexBuilder;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Default per-session ingest cap. Override via [`RlmStore::with_max_bytes`].
pub const DEFAULT_MAX_BYTES: u64 = 50 * 1024 * 1024; // 50 MB

/// Default recursion cap for `rlm__sub_query` chains. Beyond this, sub-queries
/// fall back to a leaf LLM call with no further tools.
pub const DEFAULT_MAX_DEPTH: u32 = 2;

/// Default chunk size in approximate tokens (1 token ~= 4 chars).
const CHUNK_TOKEN_TARGET: usize = 2_000;
const CHUNK_OVERLAP_TOKENS: usize = 200;
const CHARS_PER_TOKEN: usize = 4;

/// File extensions skipped during directory ingest (binary / generated).
const SKIP_EXTENSIONS: &[&str] = &[
    "png", "jpg", "jpeg", "gif", "webp", "ico", "bmp", "tiff", "svg", "pdf", "zip", "tar", "gz",
    "tgz", "bz2", "xz", "7z", "rar", "exe", "dll", "so", "dylib", "a", "o", "obj", "class", "jar",
    "wasm", "pyc", "pyo", "lock", "min.js", "min.css", "woff", "woff2", "ttf", "otf", "mp3", "mp4",
    "mov", "avi", "mkv", "wav", "flac", "ogg",
];

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub enum SearchMode {
    #[default]
    Bm25,
    Substring,
    Regex,
}

/// A single retrievable chunk in a context.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Chunk {
    pub id: String,
    /// Inclusive start, exclusive end in the context's blob bytes.
    pub byte_range: (usize, usize),
    pub token_count: usize,
    /// Heading path (for markdown-like inputs) or `None` for plain chunks.
    pub section_path: Option<String>,
    /// Source file when context was loaded from a directory.
    pub source_file: Option<PathBuf>,
}

#[derive(Clone, Debug, Serialize)]
pub struct SearchHit {
    pub chunk_id: String,
    pub section_path: Option<String>,
    pub source_file: Option<PathBuf>,
    pub score: f32,
    pub preview: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct ChunkData {
    pub chunk_id: String,
    pub text: String,
    pub truncated: bool,
    pub total_chars: usize,
    pub section_path: Option<String>,
    pub source_file: Option<PathBuf>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ContextSummary {
    pub name: String,
    pub source: String,
    pub total_chars: usize,
    pub total_tokens: usize,
    pub n_chunks: usize,
}

/// A loaded context. Immutable after construction.
#[derive(Debug)]
pub struct RlmContext {
    pub name: String,
    pub source: String,
    blob: Vec<u8>,
    pub chunks: Vec<Chunk>,
    pub total_chars: usize,
    pub total_tokens: usize,
    bm25: Option<SearchEngine<usize>>,
}

impl RlmContext {
    pub fn summary(&self) -> ContextSummary {
        ContextSummary {
            name: self.name.clone(),
            source: self.source.clone(),
            total_chars: self.total_chars,
            total_tokens: self.total_tokens,
            n_chunks: self.chunks.len(),
        }
    }

    fn chunk_text(&self, idx: usize) -> Option<&str> {
        let chunk = self.chunks.get(idx)?;
        let (s, e) = chunk.byte_range;
        std::str::from_utf8(self.blob.get(s..e)?).ok()
    }

    fn chunk_text_by_id(&self, id: &str) -> Option<(usize, &str)> {
        let idx = self.chunks.iter().position(|c| c.id == id)?;
        Some((idx, self.chunk_text(idx)?))
    }

    pub fn search(&self, query: &str, k: usize, mode: SearchMode) -> Vec<SearchHit> {
        match mode {
            SearchMode::Bm25 => self.search_bm25(query, k),
            SearchMode::Substring => self.search_substring(query, k, false),
            SearchMode::Regex => self.search_regex(query, k),
        }
    }

    fn search_bm25(&self, query: &str, k: usize) -> Vec<SearchHit> {
        let Some(engine) = self.bm25.as_ref() else {
            return Vec::new();
        };
        let mut results: Vec<_> = engine
            .search(query, k.max(1))
            .into_iter()
            .map(|r| (r.document.id, r.score))
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        results
            .into_iter()
            .filter_map(|(idx, score)| self.make_hit(idx, score))
            .collect()
    }

    fn search_substring(&self, query: &str, k: usize, _: bool) -> Vec<SearchHit> {
        let needle = query.to_lowercase();
        if needle.is_empty() {
            return Vec::new();
        }
        let mut hits = Vec::new();
        for (idx, chunk) in self.chunks.iter().enumerate() {
            let Some(text) = self.chunk_text(idx) else {
                continue;
            };
            let count = text.to_lowercase().matches(&needle).count();
            if count > 0 {
                if let Some(mut hit) = self.make_hit(idx, count as f32) {
                    if let Some(pos) = text.to_lowercase().find(&needle) {
                        hit.preview = preview_window(text, pos, needle.len(), 200);
                    }
                    hits.push(hit);
                }
            }
            if hits.len() >= k * 4 {
                break;
            }
            let _ = chunk;
        }
        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hits.truncate(k);
        hits
    }

    fn search_regex(&self, pattern: &str, k: usize) -> Vec<SearchHit> {
        let Ok(re) = RegexBuilder::new(pattern).case_insensitive(true).build() else {
            return Vec::new();
        };
        let mut hits = Vec::new();
        for idx in 0..self.chunks.len() {
            let Some(text) = self.chunk_text(idx) else {
                continue;
            };
            let matches: Vec<_> = re.find_iter(text).collect();
            if !matches.is_empty() {
                if let Some(mut hit) = self.make_hit(idx, matches.len() as f32) {
                    let m = &matches[0];
                    hit.preview = preview_window(text, m.start(), m.end() - m.start(), 200);
                    hits.push(hit);
                }
            }
        }
        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hits.truncate(k);
        hits
    }

    fn make_hit(&self, idx: usize, score: f32) -> Option<SearchHit> {
        let chunk = self.chunks.get(idx)?;
        let text = self.chunk_text(idx)?;
        Some(SearchHit {
            chunk_id: chunk.id.clone(),
            section_path: chunk.section_path.clone(),
            source_file: chunk.source_file.clone(),
            score,
            preview: text.chars().take(200).collect(),
        })
    }

    pub fn get_chunk_by_id(&self, id: &str, max_chars: usize) -> Option<ChunkData> {
        let (idx, text) = self.chunk_text_by_id(id)?;
        let chunk = &self.chunks[idx];
        let total_chars = text.chars().count();
        let truncated = total_chars > max_chars;
        let body: String = if truncated {
            text.chars().take(max_chars).collect()
        } else {
            text.to_string()
        };
        Some(ChunkData {
            chunk_id: chunk.id.clone(),
            text: body,
            truncated,
            total_chars,
            section_path: chunk.section_path.clone(),
            source_file: chunk.source_file.clone(),
        })
    }

    pub fn get_chunk_by_range(
        &self,
        start: usize,
        end: usize,
        max_chars: usize,
    ) -> Option<ChunkData> {
        if start >= end || end > self.blob.len() {
            return None;
        }
        let raw = std::str::from_utf8(&self.blob[start..end]).ok()?;
        let total_chars = raw.chars().count();
        let truncated = total_chars > max_chars;
        let body: String = if truncated {
            raw.chars().take(max_chars).collect()
        } else {
            raw.to_string()
        };
        Some(ChunkData {
            chunk_id: format!("{}:bytes:{}-{}", self.name, start, end),
            text: body,
            truncated,
            total_chars,
            section_path: None,
            source_file: None,
        })
    }

    pub fn get_chunks_by_section(&self, section: &str, max_chars: usize) -> Vec<ChunkData> {
        let needle = section.to_lowercase();
        let mut out = Vec::new();
        let mut budget = max_chars;
        for (idx, chunk) in self.chunks.iter().enumerate() {
            let path_match = chunk
                .section_path
                .as_deref()
                .map(|p| p.to_lowercase().contains(&needle))
                .unwrap_or(false);
            if !path_match {
                continue;
            }
            if let Some(text) = self.chunk_text(idx) {
                let total = text.chars().count();
                let take = budget.min(total);
                let truncated = take < total;
                let body: String = text.chars().take(take).collect();
                budget = budget.saturating_sub(take);
                out.push(ChunkData {
                    chunk_id: chunk.id.clone(),
                    text: body,
                    truncated,
                    total_chars: total,
                    section_path: chunk.section_path.clone(),
                    source_file: chunk.source_file.clone(),
                });
                if budget == 0 {
                    break;
                }
            }
        }
        out
    }
}

fn preview_window(text: &str, match_start: usize, match_len: usize, window: usize) -> String {
    let start = match_start.saturating_sub(window / 2);
    let end = (match_start + match_len + window / 2).min(text.len());
    let mut adjusted_start = start;
    while adjusted_start > 0 && !text.is_char_boundary(adjusted_start) {
        adjusted_start -= 1;
    }
    let mut adjusted_end = end;
    while adjusted_end < text.len() && !text.is_char_boundary(adjusted_end) {
        adjusted_end += 1;
    }
    let mut out = String::new();
    if adjusted_start > 0 {
        out.push('…');
    }
    out.push_str(&text[adjusted_start..adjusted_end]);
    if adjusted_end < text.len() {
        out.push('…');
    }
    out
}

/// Lock-free per-session RLM state.
pub struct RlmStore {
    contexts: DashMap<String, Arc<RlmContext>>,
    memory: DashMap<String, Arc<Value>>,
    bytes_ingested: AtomicU64,
    max_bytes: u64,
    max_depth: u32,
}

impl Default for RlmStore {
    fn default() -> Self {
        Self::new()
    }
}

impl RlmStore {
    pub fn new() -> Self {
        Self::with_max_bytes(DEFAULT_MAX_BYTES)
    }

    pub fn with_max_bytes(max_bytes: u64) -> Self {
        Self {
            contexts: DashMap::new(),
            memory: DashMap::new(),
            bytes_ingested: AtomicU64::new(0),
            max_bytes,
            max_depth: DEFAULT_MAX_DEPTH,
        }
    }

    pub fn with_max_depth(mut self, max_depth: u32) -> Self {
        self.max_depth = max_depth;
        self
    }

    pub fn max_depth(&self) -> u32 {
        self.max_depth
    }

    pub fn list_contexts(&self) -> Vec<ContextSummary> {
        let mut out: Vec<_> = self
            .contexts
            .iter()
            .map(|entry| entry.value().summary())
            .collect();
        out.sort_by(|a, b| a.name.cmp(&b.name));
        out
    }

    pub fn get(&self, name: &str) -> Option<Arc<RlmContext>> {
        self.contexts.get(name).map(|e| e.value().clone())
    }

    pub fn store_memory(&self, key: impl Into<String>, value: Value) {
        self.memory.insert(key.into(), Arc::new(value));
    }

    pub fn retrieve_memory(&self, key: &str) -> Option<Arc<Value>> {
        self.memory.get(key).map(|e| e.value().clone())
    }

    pub fn list_memory_keys(&self) -> Vec<String> {
        let mut keys: Vec<_> = self.memory.iter().map(|e| e.key().clone()).collect();
        keys.sort();
        keys
    }

    /// Load a single file as a context.
    pub fn load_file(&self, path: &Path, name: &str) -> Result<Arc<RlmContext>> {
        let raw = std::fs::read(path)
            .with_context(|| format!("reading rlm context file {}", path.display()))?;
        let text = String::from_utf8(raw)
            .map_err(|_| anyhow!("rlm context file is not valid UTF-8: {}", path.display()))?;
        self.check_budget(text.len() as u64)?;
        let chunks = chunk_text(&text, name, None);
        let ctx = build_context(name.to_string(), path.display().to_string(), text, chunks);
        let arc = Arc::new(ctx);
        self.contexts.insert(name.to_string(), arc.clone());
        Ok(arc)
    }

    /// Recursively load a directory as a single context. Honours `.gitignore`
    /// and skips known binary/generated extensions.
    pub fn load_directory(&self, path: &Path, name: &str) -> Result<Arc<RlmContext>> {
        let mut blob = String::new();
        let mut chunks: Vec<Chunk> = Vec::new();
        let mut total_files = 0usize;

        let walker = WalkBuilder::new(path)
            .standard_filters(true)
            .git_ignore(true)
            .git_exclude(true)
            .hidden(true)
            .build();

        for result in walker {
            let entry = match result {
                Ok(e) => e,
                Err(_) => continue,
            };
            if !entry.file_type().is_some_and(|t| t.is_file()) {
                continue;
            }
            let file_path = entry.path();
            if should_skip_path(file_path) {
                continue;
            }
            let raw = match std::fs::read(file_path) {
                Ok(r) => r,
                Err(_) => continue,
            };
            // Skip large binaries.
            if raw.len() > 1_000_000 && !is_likely_text(&raw[..raw.len().min(8192)]) {
                continue;
            }
            let text = match String::from_utf8(raw) {
                Ok(t) => t,
                Err(_) => continue,
            };
            self.check_budget(text.len() as u64)?;

            let header = format!("\n\n===== {} =====\n", file_path.display());
            blob.push_str(&header);
            let file_offset = blob.len();
            blob.push_str(&text);
            let file_chunks = chunk_text_with_offset(
                &text,
                name,
                Some(file_path.to_path_buf()),
                file_offset,
                chunks.len(),
            );
            chunks.extend(file_chunks);
            total_files += 1;
        }

        if total_files == 0 {
            return Err(anyhow!(
                "no readable text files found under {}",
                path.display()
            ));
        }

        let ctx = build_context(name.to_string(), path.display().to_string(), blob, chunks);
        let arc = Arc::new(ctx);
        self.contexts.insert(name.to_string(), arc.clone());
        Ok(arc)
    }

    fn check_budget(&self, additional: u64) -> Result<()> {
        let prev = self.bytes_ingested.fetch_add(additional, Ordering::Relaxed);
        let total = prev.saturating_add(additional);
        if total > self.max_bytes {
            // Roll back the count so subsequent loads can succeed if they fit.
            self.bytes_ingested.fetch_sub(additional, Ordering::Relaxed);
            return Err(anyhow!(
                "rlm ingest limit exceeded: {} bytes (cap {})",
                total,
                self.max_bytes
            ));
        }
        Ok(())
    }
}

fn should_skip_path(path: &Path) -> bool {
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        if SKIP_EXTENSIONS.contains(&ext.to_lowercase().as_str()) {
            return true;
        }
    }
    if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
        let lower = name.to_lowercase();
        for skip in SKIP_EXTENSIONS {
            if lower.ends_with(&format!(".{}", skip)) {
                return true;
            }
        }
    }
    false
}

fn is_likely_text(sample: &[u8]) -> bool {
    if sample.is_empty() {
        return true;
    }
    let nonprint = sample
        .iter()
        .filter(|b| **b < 0x09 || (**b > 0x0d && **b < 0x20 && **b != 0x1b))
        .count();
    (nonprint * 100 / sample.len()) < 5
}

fn build_context(name: String, source: String, blob: String, chunks: Vec<Chunk>) -> RlmContext {
    let total_chars = blob.chars().count();
    let total_tokens = chunks.iter().map(|c| c.token_count).sum();
    let bytes = blob.into_bytes();

    let bm25 = build_bm25(&bytes, &chunks);

    RlmContext {
        name,
        source,
        blob: bytes,
        chunks,
        total_chars,
        total_tokens,
        bm25,
    }
}

fn build_bm25(blob: &[u8], chunks: &[Chunk]) -> Option<SearchEngine<usize>> {
    if chunks.is_empty() {
        return None;
    }
    let docs: Vec<bm25::Document<usize>> = chunks
        .iter()
        .enumerate()
        .filter_map(|(idx, chunk)| {
            let (s, e) = chunk.byte_range;
            let text = std::str::from_utf8(blob.get(s..e)?).ok()?;
            Some(bm25::Document {
                id: idx,
                contents: text.to_string(),
            })
        })
        .collect();
    if docs.is_empty() {
        return None;
    }
    SearchEngineBuilder::<usize>::with_documents(Language::English, docs)
        .build()
        .into()
}

fn chunk_text(text: &str, ctx_name: &str, source_file: Option<PathBuf>) -> Vec<Chunk> {
    chunk_text_with_offset(text, ctx_name, source_file, 0, 0)
}

fn chunk_text_with_offset(
    text: &str,
    ctx_name: &str,
    source_file: Option<PathBuf>,
    base_offset: usize,
    chunk_id_start: usize,
) -> Vec<Chunk> {
    let mut chunks = Vec::new();
    let target_chars = CHUNK_TOKEN_TARGET * CHARS_PER_TOKEN;
    let overlap_chars = CHUNK_OVERLAP_TOKENS * CHARS_PER_TOKEN;

    // First pass: split on top-level headings (lines starting with `# ` / `## ` / `### `).
    let sections = split_on_headings(text);

    for (heading, body) in sections {
        // Split very long sections into windows.
        let body_bytes = body.as_bytes();
        let body_offset = body.as_ptr() as usize - text.as_ptr() as usize;
        if body_bytes.len() <= target_chars {
            push_chunk(
                &mut chunks,
                ctx_name,
                chunk_id_start,
                source_file.clone(),
                heading.clone(),
                base_offset + body_offset,
                base_offset + body_offset + body_bytes.len(),
                body,
            );
            continue;
        }
        let mut start = 0usize;
        while start < body_bytes.len() {
            let end = (start + target_chars).min(body_bytes.len());
            // Snap end to a UTF-8 boundary.
            let mut snap = end;
            while snap > start && !body.is_char_boundary(snap) {
                snap -= 1;
            }
            let slice = &body[start..snap];
            push_chunk(
                &mut chunks,
                ctx_name,
                chunk_id_start,
                source_file.clone(),
                heading.clone(),
                base_offset + body_offset + start,
                base_offset + body_offset + snap,
                slice,
            );
            if snap == body_bytes.len() {
                break;
            }
            start = snap.saturating_sub(overlap_chars);
            // Snap start to char boundary.
            while start > 0 && !body.is_char_boundary(start) {
                start -= 1;
            }
        }
    }

    chunks
}

fn push_chunk(
    chunks: &mut Vec<Chunk>,
    ctx_name: &str,
    id_start: usize,
    source_file: Option<PathBuf>,
    section_path: Option<String>,
    start: usize,
    end: usize,
    text: &str,
) {
    if text.trim().is_empty() {
        return;
    }
    let id = format!("{}:{:04}", ctx_name, id_start + chunks.len());
    let token_count = text.len().div_ceil(CHARS_PER_TOKEN);
    chunks.push(Chunk {
        id,
        byte_range: (start, end),
        token_count,
        section_path,
        source_file,
    });
}

// Slices in this function are all on UTF-8 boundaries: `split_inclusive('\n')`
// yields whole lines, and `hashes` is a count of leading ASCII `#` chars.
#[allow(clippy::string_slice)]
fn split_on_headings(text: &str) -> Vec<(Option<String>, &str)> {
    let mut sections = Vec::new();
    let mut current_heading: Option<String> = None;
    let mut current_start = 0usize;
    let mut last_split = 0usize;

    for (idx, line) in text.split_inclusive('\n').enumerate() {
        let _ = idx;
        let line_start = current_start;
        current_start += line.len();
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix('#') {
            // Match `#`, `##`, `###` followed by space.
            let hashes = rest.chars().take_while(|c| *c == '#').count() + 1;
            let after = &trimmed[hashes..];
            if hashes <= 6 && after.starts_with(' ') {
                if line_start > last_split {
                    sections.push((current_heading.clone(), &text[last_split..line_start]));
                }
                current_heading = Some(after.trim().to_string());
                last_split = line_start;
            }
        }
    }
    if last_split < text.len() {
        sections.push((current_heading, &text[last_split..]));
    }
    if sections.is_empty() {
        sections.push((None, text));
    }
    sections
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn load_file_chunks_and_searches() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        let body = format!(
            "# Intro\n\nThis is a doc.\n\n# Secret\n\nThe secret number is 42, hidden well.\n\n# Outro\n\nThe end.\n",
        );
        f.write_all(body.as_bytes()).unwrap();
        let store = RlmStore::new();
        let ctx = store.load_file(f.path(), "doc").unwrap();
        assert!(!ctx.chunks.is_empty());
        assert!(ctx.total_tokens > 0);

        let bm25 = ctx.search("secret number", 5, SearchMode::Bm25);
        assert!(!bm25.is_empty(), "bm25 should hit the secret chunk");
        let top = ctx.get_chunk_by_id(&bm25[0].chunk_id, 4096).unwrap();
        assert!(top.text.contains("42"));

        let sub = ctx.search("42", 5, SearchMode::Substring);
        assert!(!sub.is_empty(), "substring should find 42");
        assert!(sub[0].preview.contains("42"));
    }

    #[test]
    fn load_directory_walks_and_skips_binaries() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.md"), "# A\n\nhello world apple\n").unwrap();
        std::fs::write(dir.path().join("b.txt"), "banana cherry\n").unwrap();
        // A binary-extension file should be skipped.
        std::fs::write(dir.path().join("c.png"), b"\x89PNG\r\n\x1a\n").unwrap();

        let store = RlmStore::new();
        let ctx = store.load_directory(dir.path(), "repo").unwrap();
        let summary = ctx.summary();
        assert!(summary.n_chunks >= 2);
        let hits = ctx.search("banana", 5, SearchMode::Bm25);
        assert!(!hits.is_empty());
    }

    #[test]
    fn memory_kv_round_trip() {
        let store = RlmStore::new();
        store.store_memory("findings", serde_json::json!({"n": 3}));
        let v = store.retrieve_memory("findings").unwrap();
        assert_eq!(v["n"], 3);
        assert_eq!(store.list_memory_keys(), vec!["findings".to_string()]);
    }

    #[test]
    fn budget_cap_rejects_oversize() {
        let store = RlmStore::with_max_bytes(64);
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(&vec![b'x'; 200]).unwrap();
        let err = store.load_file(f.path(), "x").unwrap_err();
        assert!(format!("{err}").contains("ingest limit"));
    }

    #[test]
    fn regex_search_finds_pattern() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(b"# h\nfoo BLUE_TOKEN_91 bar\n").unwrap();
        let store = RlmStore::new();
        let ctx = store.load_file(f.path(), "doc").unwrap();
        let hits = ctx.search(r"BLUE_TOKEN_\d+", 5, SearchMode::Regex);
        assert_eq!(hits.len(), 1);
        assert!(hits[0].preview.contains("BLUE_TOKEN_91"));
    }

    #[test]
    fn haystack_search_smoke() {
        // Hide a known token in ~50KB of filler and confirm BM25 + substring find it.
        let mut body = String::new();
        for _ in 0..1000 {
            body.push_str("lorem ipsum dolor sit amet ");
        }
        body.push_str("\n\nthe magic phrase is BLUE_TOKEN_91 here\n\n");
        for _ in 0..1000 {
            body.push_str("consectetur adipiscing elit ");
        }
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(body.as_bytes()).unwrap();
        let store = RlmStore::new();
        let ctx = store.load_file(f.path(), "big").unwrap();
        let hits = ctx.search("magic phrase BLUE_TOKEN", 3, SearchMode::Bm25);
        assert!(!hits.is_empty());
        let top = ctx.get_chunk_by_id(&hits[0].chunk_id, 8192).unwrap();
        assert!(top.text.contains("BLUE_TOKEN_91"));
    }
}
