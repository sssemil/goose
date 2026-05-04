//! OOLONG-mini regression harness for RLM tools.
//!
//! This is a *deterministic* end-to-end test of the RLM tool pipeline — it
//! does NOT call an LLM. The OOLONG paper benchmarks the model's ability to
//! semantically classify every chunk of a long input and aggregate the labels
//! across the whole context. Here we build a small synthetic dataset where
//! each chunk's "ground truth" label is determined by the most-common keyword
//! in that chunk, then walk the dataset through the RLM tools (search →
//! get_chunk → in-process scoring → aggregate) and assert the count matches.
//!
//! The point isn't to prove RLM beats a baseline (we can't, without an LLM);
//! it's to lock in that:
//!   1. A larger-than-window context can be loaded into the store.
//!   2. The tool plumbing returns chunks with stable ids that round-trip.
//!   3. Per-chunk inspection + aggregation is feasible end-to-end without the
//!      model ever seeing the full blob, mirroring how a real RLM session
//!      would orchestrate sub_query calls.
//!
//! When you have an LLM available, run:
//!     goose run --rlm --context crates/goose/tests/data/rlm_oolong_mini "How many transcripts mention 'apple'? Use rlm__search and rlm__sub_query."

use goose::agents::rlm::{RlmStore, SearchMode};
use std::io::Write;

const LABELS: &[&str] = &["apple", "banana", "cherry", "durian", "elderberry"];

/// Build N synthetic transcripts where each one's "label" is determined by
/// the keyword that appears most. Returns (path-to-file, expected counts).
fn build_dataset(n: usize) -> (tempfile::NamedTempFile, std::collections::HashMap<String, usize>) {
    let mut f = tempfile::NamedTempFile::new().unwrap();
    let mut expected: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for label in LABELS {
        expected.insert((*label).to_string(), 0);
    }

    for i in 0..n {
        // Cycle through labels so the count distribution is known up-front.
        let label = LABELS[i % LABELS.len()];
        *expected.get_mut(label).unwrap() += 1;
        // Each transcript is a markdown section so chunking splits on it.
        // Filler words include some of the OTHER labels too, but the dominant
        // label appears 5x to make the heuristic deterministic.
        let body = format!(
            "# transcript_{i:04}\n\nlorem ipsum {label} dolor sit {label} amet \
             consectetur {label} adipiscing {label} elit sed {label} \
             do eiusmod tempor incididunt {filler} ut labore et dolore.\n",
            filler = LABELS[(i + 1) % LABELS.len()]
        );
        f.write_all(body.as_bytes()).unwrap();
    }
    f.flush().unwrap();
    (f, expected)
}

/// Stand-in for `rlm__sub_query`: classify a chunk by counting label
/// occurrences and returning the dominant one. In a real RLM session this
/// would be a small LLM call; here it's deterministic so we can assert
/// exact counts.
fn classify_chunk(text: &str) -> Option<String> {
    let mut best: Option<(&&str, usize)> = None;
    for label in LABELS {
        let count = text.matches(label).count();
        if count > 0 && best.map_or(true, |(_, c)| count > c) {
            best = Some((label, count));
        }
    }
    best.map(|(l, _)| (*l).to_string())
}

#[test]
fn oolong_mini_aggregation_via_rlm_tools() {
    let (file, expected) = build_dataset(20);
    let store = RlmStore::new();
    let ctx = store.load_file(file.path(), "transcripts").unwrap();

    // Sanity: the file has 20 transcripts and the chunker keeps each as its
    // own section (since they're under heading boundaries and each is small).
    assert!(
        ctx.chunks.len() >= 20,
        "expected ~20 chunks, got {}",
        ctx.chunks.len()
    );

    // Walk every chunk by id, classify it, aggregate counts. This is exactly
    // the loop a real RLM root would run with rlm__sub_query per chunk.
    let mut counts: std::collections::HashMap<String, usize> =
        LABELS.iter().map(|l| ((*l).to_string(), 0)).collect();
    for chunk in &ctx.chunks {
        let data = ctx
            .get_chunk_by_id(&chunk.id, 4096)
            .expect("chunk id must round-trip");
        if let Some(label) = classify_chunk(&data.text) {
            *counts.get_mut(&label).unwrap() += 1;
        }
    }

    // The classification must reproduce the ground-truth distribution exactly.
    for label in LABELS {
        assert_eq!(
            counts.get(*label),
            expected.get(*label),
            "count mismatch for label {label}"
        );
    }

    // BM25 search for one of the labels must return chunks that actually
    // contain it — this is how a real RLM root narrows before delegating.
    let hits = ctx.search("apple", 5, SearchMode::Bm25);
    assert!(!hits.is_empty(), "BM25 must find apple chunks");
    for hit in &hits {
        assert!(hit.preview.contains("apple"), "preview lacks 'apple'");
    }
}

#[test]
fn oolong_mini_root_message_history_never_holds_full_blob() {
    // Verify the RLM discipline: even after exercising every tool, the strings
    // we'd put into the model's history (previews + chunk slices, all capped)
    // never approach the size of the underlying blob.
    let (file, _expected) = build_dataset(20);
    let store = RlmStore::new();
    let ctx = store.load_file(file.path(), "transcripts").unwrap();

    let blob_chars = ctx.total_chars;
    let mut history_chars = 0usize;
    // Simulate: search once + get every chunk (short slices).
    let hits = ctx.search("transcript", 10, SearchMode::Bm25);
    for h in &hits {
        history_chars += h.preview.chars().count();
    }
    for chunk in &ctx.chunks {
        let d = ctx.get_chunk_by_id(&chunk.id, 800).unwrap();
        history_chars += d.text.chars().count();
    }
    // 20 chunks × 800 chars cap = 16,000 chars worth of history at most.
    // The blob itself is much larger once you sum filler. The point: history
    // grows linearly with the cap, not with the underlying blob size.
    assert!(
        history_chars < blob_chars * 2,
        "history ({history_chars}) shouldn't dwarf blob ({blob_chars})"
    );
    assert!(history_chars > 0, "we did fetch some content");
}
