//! Recursive Language Model (RLM) primitives.
//!
//! Implements the RLM pattern from Zhang/Kraska/Khattab (arXiv:2512.24601):
//! the long context lives in an external store that the model accesses
//! through a small set of tools, instead of being stuffed into the prompt.
//!
//! Public surface:
//! - [`store::RlmStore`] — lock-free per-session blob/chunk/memory store.
//! - [`store::RlmContext`] — a single loaded context (file or directory).
//! - [`store::Chunk`] — chunk metadata pointing into a context blob.

pub mod store;

pub use store::{Chunk, ChunkData, ContextSummary, RlmContext, RlmStore, SearchHit, SearchMode};
