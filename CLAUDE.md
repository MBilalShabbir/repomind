# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install in editable mode (development)
pip install -e .

# Install all dependencies
pip install -r requirements.txt

# Run the CLI
repomind doctor
repomind overview .
repomind index .
repomind index . --update
repomind ask "How does auth flow?"
repomind ask "How does auth flow?" --format prompt
repomind explain path/to/file.py
```

There are no test files in this repository currently.

## Architecture

RepoMind is a CLI tool that creates a per-repository semantic index and answers natural language questions about code. The package entry point is `repomind.cli.main:app` (Typer).

### Data flow

**Indexing** (`repomind index .`):
1. `FileScanner` walks the repo, filtering out binary files, ignored dirs, and large files
2. `CodeChunker` splits file content into overlapping character-bounded chunks (1200 chars, 200 overlap)
3. `SentenceTransformerEmbedder` encodes chunks via a local HuggingFace model (`all-MiniLM-L6-v2`)
4. FAISS `IndexFlatIP` stores L2-normalized vectors; metadata saved as JSONL; file fingerprints saved as JSON manifest
5. All artifacts go into `.repomind/` inside the project root (never global state)

**Querying** (`repomind ask "..."`):
1. Question is embedded with the same local model
2. `CodeRetriever` does inner-product search against the FAISS index
3. Results are optionally enriched by `LLMRouter` → `AnthropicClient` or `OpenAIClient`
4. `SummaryBuilder` assembles the final output

**Incremental update** (`repomind index . --update`):
- Compares SHA-256 fingerprints of current files against the stored manifest
- Only re-embeds changed/new files; reuses existing vectors for unchanged files

### Module layout

| Path | Role |
|------|------|
| `repomind/cli/main.py` | All Typer commands (`index`, `ask`, `explain`, `doctor`, `overview`) |
| `repomind/core/config.py` | `RepoMindConfig` dataclass; reads from `.repomind/config.toml` then env vars |
| `repomind/core/indexer.py` | `CodeIndexer`, `FileScanner`, `CodeChunker`, `ChunkMetadata`, incremental logic |
| `repomind/core/embeddings.py` | `SentenceTransformerEmbedder` — lazy-loaded, batched, log-suppressed |
| `repomind/core/faiss_store.py` | `require_faiss()` guard — raises descriptively if faiss-cpu is absent |
| `repomind/core/retriever.py` | `CodeRetriever` — loads FAISS + JSONL, returns `RetrievalResult` list |
| `repomind/core/llm.py` | `LLMRouter`, `AnthropicClient`, `OpenAIClient` — optional premium AI layer |
| `repomind/core/summarizer.py` | `SummaryBuilder` — formats snippets, prompts, and AI answers for output |
| `repomind/core/doctor.py` | `DoctorService` — environment diagnostics |
| `repomind/utils/output.py` | `CliOutput` — all Rich-based terminal rendering |
| `repomind/utils/logging.py` | `configure_logging()` — `--verbose` flag wires up debug logs |

### Configuration

Config resolution order (env vars win over file, file wins over default):

| Setting | Env var | Config key | Default |
|---------|---------|------------|---------|
| Embedding model | `REPOMIND_EMBEDDING_MODEL` | `embedding_model` | `sentence-transformers/all-MiniLM-L6-v2` |
| Max file size | `REPOMIND_MAX_FILE_SIZE_BYTES` | `max_file_size_bytes` | 1 MB |
| OpenAI key | `OPENAI_API_KEY` | — | None |
| Anthropic key | `ANTHROPIC_API_KEY` | — | None |

Config file lives at `.repomind/config.toml` inside the project being indexed.

### LLM provider selection

`LLMRouter.resolve()` tries Anthropic first, then OpenAI. No API key = Free Mode (retrieval only). Keys are never required — the tool degrades gracefully.
