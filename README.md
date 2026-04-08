# RepoMind

RepoMind is a production-focused CLI tool that indexes a repository with local embeddings and lets you ask context-aware questions using semantic retrieval, with optional OpenAI/Anthropic enhancement.

## Features

- `repomind index [PATH]`
  - Recursively scans files
  - Skips `.git`, `node_modules`, `venv`, `__pycache__`, `.repomind`
  - Chunks code into overlapping segments
  - Generates local embeddings
  - Persists FAISS index + metadata in `.repomind/` in the current repository
  - Supports incremental updates with `--update` (changed/new/deleted files only)

- `repomind ask "<question>"`
  - Retrieves top-k relevant chunks
  - Auto-detects best available LLM (`OPENAI_API_KEY` then `ANTHROPIC_API_KEY`)
  - Prints relevant file paths, summarized answer, context snippets, and a ready-to-paste AI prompt

- `repomind explain <file>`
  - Produces local structured explanation (purpose, functions/classes, flow, dependencies)
  - Enhances output with optional LLM summary if keys are configured

- `repomind doctor`
  - Environment diagnostics for embeddings, FAISS, API keys, and optional SDKs

## Installation

### Local editable install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

### Requirements-based install

```bash
pip install -r requirements.txt
```

## Configuration

RepoMind reads configuration from:

1. Environment variables (highest priority)
2. `.repomind/config.toml`
3. Defaults

Supported settings:

- `REPOMIND_EMBEDDING_MODEL` (default: `sentence-transformers/all-MiniLM-L6-v2`)
- `REPOMIND_LLM_PROVIDER` (`openai` or `anthropic`, optional)
- `OPENAI_API_KEY` (optional)
- `ANTHROPIC_API_KEY` (optional)

Example `.repomind/config.toml`:

```toml
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
llm_provider = "openai"
```

## Usage

```bash
repomind doctor
repomind index .
repomind index --update
repomind ask "Where is auth token validation implemented?" --top-k 6
repomind explain repomind/core/indexer.py
```

## Storage

Artifacts are written to `.repomind/`:

- `index.faiss`
- `metadata.jsonl`

## Notes

- First run may take time to download embedding models.
- `ask` and `explain` still work without API keys, using local summarization.
