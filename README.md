# RepoMind

> **Stop explaining your codebase to AI.**
> Index once. Ask better questions forever.

RepoMind is a fast, developer-first CLI that indexes your repository and gives you context-aware answers using semantic code search, with optional AI generation when API keys are configured.

## Why RepoMind

Most AI workflows break because context is missing.
RepoMind fixes that by grounding every answer in your actual code.

- No more copy-pasting random files
- No more “hallucinated” architecture guesses
- No more wasting tokens on irrelevant context

## Features

- Semantic repository indexing with local embeddings
- FAISS-powered retrieval for fast, relevant code search
- Two modes:
  - **Free Mode**: relevant files + snippets (no API keys needed)
  - **Premium Mode**: answer + explanation + AI-ready prompt
- Optional LLM integration:
  - Anthropic (priority when key is present)
  - OpenAI (fallback when Anthropic key is missing)
- Incremental indexing with `--update`
- Repo-scoped isolation (`.repomind/` per repository)
- Clean CLI UX with readable sections and prompt export mode

## Quick Start

```bash
pip install -e .
repomind index .
repomind ask "Where is auth token validation implemented?"
```

That’s it. You can now ask architecture and implementation questions grounded in your code.

## Demo

> Placeholder: add GIF/screenshot here (`docs/demo.gif`)

Example commands:

```bash
repomind ask "How does request auth flow through middleware?"
repomind ask "What changed in indexing logic?" --top-k 8
repomind ask "Give me a migration plan" --format prompt
```

## How It Works

RepoMind keeps the flow simple:

1. **Scan** your repository (skips noisy directories/files)
2. **Chunk** code into searchable segments
3. **Embed** chunks with local sentence-transformers
4. **Store** vectors in FAISS under `.repomind/`
5. **Retrieve** top-k relevant snippets for each question
6. **Enhance** with LLM output only if an API key is configured

## Modes

### Free Mode (default)
If no API key is present:

- semantic search only
- returns relevant files + code snippets
- no LLM calls attempted

### Premium Mode (auto)
If API key(s) are present:

- `ANTHROPIC_API_KEY` used first
- otherwise `OPENAI_API_KEY`
- returns summarized answer, explanation, and paste-ready prompt

## Configuration

RepoMind reads config from:

1. Environment variables
2. `.repomind/config.toml`
3. Defaults

### Environment Variables

- `ANTHROPIC_API_KEY` (optional)
- `OPENAI_API_KEY` (optional)
- `REPOMIND_EMBEDDING_MODEL` (optional)
- `REPOMIND_MAX_FILE_SIZE_BYTES` (optional, default: `1048576`)

### Example config file

Create `.repomind/config.toml`:

```toml
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
# llm_provider = "anthropic"  # optional hint, auto-detection still applies
# max_file_size_bytes = 1048576
```

## Performance vs Coverage

RepoMind skips very large files by default to stay fast.

- ✅ Faster indexing
- ⚠️ Possible missed context if critical logic is in oversized files

If recall looks incomplete, increase file size limit and re-index:

```bash
export REPOMIND_MAX_FILE_SIZE_BYTES=2097152
repomind index . --update
```

## CLI Commands

```bash
repomind doctor
repomind index .
repomind index . --update
repomind ask "How is rate limiting implemented?" --top-k 6
repomind ask "Generate a fix plan" --format prompt
repomind explain repomind/core/indexer.py
```

## Contributing

Contributions are welcome.

1. Fork the repo
2. Create a branch (`feat/your-change`)
3. Make focused changes with tests/docs where relevant
4. Open a PR with clear context and before/after behavior

Good first contributions:

- better file filtering heuristics
- improved chunk ranking quality
- benchmark scripts for indexing/retrieval
- richer output adapters for editor integrations

## License

MIT
