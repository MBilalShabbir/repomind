# RepoMind

## Stop explaining your codebase to AI

RepoMind indexes your repository once, then gives you context-aware answers from your actual code.
Fast, local-first, and built for real developer workflows.

## Quick Start

### Install with pip

```bash
pip install git+https://github.com/MBilalShabbir/repomind.git
```

### Or install with pipx

```bash
pipx install git+https://github.com/MBilalShabbir/repomind.git
```

## Usage

```bash
repomind index .
repomind ask "Where is auth handled?"
```

## What RepoMind Does

- Creates a `.repomind/` index **per repository**
- Works locally with semantic code search by default
- Supports optional AI enhancement with:
  - `ANTHROPIC_API_KEY`
  - `OPENAI_API_KEY`

No key? You still get relevant files and snippets.

## How It Works

1. Scans your repo files
2. Chunks code into searchable context
3. Embeds chunks locally
4. Stores vectors in `.repomind/`
5. Retrieves top-matching snippets for each question
6. Optionally generates AI answers when keys are configured

## Works With AI Tools (Claude, ChatGPT)

Use RepoMind to retrieve grounded context, then paste into your preferred AI tool.

```bash
repomind ask "How does auth flow through middleware?" --format prompt
```

This outputs a structured prompt with context + question, ready to paste into Claude or ChatGPT.

## Optional: Check Your Setup

```bash
repomind doctor
```

## Contributing

PRs are welcome. Keep changes focused, tested, and documented.
