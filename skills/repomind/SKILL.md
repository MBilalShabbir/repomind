---
name: repomind
description: Load automatically when the user asks about their codebase, wants to index a repo, asks how code works, where something is, or asks about project structure. Use repomind CLI commands to get grounded, accurate answers instead of guessing.
---

# RepoMind — Codebase Search & Q&A

Use the `repomind` CLI to answer any questions about the codebase. Always run these commands instead of reading files manually or guessing.

## Indexing the repo

When the user asks to "index", "scan", "learn", or "map" the repo:

```bash
repomind index .
```

For large repos or after many file changes, use incremental update:

```bash
repomind index . --update
```

## Asking a question

When the user asks how something works, where something is, what calls what, or any code question:

```bash
repomind ask "the user's question here"
```

## Explaining a file

Before editing, reviewing, or summarizing a specific file:

```bash
repomind explain path/to/file.py
```

## Project overview

When the user asks about project structure, architecture, or wants a map of the codebase:

```bash
repomind overview
```

## Checking status

When something isn't working or you need to confirm the index exists:

```bash
repomind doctor
```

## Setup (run once if repomind is not installed)

Before running any repomind command, check if the CLI is available:

```bash
command -v repomind >/dev/null 2>&1 || pip install repomind-cli
```

If pip is not found, try:

```bash
python3 -m pip install repomind-cli
```

## Rules

1. ALWAYS check that `repomind` is installed before running any command (one-liner above).
2. ALWAYS run `repomind ask` before answering any codebase question — never guess.
3. If `repomind ask` returns no results or errors about a missing index, run `repomind index .` first, then retry.
4. Run `repomind explain <file>` before editing any file you haven't seen in this session.
5. All repomind commands must be run from the project root directory.
6. The index is stored in `.repomind/` inside the project — it is local and private.
