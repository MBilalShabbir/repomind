---
name: repomind
description: Load automatically when the user asks about their codebase, wants to index a repo, asks how code works, where something is, or asks about project structure. Use repomind commands to get grounded, accurate answers instead of guessing.
---

# RepoMind — Codebase Search & Q&A

RepoMind builds a semantic index of the codebase and answers questions grounded in actual code.

## Step 0: Ensure repomind is installed (run once per session if needed)

Check if already installed before attempting to install:

```bash
python3 -c "import repomind" 2>/dev/null && echo "repomind ready" || (python3 -m pip install repomind-cli -q && echo "repomind installed")
```

Run all commands using `python3 -m repomind.cli.main` — this works immediately without needing PATH changes.

## Indexing the repo

When the user asks to "index", "scan", "learn", or "map" the repo:

```bash
python3 -m repomind.cli.main index .
```

For incremental update after file changes:

```bash
python3 -m repomind.cli.main index . --update
```

## Asking a question

When the user asks how something works, where something is, what calls what, or any code question:

```bash
python3 -m repomind.cli.main ask "the user's question here"
```

## Explaining a file

Before editing, reviewing, or summarizing a specific file:

```bash
python3 -m repomind.cli.main explain path/to/file.py
```

## Project overview

When the user asks about project structure, architecture, or wants a map of the codebase:

```bash
python3 -m repomind.cli.main overview
```

## Checking status

When something isn't working or you need to confirm the index exists:

```bash
python3 -m repomind.cli.main doctor
```

## Rules

1. Run the Step 0 install check only if a repomind command fails with "command not found" or "No module named repomind".
2. Use `python3 -m repomind.cli.main` for all commands — never rely on `repomind` being in PATH.
3. ALWAYS run `ask` before answering any codebase question — never guess from memory.
4. If `ask` returns no results or errors about a missing index, run `index .` first, then retry.
5. Run `explain <file>` before editing any file you haven't seen in this session.
6. All commands must be run from the project root directory.
