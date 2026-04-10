# SKILL.md — RepoMind

## Description

RepoMind is a local CLI tool that builds a semantic index of a codebase and answers natural language questions about it. Use it to ground your responses in actual code rather than guessing structure, purpose, or behavior.

---

## When to Use

Invoke RepoMind whenever:

- The user asks about **project structure**, modules, or folders
- The user asks about a **specific file** (purpose, functions, dependencies)
- The user asks a **codebase question** (how X works, where Y is handled, what calls Z)
- You need **code context** before editing, refactoring, or explaining
- The user asks you to review or understand an unfamiliar repo

Do **not** invoke RepoMind for:
- General programming questions unrelated to the current project
- Questions answerable from already-visible code in the conversation

---

## Prerequisites

Before running any query command, an index must exist:

```bash
# Check index and environment
repomind doctor

# Build index (run once per repo, or after large changes)
repomind index .

# Update index incrementally (after small changes)
repomind index . --update
```

If `repomind doctor` shows the repo is not indexed, run `repomind index .` first.

---

## Commands

### `repomind overview`

**Use when:** the user asks for a high-level understanding of the project — structure, key modules, important files, or overall purpose.

```bash
repomind overview .
```

**Output:** key modules with descriptions, important files with descriptions, project summary. Uses heuristics when no LLM key is set; uses LLM batch summarization when a key is available.

---

### `repomind ask "<question>"`

**Use when:** the user asks a specific question about how the codebase works.

```bash
repomind ask "Where is authentication handled?"
repomind ask "How does the indexing pipeline work?"
repomind ask "What calls CodeRetriever?"
```

**Options:**
- `--top-k N` — retrieve N chunks (default: 5); increase for broader questions
- `--format prompt` — output a paste-ready prompt block instead of a rendered answer

**Output:** relevant files, code snippets with line ranges, and (if an API key is set) an AI-generated answer.

---

### `repomind explain <file>`

**Use when:** the user asks about a specific file's purpose, functions, or structure.

```bash
repomind explain repomind/core/indexer.py
repomind explain src/auth/middleware.py
```

**Output:** purpose, key functions/classes, execution flow, external dependencies. AI-enhanced when an API key is available.

---

## How to Incorporate Output

1. **Run the appropriate command** using the Bash tool.
2. **Read the output sections** — relevant files, snippets, and summary.
3. **Use the output as grounding context** in your response:
   - Cite file paths and line numbers when referencing code
   - Do not fabricate details not present in the output
   - If the output says "No relevant files found", tell the user and suggest re-indexing or rephrasing

### Suggested patterns

```
# Understand the project before editing
repomind overview .

# Find where something is implemented before modifying it
repomind ask "Where is <feature> implemented?"

# Get file context before explaining or refactoring it
repomind explain <target-file>

# Build a paste-ready context block for a complex task
repomind ask "<question>" --format prompt
```

---

## Output Format Reference

| Section | Content |
|---|---|
| `📁 Key Modules` | Top directories with one-line purpose |
| `📄 Important Files` | Entry points and dense files with descriptions |
| `🧠 Summary` | Project-level or question-level summary |
| `📄 Code Snippets` | `file:start-end` chunks from the FAISS index |
| `Relevant Files` | Ranked file list from semantic retrieval |

---

## Failure Modes and Recovery

| Symptom | Cause | Fix |
|---|---|---|
| `No RepoMind index found` | Index not built | Run `repomind index .` |
| `No relevant files found` | Question too vague or index stale | Rephrase or run `repomind index . --update` |
| `Embedding initialization failed` | `sentence-transformers` not installed | Run `pip install sentence-transformers` |
| `AI provider unavailable` | Key set but SDK missing | Run `pip install anthropic` or `pip install openai` |
| Snippets are off-topic | `--top-k` too low | Retry with `--top-k 10` |

---

## Environment

- Index stored in `.repomind/` inside the project root — never global
- API keys: `ANTHROPIC_API_KEY` (preferred) or `OPENAI_API_KEY` — optional; Free Mode works without them
- Free Mode returns files + snippets only; Premium Mode adds AI-generated answers
