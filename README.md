# RepoMind

## Stop explaining your codebase to AI

RepoMind creates a **brain per repo**.
Index once, then ask natural questions and get grounded answers from your actual code.

## Quick Start

```bash
pip install git+https://github.com/MBilalShabbir/repomind.git
repomind index .
repomind ask "Where is auth handled?"
```

## What RepoMind Does

- Builds a local semantic index in `.repomind/` (inside the current repo)
- Retrieves the most relevant files and snippets for your question
- Optionally upgrades to AI-generated answers when API keys are set

No global state. No cross-repo context leaks.

## Example Output

```text
📁 Relevant Files
- app/auth/middleware.py
- app/auth/service.py

📄 Code Snippets
- app/auth/middleware.py:12-48
  def validate_token(...)
  ...

🧠 Answer
Auth is validated in middleware before route handlers.

🤖 Paste into AI
Context:
...
Question:
Where is auth handled?
```

## Free vs Premium

### Free Mode (default)
- No API key required
- Semantic retrieval only
- Returns relevant files + code snippets

### Premium Mode (auto)
- Uses `ANTHROPIC_API_KEY` first, then `OPENAI_API_KEY`
- Adds summarized answer + explanation + paste-ready prompt

## Commands

```bash
repomind doctor
repomind index .
repomind index . --update
repomind ask "How does auth flow?"
repomind ask "How does auth flow?" --format prompt
repomind explain path/to/file.py
```

## Notes

- `.repomind/` is created per project
- If no index exists, run: `repomind index .`
- If no API key exists, RepoMind still works in Free Mode
