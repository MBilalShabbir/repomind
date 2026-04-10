---
description: Ask a natural language question about this codebase using RepoMind
argument-hint: <your question about the codebase>
allowed-tools: Bash
---

Run RepoMind semantic search against the current repository index and use the output to ground your answer.

**Question:** $ARGUMENTS

**RepoMind output:**
!`repomind ask "$ARGUMENTS" 2>&1`

Using only the files, snippets, and summary returned above:
- Answer the question precisely and concisely
- Cite file paths and line numbers when referencing code
- If the output says "No relevant files found", tell the user and suggest running `repomind index .` or rephrasing the question
- Do not fabricate details not present in the output
