---
description: Explain a source file — purpose, key functions/classes, and execution flow
argument-hint: <path/to/file.py>
allowed-tools: Bash
---

Run RepoMind explain on the given file and produce a structured summary.

**File:** $ARGUMENTS

**RepoMind output:**
!`repomind explain "$ARGUMENTS" 2>&1`

Based on the output above, explain this file clearly and concisely:
- **Purpose** — what problem this file solves and its role in the codebase
- **Key symbols** — the most important functions and classes and what they do
- **How it fits** — what calls into this file and what this file depends on
- If the output shows an error, tell the user what went wrong and how to fix it
