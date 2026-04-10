---
description: Get a high-level overview of the current repository using RepoMind
allowed-tools: Bash
---

Run RepoMind overview on the current project and present a structured summary.

**RepoMind output:**
!`repomind overview . 2>&1`

Based on the output above, present a clear overview:
- **Project purpose** — what this codebase does in 1–2 sentences
- **Key modules** — the main directories and what each is responsible for
- **Important files** — entry points, configuration files, and other critical files
- **Suggested next steps** — what to look at first depending on common tasks (adding a feature, fixing a bug, understanding the data flow)
- If the output includes a warning about no index, tell the user to run `repomind index .` first
