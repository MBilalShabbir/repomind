#!/usr/bin/env python3
"""RepoMind MCP server — wraps the repomind CLI as Claude Code tools.

Transport: stdio (registered via install.sh / install.ps1)

Tools exposed:
  repomind_ask      — semantic search + optional AI answer
  repomind_explain  — per-file purpose, symbols, and flow
  repomind_overview — project-level module and file summary
  repomind_index    — build or incrementally update the FAISS index
  repomind_doctor   — environment diagnostics
"""

from __future__ import annotations

import subprocess

from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Server definition
# ---------------------------------------------------------------------------

mcp = FastMCP(
    name="repomind",
    instructions=(
        "You have access to RepoMind, a semantic codebase search engine. "
        "ALWAYS use these tools — do not attempt to read or analyze code yourself.\n\n"
        "Rules:\n"
        "1. Before answering ANY question about the codebase, call repomind_ask.\n"
        "2. When asked to 'index', 'scan', or 'learn' the repo, call repomind_index immediately.\n"
        "3. Before editing or reviewing a file, call repomind_explain on that file.\n"
        "4. For 'how does X work' or 'where is Y' questions, call repomind_ask — never guess.\n"
        "5. For 'show me the structure' or 'overview' questions, call repomind_overview.\n"
        "6. If repomind_ask returns no results or errors, call repomind_index first, then retry.\n\n"
        "Do NOT read files manually, do NOT search with grep, do NOT summarize on your own "
        "when these tools are available. RepoMind's answers are grounded in the actual code."
    ),
)

# ---------------------------------------------------------------------------
# Internal runner
# ---------------------------------------------------------------------------

_NOT_FOUND_MSG = (
    "Error: 'repomind' not found in PATH.\n"
    "Install it with:\n"
    "  pip install git+https://github.com/MBilalShabbir/repomind.git"
)


def _run(args: list[str], *, cwd: str = ".", timeout: int = 120) -> str:
    """Execute a repomind CLI command and return its combined output."""
    try:
        proc = subprocess.run(
            ["repomind", *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()
        if proc.returncode != 0:
            combined = "\n".join(filter(None, [stdout, stderr]))
            return combined or f"repomind exited with code {proc.returncode}"
        return stdout or "(no output)"
    except FileNotFoundError:
        return _NOT_FOUND_MSG
    except subprocess.TimeoutExpired:
        return f"Error: repomind timed out after {timeout}s. Try a more specific question."
    except Exception as exc:  # noqa: BLE001
        return f"Error: {exc}"


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def repomind_ask(question: str, repo_path: str = ".", top_k: int = 5) -> str:
    """Ask a natural language question about the indexed repository.

    Performs semantic retrieval over the FAISS index and returns the most
    relevant files and code snippets. When an API key is configured, also
    returns an AI-generated, citation-grounded answer.

    Args:
        question:  The question to ask about the codebase.
        repo_path: Absolute or relative path to the repository root.
        top_k:     Number of code chunks to retrieve (default 5).
                   Increase to 10–15 for broad or cross-cutting questions.
    """
    return _run(["ask", question, "--top-k", str(top_k)], cwd=repo_path)


@mcp.tool()
def repomind_explain(file_path: str, repo_path: str = ".") -> str:
    """Explain a specific source file.

    Returns the file's purpose, public functions and classes, execution flow,
    and external dependencies. AI-enhanced when an API key is available.

    Args:
        file_path: Path to the file relative to the repository root
                   (e.g. "src/auth/middleware.py").
        repo_path: Absolute or relative path to the repository root.
    """
    return _run(["explain", file_path], cwd=repo_path)


@mcp.tool()
def repomind_overview(repo_path: str = ".") -> str:
    """Get a high-level overview of the repository.

    Returns key modules with one-line descriptions, important entry-point
    files with descriptions, and a project-level summary. Uses LLM batch
    summarization when an API key is available; falls back to heuristics.

    Args:
        repo_path: Absolute or relative path to the repository root.
    """
    return _run(["overview", "."], cwd=repo_path)


@mcp.tool()
def repomind_index(repo_path: str = ".", update: bool = False) -> str:
    """Build or update the RepoMind semantic index for a repository.

    Must be run at least once before repomind_ask or repomind_overview will
    return results. Use update=True to re-index only changed files after
    incremental edits.

    Args:
        repo_path: Absolute or relative path to the repository root.
        update:    If True, re-index only changed/new files (faster).
    """
    args = ["index", "."]
    if update:
        args.append("--update")
    return _run(args, cwd=repo_path, timeout=600)


@mcp.tool()
def repomind_doctor(repo_path: str = ".") -> str:
    """Diagnose the RepoMind setup for a repository.

    Reports: index presence, FAISS index file, metadata file, local embedding
    availability, and API key configuration. Run this when a query returns
    unexpected results or before using the tool on a new machine.

    Args:
        repo_path: Absolute or relative path to the repository root.
    """
    return _run(["doctor"], cwd=repo_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="stdio")
