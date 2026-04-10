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
        "Use these tools to get grounded codebase context before answering questions "
        "about code. Prefer repomind_ask for specific questions, repomind_overview "
        "for structural questions, and repomind_explain before editing a file. "
        "Run repomind_index first if the repository has not been indexed yet."
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
