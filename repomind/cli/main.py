"""Typer CLI entrypoint for RepoMind."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer

from repomind.core.config import ConfigLoader, RepoMindConfig
from repomind.core.doctor import DoctorService
from repomind.core.embeddings import EmbeddingError, SentenceTransformerEmbedder
from repomind.core.indexer import CodeIndexer, IndexProgress
from repomind.core.llm import LLMRouter
from repomind.core.retriever import NO_INDEX_MESSAGE, CodeRetriever, RetrievalResult
from repomind.core.summarizer import SummaryBuilder
from repomind.utils.logging import configure_logging
from repomind.utils.output import CliOutput

logger = logging.getLogger(__name__)
output = CliOutput()

app = typer.Typer(
    name="repomind",
    help="Index your codebase and ask context-aware questions.",
    add_completion=False,
)


def _build_embedder(config: RepoMindConfig) -> SentenceTransformerEmbedder:
    """Create the default local embedding provider from config."""
    return SentenceTransformerEmbedder(model_name=config.embedding_model)


def _load_config() -> RepoMindConfig:
    """Load runtime config and fail gracefully on invalid settings."""
    try:
        return ConfigLoader(project_root=Path.cwd()).load()
    except Exception as exc:  # noqa: BLE001
        output.error(f"Configuration error: {exc}")
        raise typer.Exit(code=1) from exc


def _load_runtime_context() -> tuple[RepoMindConfig, SentenceTransformerEmbedder]:
    """Build shared runtime dependencies for CLI commands."""
    config = _load_config()
    embedder = _build_embedder(config)
    return config, embedder


def _ensure_repo_index_exists(config: RepoMindConfig) -> None:
    """Ensure current repository has a local RepoMind index."""
    if (
        not config.data_dir.exists()
        or not config.index_path.exists()
        or not config.metadata_path.exists()
    ):
        output.error(NO_INDEX_MESSAGE)
        output.info("Next step: run `repomind index .` in your project root.")
        raise typer.Exit(code=1)


def _render_index_progress(progress: IndexProgress) -> None:
    """Render concise indexing progress updates."""
    if progress.total_files > 0:
        output.info(
            f"[{progress.phase}] {progress.message} "
            f"({progress.processed_files}/{progress.total_files} files, "
            f"{progress.processed_chunks} chunks)"
        )
        return
    output.info(f"[{progress.phase}] {progress.message}")


def _warn_free_mode() -> None:
    """Show free-mode warning and upgrade hint."""
    output.warning("No AI provider configured. Showing relevant code instead.")
    output.info("Tip: Set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` to enable Premium mode.")


def _warn_ai_unavailable() -> None:
    """Show warning when keys exist but provider call is unavailable."""
    output.warning("AI provider unavailable. Showing relevant code instead.")
    output.info("Tip: Verify API key validity, network access, and provider SDK installation.")


def _split_ai_answer(answer: str) -> tuple[str, str]:
    """Split AI answer into summary and explanation blocks."""
    text = answer.strip()
    if not text:
        return "No summarized answer returned.", "No explanation returned."

    blocks = [block.strip() for block in text.split("\n\n") if block.strip()]
    if len(blocks) == 1:
        return blocks[0], blocks[0]
    return blocks[0], "\n\n".join(blocks[1:])


def _truncate_text(text: str, max_chars: int = 900) -> str:
    """Truncate long text blocks for cleaner terminal output."""
    stripped = text.strip()
    if len(stripped) <= max_chars:
        return stripped
    return stripped[:max_chars].rstrip() + "\n... [truncated]"


def _build_prompt_only_output(question: str, results: list[RetrievalResult]) -> str:
    """Build a prompt-only payload for copy-pasting into AI tools."""
    context_blocks: list[str] = []
    for idx, item in enumerate(results, start=1):
        meta = item.metadata
        context_blocks.append(
            (
                f"[{idx}] {meta.file_path}:{meta.start_line}-{meta.end_line}\n"
                f"{meta.text.strip()}"
            )
        )
    context_text = "\n\n".join(context_blocks)
    return (
        "Context:\n"
        f"{context_text}\n\n"
        "Question:\n"
        f"{question.strip()}"
    )


@app.callback()
def main(
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable debug logs."),
    ] = False,
) -> None:
    """Initialize CLI application."""
    configure_logging(verbose=verbose)


@app.command()
def index(
    path: Annotated[
        Path,
        typer.Argument(help="Path to the project root to index."),
    ] = Path("."),
    update: Annotated[
        bool,
        typer.Option(
            "--update",
            help="Incrementally re-index only changed files in this repository.",
        ),
    ] = False,
) -> None:
    """Scan a project, generate embeddings, and build a FAISS index."""
    config, embedder = _load_runtime_context()
    indexer = CodeIndexer(config=config, embedder=embedder)

    try:
        output.section("RepoMind Index")
        stats = indexer.index(path, update=update, progress_callback=_render_index_progress)
    except EmbeddingError as exc:
        logger.error("Embedding initialization failed: %s", exc)
        output.error(f"Embedding initialization failed: {exc}")
        raise typer.Exit(code=1) from exc
    except FileNotFoundError as exc:
        logger.error("Index path error: %s", exc)
        output.error(str(exc))
        raise typer.Exit(code=1) from exc
    except Exception as exc:  # noqa: BLE001
        logger.error("Indexing failed: %s", exc)
        output.error(f"Indexing failed: {exc}")
        raise typer.Exit(code=1) from exc

    output.subsection("Summary")
    output.kv("Mode", stats.mode)
    output.kv("Scanned files", str(stats.scanned_files))
    output.kv("Indexed files", str(stats.files_indexed))
    output.kv("Indexed chunks", str(stats.chunks_indexed))
    output.kv("FAISS index", str(stats.index_path))
    output.kv("Metadata", str(stats.metadata_path))
    if update:
        output.kv("Updated files", str(stats.updated_files))
        output.kv("Removed files", str(stats.removed_files))
        output.info("Next step: run `repomind ask \"<your question>\"`.")
    else:
        output.info("Next step: run `repomind ask \"<your question>\"`.")


@app.command()
def ask(
    question: Annotated[
        str,
        typer.Argument(help="Question about the indexed repository."),
    ],
    top_k: Annotated[
        int,
        typer.Option("--top-k", "-k", min=1, help="Number of chunks to retrieve."),
    ] = 5,
    output_format: Annotated[
        str,
        typer.Option(
            "--format",
            help="Output format: 'default' or 'prompt'.",
        ),
    ] = "default",
) -> None:
    """Answer a repository question using semantic retrieval and optional LLMs."""
    normalized_format = output_format.strip().lower()
    if normalized_format not in {"default", "prompt"}:
        output.error("Invalid --format value. Use 'default' or 'prompt'.")
        raise typer.Exit(code=1)

    config, embedder = _load_runtime_context()
    _ensure_repo_index_exists(config)
    retriever = CodeRetriever(config=config, embedder=embedder)

    try:
        results = retriever.retrieve(question=question, top_k=top_k)
    except EmbeddingError as exc:
        logger.error("Embedding error: %s", exc)
        output.error(str(exc))
        raise typer.Exit(code=1) from exc
    except FileNotFoundError as exc:
        output.error(str(exc))
        raise typer.Exit(code=1) from exc
    except Exception as exc:  # noqa: BLE001
        logger.error("Retrieval failed: %s", exc)
        output.error(f"Retrieval failed: {exc}")
        raise typer.Exit(code=1) from exc

    if normalized_format == "prompt":
        typer.echo(_build_prompt_only_output(question=question, results=results))
        return

    summarizer = SummaryBuilder()
    llm_client = LLMRouter(config).resolve()
    ai_answer: str | None = None
    provider: str | None = None
    ai_call_failed = False

    if llm_client is not None:
        try:
            llm_response = llm_client.answer_question(
                question=question,
                contexts=[item.metadata for item in results],
            )
            ai_answer = llm_response.text
            provider = llm_response.provider
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM answer failed, falling back to local summary: %s", exc)
            ai_call_failed = True

    summary = summarizer.build_for_question(
        question=question,
        results=results,
        ai_answer=ai_answer,
        llm_provider=provider,
    )

    output.section("📁 Relevant Files")
    for file_path in summary.relevant_files[:10]:
        output.bullet(file_path)
    if not summary.relevant_files:
        output.warning("No relevant files found in current index.")
        output.info("Try rephrasing your question or increase `--top-k`.")

    if not summary.llm_provider and ai_call_failed:
        _warn_ai_unavailable()
    elif not summary.llm_provider:
        _warn_free_mode()

    output.section("📄 Code Snippets")
    for snippet in summary.snippets[:top_k]:
        output.info("-" * 72)
        output.info(_truncate_text(snippet))
    if not summary.snippets:
        output.warning("No snippets available for this question.")
        output.info("Try running `repomind index --update` and ask again.")

    if summary.llm_provider:
        output.section("🧠 Answer")
        output.kv("Provider", summary.llm_provider)
        summarized_answer, explanation = _split_ai_answer(summary.summary)
        output.subsection("Summary")
        output.info(summarized_answer)

        output.subsection("Explanation")
        output.info(explanation)

        output.section("🤖 Paste into AI")
        output.info(summary.prompt)
    else:
        output.info("Next step: set an API key for AI answers, or keep asking for code retrieval.")


@app.command()
def explain(
    file: Annotated[
        Path,
        typer.Argument(help="Path to a source file relative to current directory."),
    ],
) -> None:
    """Explain file purpose, key functions, and execution flow."""
    config, _ = _load_runtime_context()
    _ensure_repo_index_exists(config)
    summarizer = SummaryBuilder()

    target = file.resolve()
    if not target.exists() or not target.is_file():
        logger.error("File not found: %s", file)
        output.error(f"File not found: {file}")
        raise typer.Exit(code=1)
    if not target.is_relative_to(config.project_root):
        logger.error("File must be inside current repository: %s", file)
        output.error(f"File must be inside current repository: {file}")
        raise typer.Exit(code=1)

    try:
        content = target.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        logger.error("Unable to read file: %s", exc)
        output.error(f"Unable to read file: {exc}")
        raise typer.Exit(code=1) from exc

    base_explanation = summarizer.explain_file_locally(file_path=target, content=content)

    output.section("RepoMind Explain")
    output.kv("File", str(file))
    output.subsection("Local Explanation")
    output.info(base_explanation)

    llm_client = LLMRouter(config).resolve()
    if llm_client is None:
        _warn_free_mode()
        output.info("Next step: set an API key to enable AI-enhanced explanations.")
        return

    try:
        response = llm_client.explain_file(file_path=str(file), content=content)
        output.subsection(f"AI Enhancement ({response.provider})")
        output.info(response.text)
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM explanation failed, using local explanation: %s", exc)
        _warn_ai_unavailable()


@app.command()
def doctor() -> None:
    """Run environment diagnostics and suggest setup improvements."""
    config = _load_config()
    report = DoctorService().inspect(config)

    output.section("RepoMind Doctor:")

    if report.repo_indexed:
        output.info("✅ Repo indexed")
    else:
        output.info("⚠️ Repo not indexed")

    if report.repomind_dir_exists:
        output.info("✅ RepoMind index directory (.repomind/) found")
    else:
        output.info("⚠️ RepoMind index directory (.repomind/) not found")

    if report.faiss_index_present:
        output.info("✅ FAISS index file found")
    else:
        output.info("⚠️ FAISS index file not found")
    if report.metadata_present:
        output.info("✅ Metadata file found")
    else:
        output.info("⚠️ Metadata file not found")

    if report.local_embeddings_ok:
        output.info("✅ Local embeddings working")
    else:
        output.info("⚠️ Local embeddings unavailable")

    if report.openai_key_configured:
        output.info("✅ OpenAI key configured")
    else:
        output.info("⚠️ OpenAI key not configured")

    if report.anthropic_key_configured:
        output.info("✅ Anthropic key configured")
    else:
        output.info("⚠️ Anthropic key not configured")

    output.subsection("Tip")
    if report.openai_key_configured or report.anthropic_key_configured:
        output.info("Add or keep an API key configured for better AI-assisted results.")
    else:
        output.info("Add an API key for better results.")
    if not report.local_embeddings_ok:
        output.info("Install local embeddings support with `pip install sentence-transformers`.")
    if not report.repo_indexed:
        output.info("Run `repomind index` in this repository to create a local index.")


if __name__ == "__main__":
    app()
