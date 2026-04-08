"""Typer CLI entrypoint for RepoMind."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer

from repomind.core.config import ConfigLoader, RepoMindConfig
from repomind.core.doctor import DoctorService
from repomind.core.embeddings import EmbeddingError, SentenceTransformerEmbedder
from repomind.core.indexer import CodeIndexer
from repomind.core.llm import LLMRouter
from repomind.core.retriever import NO_INDEX_MESSAGE, CodeRetriever
from repomind.core.summarizer import SummaryBuilder
from repomind.utils.logging import configure_logging

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="repomind",
    help="Index your codebase and ask context-aware questions.",
    add_completion=False,
)


def _build_embedder(config: RepoMindConfig) -> SentenceTransformerEmbedder:
    """Create the default local embedding provider from config."""
    return SentenceTransformerEmbedder(model_name=config.embedding_model)


def _ensure_repo_index_exists(config: RepoMindConfig) -> None:
    """Ensure current repository has a local RepoMind index."""
    if (
        not config.data_dir.exists()
        or not config.index_path.exists()
        or not config.metadata_path.exists()
    ):
        typer.echo(NO_INDEX_MESSAGE)
        raise typer.Exit(code=1)


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
    config = ConfigLoader(project_root=Path.cwd()).load()
    embedder = _build_embedder(config)
    indexer = CodeIndexer(config=config, embedder=embedder)

    try:
        stats = indexer.index(path, update=update)
    except EmbeddingError as exc:
        logger.error("Embedding initialization failed: %s", exc)
        raise typer.Exit(code=1) from exc
    except Exception as exc:  # noqa: BLE001
        logger.error("Indexing failed: %s", exc)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Indexed files: {stats.files_indexed}")
    typer.echo(f"Indexed chunks: {stats.chunks_indexed}")
    typer.echo(f"FAISS index: {stats.index_path}")
    typer.echo(f"Metadata: {stats.metadata_path}")
    if update:
        typer.echo(f"Updated files: {stats.updated_files}")
        typer.echo(f"Removed files: {stats.removed_files}")


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
) -> None:
    """Answer a repository question using semantic retrieval and optional LLMs."""
    config = ConfigLoader(project_root=Path.cwd()).load()
    _ensure_repo_index_exists(config)
    embedder = _build_embedder(config)
    retriever = CodeRetriever(config=config, embedder=embedder)
    summarizer = SummaryBuilder()

    try:
        results = retriever.retrieve(question=question, top_k=top_k)
    except Exception as exc:  # noqa: BLE001
        logger.error("Retrieval failed: %s", exc)
        raise typer.Exit(code=1) from exc

    llm_client = LLMRouter(config).resolve()
    ai_answer: str | None = None
    provider: str | None = None

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

    summary = summarizer.build_for_question(
        question=question,
        results=results,
        ai_answer=ai_answer,
        llm_provider=provider,
    )

    typer.echo("Relevant files:")
    for file_path in summary.relevant_files[:10]:
        typer.echo(f"- {file_path}")

    if summary.llm_provider:
        typer.echo(f"\nLLM provider: {summary.llm_provider}")

    typer.echo("\nSummarized answer:")
    typer.echo(summary.summary)

    typer.echo("\nContext snippets:")
    for snippet in summary.snippets[:top_k]:
        typer.echo("-" * 80)
        typer.echo(snippet)

    typer.echo("\nReady-to-paste AI prompt:")
    typer.echo(summary.prompt)


@app.command()
def explain(
    file: Annotated[
        Path,
        typer.Argument(help="Path to a source file relative to current directory."),
    ],
) -> None:
    """Explain file purpose, key functions, and execution flow."""
    config = ConfigLoader(project_root=Path.cwd()).load()
    _ensure_repo_index_exists(config)
    summarizer = SummaryBuilder()

    target = file.resolve()
    if not target.exists() or not target.is_file():
        logger.error("File not found: %s", file)
        raise typer.Exit(code=1)
    if not target.is_relative_to(config.project_root):
        logger.error("File must be inside current repository: %s", file)
        raise typer.Exit(code=1)

    try:
        content = target.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        logger.error("Unable to read file: %s", exc)
        raise typer.Exit(code=1) from exc

    base_explanation = summarizer.explain_file_locally(file_path=target, content=content)

    llm_client = LLMRouter(config).resolve()
    if llm_client is None:
        typer.echo(base_explanation)
        return

    try:
        response = llm_client.explain_file(file_path=str(file), content=content)
        typer.echo(base_explanation)
        typer.echo(f"\nAI Enhancement ({response.provider}):\n")
        typer.echo(response.text)
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM explanation failed, using local explanation: %s", exc)
        typer.echo(base_explanation)


@app.command()
def doctor() -> None:
    """Run environment diagnostics and suggest setup improvements."""
    config = ConfigLoader(project_root=Path.cwd()).load()
    report = DoctorService().inspect(config)

    embeddings_label = "✅" if report.local_embeddings_ok else "❌"
    faiss_label = "✅" if report.faiss_ok else "❌"

    typer.echo(f"Local embeddings: {embeddings_label}")
    typer.echo(f"FAISS: {faiss_label}")
    typer.echo(
        "OpenAI key: configured"
        if report.openai_key_configured
        else "OpenAI key: missing"
    )
    typer.echo(
        "Anthropic key: configured"
        if report.anthropic_key_configured
        else "Anthropic key: missing"
    )

    if report.openai_key_configured and not report.openai_sdk_ok:
        typer.echo("Tip: Install the OpenAI SDK with `pip install openai`.")
    if report.anthropic_key_configured and not report.anthropic_sdk_ok:
        typer.echo("Tip: Install the Anthropic SDK with `pip install anthropic`.")
    if not report.local_embeddings_ok:
        typer.echo(
            "Tip: Install local embeddings support with `pip install sentence-transformers`."
        )
    if report.local_embeddings_ok and report.faiss_ok:
        typer.echo("RepoMind core features are ready.")


if __name__ == "__main__":
    app()
