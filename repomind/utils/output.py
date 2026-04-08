"""Terminal output helpers for consistent CLI UX."""

from __future__ import annotations

from dataclasses import dataclass, field

import typer

try:
    from rich.console import Console
    from rich.rule import Rule
except ImportError:  # pragma: no cover
    Console = None  # type: ignore[assignment]
    Rule = None  # type: ignore[assignment]


@dataclass(slots=True)
class CliOutput:
    """Helpers for structured CLI output formatting."""

    width: int = 72
    use_color: bool = True
    _console: Console | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize optional rich console."""
        self._console = Console() if self.use_color and Console is not None else None

    def section(self, title: str, style: str = "bold cyan") -> None:
        """Print a top-level titled section."""
        if self._console and Rule is not None:
            self._console.print()
            self._console.print(Rule(title, style=style))
            return
        typer.echo(f"\n{title}")
        typer.echo("=" * min(self.width, max(12, len(title))))

    def subsection(self, title: str, style: str = "bold white") -> None:
        """Print a subsection heading."""
        if self._console and Rule is not None:
            self._console.print()
            self._console.print(Rule(title, style=style))
            return
        typer.echo(f"\n{title}")
        typer.echo("-" * min(self.width, max(10, len(title))))

    def kv(self, label: str, value: str) -> None:
        """Print a key-value row."""
        if self._console:
            self._console.print(f"[bold]{label}:[/bold] {value}")
            return
        typer.echo(f"{label}: {value}")

    def bullet(self, value: str) -> None:
        """Print a bullet row."""
        if self._console:
            self._console.print(f"- {value}")
            return
        typer.echo(f"- {value}")

    def info(self, message: str) -> None:
        """Print informational line."""
        if self._console:
            self._console.print(message)
            return
        typer.echo(message)

    def success(self, message: str) -> None:
        """Print success line."""
        if self._console:
            self._console.print(f"[green][OK][/green] {message}")
            return
        typer.echo(f"[OK] {message}")

    def warning(self, message: str) -> None:
        """Print warning line."""
        if self._console:
            self._console.print(f"[yellow][WARN][/yellow] {message}")
            return
        typer.echo(f"[WARN] {message}")

    def error(self, message: str) -> None:
        """Print error line."""
        if self._console:
            self._console.print(f"[red][ERROR][/red] {message}")
            return
        typer.echo(f"[ERROR] {message}")
