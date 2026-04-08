"""Logging utilities for RepoMind."""

from __future__ import annotations

import logging


def configure_logging(verbose: bool = False) -> None:
    """Configure package-wide logging.

    Args:
        verbose: Enable debug logging when True.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
