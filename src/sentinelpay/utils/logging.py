"""Consistent logging plus a small timer used to track each pipeline step."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Return a logger with the project's format (configured once)."""
    root = logging.getLogger("sentinelpay")
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt="%H:%M:%S"))
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        root.propagate = False
    return logging.getLogger(name if name.startswith("sentinelpay") else f"sentinelpay.{name}")


@contextmanager
def timed(step: str, timings: dict[str, float] | None = None) -> Iterator[None]:
    """Log how long a block takes and optionally record it in ``timings``."""
    log = get_logger("sentinelpay.timing")
    start = time.perf_counter()
    log.info("▶ %s ...", step)
    yield
    elapsed = time.perf_counter() - start
    if timings is not None:
        timings[step] = round(elapsed, 2)
    log.info("✔ %s done in %.1fs", step, elapsed)
