"""Small file helpers for JSON reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write ``payload`` as pretty JSON, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False, default=str) + "\n")
    return path


def read_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON report."""
    return json.loads(Path(path).read_text())
