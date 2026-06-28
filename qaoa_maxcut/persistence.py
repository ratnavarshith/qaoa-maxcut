"""JSON / CSV persistence for QAOA experiment results."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def _coerce(obj):
    """Numpy → builtins, recursively, for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _coerce(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_coerce(x) for x in obj]
    return obj


def save_results(payload: dict[str, Any], output_dir: str | Path,
                 stem: str = "results") -> dict[str, Path]:
    """Save an arbitrary nested-dict payload to JSON.

    If `payload` contains a top-level `summary_rows` list-of-dicts, that list is
    additionally written as CSV for spreadsheet-friendly inspection.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{stem}.json"
    json_path.write_text(json.dumps(_coerce(payload), indent=2))
    written = {"json": json_path}

    rows = payload.get("summary_rows")
    if isinstance(rows, list) and rows:
        csv_path = output_dir / f"{stem}.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows([_coerce(r) for r in rows])
        written["csv"] = csv_path
    return written


def load_results(path: str | Path) -> dict:
    """Load JSON results; CSV is summary-only and not round-trip safe."""
    path = Path(path)
    if path.suffix != ".json":
        raise ValueError("load_results expects a .json file")
    return json.loads(path.read_text())
