"""Tests for JSON / CSV persistence."""
import json
import numpy as np

from qaoa_maxcut.persistence import save_results, load_results


def test_save_load_roundtrip(tmp_path):
    payload = {
        "config": {"n": 6, "p_max": 3},
        "summary_rows": [
            {"p": 1, "approximation_ratio": 0.7},
            {"p": 2, "approximation_ratio": 0.85},
        ],
        "depths": {"1": {"params": np.array([0.1, 0.2])}},
    }
    paths = save_results(payload, tmp_path)
    assert paths["json"].exists()
    assert paths["csv"].exists()  # summary_rows triggers CSV

    loaded = load_results(paths["json"])
    assert loaded["config"]["n"] == 6
    # numpy was coerced to a plain list
    assert loaded["depths"]["1"]["params"] == [0.1, 0.2]


def test_save_no_summary_rows_no_csv(tmp_path):
    payload = {"config": {"n": 6}}
    paths = save_results(payload, tmp_path)
    assert "json" in paths
    assert "csv" not in paths


def test_csv_has_header(tmp_path):
    payload = {
        "summary_rows": [
            {"p": 1, "ratio": 0.5},
            {"p": 2, "ratio": 0.7},
        ],
    }
    paths = save_results(payload, tmp_path)
    text = paths["csv"].read_text()
    assert text.splitlines()[0].strip() == "p,ratio"
