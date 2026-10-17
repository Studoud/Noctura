import json
import sys
from pathlib import Path

import pandas as pd

from data.validate_clean import main


ROOT = Path(__file__).resolve().parents[1]


def test_validator_counts_and_drops_duplicates_across_chunks(
    tmp_path: Path,
    monkeypatch,
) -> None:
    row = pd.read_csv(ROOT / "data" / "dataset_ml_frequentation_24h_clean.csv").head(1)
    source = tmp_path / "repeated.csv"
    pd.concat([row, row, row], ignore_index=True).to_csv(source, index=False)
    report_dir = tmp_path / "report"

    monkeypatch.setattr(
        sys,
        "argv",
        ["validate_clean.py", str(source), "--out-dir", str(report_dir), "--chunksize", "1"],
    )
    assert main() == 0
    report = json.loads((report_dir / "repeated_report.json").read_text(encoding="utf-8"))
    cleaned = pd.read_csv(report_dir / "repeated_clean.csv")
    assert report["doublons_exacts"] == 2
    assert len(cleaned) == 3

    deduplicated_dir = tmp_path / "deduplicated"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_clean.py",
            str(source),
            "--out-dir",
            str(deduplicated_dir),
            "--chunksize",
            "1",
            "--drop-duplicates",
        ],
    )
    assert main() == 0
    deduplicated = pd.read_csv(deduplicated_dir / "repeated_clean.csv")
    assert len(deduplicated) == 1
