from pathlib import Path

import pandas as pd
import pytest

from mibici_ml import monitoring


def _monitoring_frame(rows: int = 20) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "hour": pd.date_range("2026-01-01", periods=rows, freq="h"),
            "station_id": [1, 2] * (rows // 2),
        }
    )
    for column in monitoring.MONITORING_NUMERICAL_COLUMNS:
        frame[column] = range(rows)
    for column in monitoring.MONITORING_CATEGORICAL_COLUMNS:
        if column != "station_id":
            frame[column] = [0, 1] * (rows // 2)
    return frame


def test_deterministic_sample_is_bounded_and_reproducible():
    frame = _monitoring_frame()

    first = monitoring.deterministic_sample(frame, 7)
    second = monitoring.deterministic_sample(frame, 7)

    assert len(first) == 7
    pd.testing.assert_frame_equal(first, second)


def test_summarize_quality_uses_station_hour_as_key():
    frame = _monitoring_frame()
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)

    result = monitoring.summarize_quality(frame)

    assert result["rows"] == 21
    assert result["stations"] == 2
    assert result["missing_values"] == 0
    assert result["duplicate_station_hours"] == 1
    assert result["target_positive_rate"]["risk_no_bikes"] == pytest.approx(10 / 21)


def test_summarize_station_coverage_finds_unseen_categories():
    reference = _monitoring_frame(4)
    current = _monitoring_frame(4)
    current["station_id"] = [2, 2, 3, 3]

    result = monitoring.summarize_station_coverage(reference, current)

    assert result["shared_stations"] == 1
    assert result["unseen_in_training"] == [3]
    assert result["unseen_station_rows"] == 2
    assert result["unseen_station_row_share"] == 0.5
    assert result["absent_from_current"] == [1]


def test_extract_drift_metrics_marks_failed_columns():
    class Snapshot:
        def dict(self):
            return {
                "metrics": [
                    {
                        "id": "all",
                        "config": {"type": "evidently:metric_v2:DriftedColumnsCount"},
                        "value": {"count": 1.0, "share": 0.5},
                    },
                    {
                        "id": "one",
                        "config": {
                            "type": "evidently:metric_v2:ValueDrift",
                            "column": "capacity",
                            "method": "psi",
                            "threshold": 0.2,
                        },
                        "value": 0.3,
                    },
                ],
                "tests": [
                    {"metric_config": {"metric_id": "all"}, "status": "PASS"},
                    {"metric_config": {"metric_id": "one"}, "status": "FAIL"},
                ],
            }

    dataset, columns = monitoring.extract_drift_metrics(Snapshot())

    assert dataset == {"count": 1.0, "share": 0.5, "status": "PASS"}
    assert columns[0]["column"] == "capacity"
    assert columns[0]["drift_detected"] is True


def test_ensure_current_source_keeps_training_manifest_untouched(tmp_path, monkeypatch):
    destination = tmp_path / monitoring.P2_MONITORING_CURRENT_FILE
    training_manifest = tmp_path / "manifest.json"
    training_manifest.write_text('{"data_version": "original"}', encoding="utf-8")
    monkeypatch.setattr(
        monitoring,
        "list_public_drive_folder",
        lambda folder_id: [
            {"name": monitoring.P2_MONITORING_CURRENT_FILE, "file_id": "october-id"}
        ],
    )
    monkeypatch.setattr(
        monitoring,
        "download_public_file",
        lambda file_id, path: Path(path).write_text("october", encoding="utf-8"),
    )

    result = monitoring.ensure_current_source(tmp_path)

    assert destination.exists()
    assert result["file_id"] == "october-id"
    assert result["sha256"]
    assert training_manifest.read_text(encoding="utf-8") == '{"data_version": "original"}'
