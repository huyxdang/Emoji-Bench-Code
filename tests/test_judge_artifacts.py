from __future__ import annotations

import json

import pytest

from emoji_bench.judge_artifacts import (
    build_prediction_fingerprint_map,
    ensure_full_judge_coverage,
    load_validated_judge_rows,
    prediction_fingerprint,
)


def _prediction(*, example_id: str = "cont-000001", text: str = "Step 3: ...") -> dict[str, object]:
    return {
        "example_id": example_id,
        "prefill_error_step": 3,
        "raw_continuation_text": text,
    }


def _write_jsonl(path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_build_prediction_fingerprint_map_rejects_duplicate_prediction_ids():
    predictions = [_prediction(example_id="dup"), _prediction(example_id="dup", text="other")]
    with pytest.raises(ValueError, match="duplicate prediction example_id"):
        build_prediction_fingerprint_map(predictions)


def test_load_validated_judge_rows_rejects_duplicate_judgment_ids(tmp_path):
    prediction = _prediction()
    fingerprints = build_prediction_fingerprint_map([prediction])
    judge_path = tmp_path / "judge.jsonl"
    _write_jsonl(
        judge_path,
        [
            {
                "example_id": "cont-000001",
                "prediction_fingerprint": prediction_fingerprint(prediction),
                "detected_error": True,
                "corrected_step_y": False,
            },
            {
                "example_id": "cont-000001",
                "prediction_fingerprint": prediction_fingerprint(prediction),
                "detected_error": False,
                "corrected_step_y": False,
            },
        ],
    )

    with pytest.raises(ValueError, match="duplicate judgment"):
        load_validated_judge_rows(judge_path, expected_fingerprints=fingerprints)


def test_load_validated_judge_rows_rejects_missing_prediction_fingerprint(tmp_path):
    prediction = _prediction()
    fingerprints = build_prediction_fingerprint_map([prediction])
    judge_path = tmp_path / "judge.jsonl"
    _write_jsonl(
        judge_path,
        [
            {
                "example_id": "cont-000001",
                "detected_error": True,
                "corrected_step_y": False,
            },
        ],
    )

    with pytest.raises(ValueError, match="missing prediction_fingerprint"):
        load_validated_judge_rows(judge_path, expected_fingerprints=fingerprints)


def test_load_validated_judge_rows_rejects_stale_prediction_fingerprint(tmp_path):
    prediction = _prediction()
    fingerprints = build_prediction_fingerprint_map([prediction])
    judge_path = tmp_path / "judge.jsonl"
    _write_jsonl(
        judge_path,
        [
            {
                "example_id": "cont-000001",
                "prediction_fingerprint": prediction_fingerprint(
                    _prediction(text="stale continuation")
                ),
                "detected_error": True,
                "corrected_step_y": False,
            },
        ],
    )

    with pytest.raises(ValueError, match="stale judge row"):
        load_validated_judge_rows(judge_path, expected_fingerprints=fingerprints)


def test_ensure_full_judge_coverage_rejects_partial_rows():
    with pytest.raises(ValueError, match="coverage incomplete: 1/2"):
        ensure_full_judge_coverage(
            prediction_ids={"cont-000001", "cont-000002"},
            judge_rows={"cont-000001": {"example_id": "cont-000001"}},
        )


def test_load_validated_judge_rows_accepts_fully_matched_rows(tmp_path):
    predictions = [
        _prediction(example_id="cont-000001", text="Step 3: first"),
        _prediction(example_id="cont-000002", text="Step 5: second"),
    ]
    fingerprints = build_prediction_fingerprint_map(predictions)
    judge_path = tmp_path / "judge.jsonl"
    _write_jsonl(
        judge_path,
        [
            {
                "example_id": prediction["example_id"],
                "prediction_fingerprint": prediction_fingerprint(prediction),
                "detected_error": True,
                "corrected_step_y": False,
            }
            for prediction in predictions
        ],
    )

    rows = load_validated_judge_rows(judge_path, expected_fingerprints=fingerprints)
    ensure_full_judge_coverage(
        prediction_ids=set(fingerprints),
        judge_rows=rows,
    )

    assert set(rows) == {"cont-000001", "cont-000002"}
