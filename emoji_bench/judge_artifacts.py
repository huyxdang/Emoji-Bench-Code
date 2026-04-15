from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from emoji_bench.jsonl_io import load_jsonl_records


_FINGERPRINT_FIELDS: tuple[str, ...] = (
    "example_id",
    "prefill_error_step",
    "raw_continuation_text",
)


def prediction_fingerprint(prediction_row: Mapping[str, Any]) -> str:
    missing = [field for field in _FINGERPRINT_FIELDS if field not in prediction_row]
    if missing:
        raise ValueError(
            f"prediction row {prediction_row.get('example_id')!r} missing fields "
            f"for judge fingerprinting: {missing}"
        )

    payload = {field: prediction_row[field] for field in _FINGERPRINT_FIELDS}
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_prediction_fingerprint_map(
    predictions: list[dict[str, Any]],
) -> dict[str, str]:
    fingerprints: dict[str, str] = {}
    for prediction in predictions:
        example_id = prediction.get("example_id")
        if not isinstance(example_id, str) or not example_id:
            raise ValueError("prediction row missing non-empty string example_id")
        if example_id in fingerprints:
            raise ValueError(f"duplicate prediction example_id in predictions.jsonl: {example_id!r}")
        fingerprints[example_id] = prediction_fingerprint(prediction)
    return fingerprints


def load_validated_judge_rows(
    path: str | Path,
    *,
    expected_fingerprints: Mapping[str, str],
) -> dict[str, dict[str, Any]]:
    judge_rows: dict[str, dict[str, Any]] = {}
    for row in load_jsonl_records(path):
        example_id = row.get("example_id")
        if not isinstance(example_id, str) or not example_id:
            raise ValueError("judge.jsonl row missing non-empty string example_id")
        if example_id in judge_rows:
            raise ValueError(f"duplicate judgment for example_id {example_id!r} in judge.jsonl")
        expected = expected_fingerprints.get(example_id)
        if expected is None:
            raise ValueError(
                f"judge.jsonl contains unknown example_id {example_id!r} that is not "
                "present in predictions.jsonl"
            )
        fingerprint = row.get("prediction_fingerprint")
        if not isinstance(fingerprint, str) or not fingerprint:
            raise ValueError(
                f"judge.jsonl row {example_id!r} is missing prediction_fingerprint; "
                "rerun scripts/judge_continuation.py with --no-resume to refresh it"
            )
        if fingerprint != expected:
            raise ValueError(
                f"stale judge row for example_id {example_id!r}: prediction_fingerprint "
                "does not match predictions.jsonl. Rerun scripts/judge_continuation.py "
                "with --no-resume."
            )
        judge_rows[example_id] = row
    return judge_rows


def ensure_full_judge_coverage(
    *,
    prediction_ids: set[str],
    judge_rows: Mapping[str, dict[str, Any]],
) -> None:
    missing = sorted(prediction_ids - set(judge_rows))
    if missing:
        preview = ", ".join(repr(example_id) for example_id in missing[:5])
        suffix = "" if len(missing) <= 5 else ", ..."
        raise ValueError(
            f"judge.jsonl coverage incomplete: {len(judge_rows)}/{len(prediction_ids)} rows "
            f"present; missing example_id(s): {preview}{suffix}"
        )
