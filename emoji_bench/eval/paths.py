from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


PREDICTIONS_FILENAME = "predictions.jsonl"
SUMMARY_FILENAME = "summary.json"
JUDGE_FILENAME = "judge.jsonl"
SCORES_FILENAME = "scores.jsonl"
NESTED_SCORES_FILENAME = "nested_scores.jsonl"
SCORE_SUMMARY_FILENAME = "score_summary.json"


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue
        if value and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value


def resolve_dataset_split_path(raw_path: str | Path, *, split: str = "test") -> Path:
    path = Path(raw_path)
    if path.is_dir():
        path = path / f"{split}.jsonl"
    return path


def resolve_predictions_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_dir():
        path = path / PREDICTIONS_FILENAME
    return path


def resolve_dataset_path_from_summary(summary_path: Path) -> Path | None:
    if not summary_path.exists():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    input_path = summary.get("input_path")
    if not input_path:
        return None
    path = resolve_dataset_split_path(input_path)
    if not path.exists():
        return None
    return path


def resolve_dataset_path(
    *,
    explicit: str | Path | None,
    summary_path: Path,
) -> Path | None:
    if explicit is not None:
        return resolve_dataset_split_path(explicit)
    return resolve_dataset_path_from_summary(summary_path)


@dataclass(frozen=True)
class EvalArtifactPaths:
    output_dir: Path
    predictions_path: Path
    summary_path: Path


@dataclass(frozen=True)
class JudgeArtifactPaths:
    output_dir: Path
    predictions_path: Path
    summary_path: Path
    judge_path: Path


@dataclass(frozen=True)
class ScoreArtifactPaths:
    output_dir: Path
    predictions_path: Path
    summary_path: Path
    judge_path: Path
    scores_path: Path
    nested_scores_path: Path
    score_summary_path: Path


def build_eval_artifact_paths(output_dir: str | Path) -> EvalArtifactPaths:
    resolved_output_dir = Path(output_dir)
    return EvalArtifactPaths(
        output_dir=resolved_output_dir,
        predictions_path=resolved_output_dir / PREDICTIONS_FILENAME,
        summary_path=resolved_output_dir / SUMMARY_FILENAME,
    )


def build_judge_artifact_paths(predictions_path: str | Path) -> JudgeArtifactPaths:
    resolved_predictions_path = Path(predictions_path)
    output_dir = resolved_predictions_path.parent
    return JudgeArtifactPaths(
        output_dir=output_dir,
        predictions_path=resolved_predictions_path,
        summary_path=output_dir / SUMMARY_FILENAME,
        judge_path=output_dir / JUDGE_FILENAME,
    )


def build_score_artifact_paths(
    predictions_path: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> ScoreArtifactPaths:
    resolved_predictions_path = Path(predictions_path)
    resolved_output_dir = (
        Path(output_dir) if output_dir is not None else resolved_predictions_path.parent
    )
    return ScoreArtifactPaths(
        output_dir=resolved_output_dir,
        predictions_path=resolved_predictions_path,
        summary_path=resolved_output_dir / SUMMARY_FILENAME,
        judge_path=resolved_output_dir / JUDGE_FILENAME,
        scores_path=resolved_output_dir / SCORES_FILENAME,
        nested_scores_path=resolved_output_dir / NESTED_SCORES_FILENAME,
        score_summary_path=resolved_output_dir / SCORE_SUMMARY_FILENAME,
    )
