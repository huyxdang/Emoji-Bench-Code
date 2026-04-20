"""Phase 5: scoring for E-CONTINUE predictions.

Scoring is split from inference so saved predictions can be rescored as
the scoring rules evolve without re-spending API calls. The pipeline:

1. ``extract_final_output`` pulls the model's ``Final Output: <symbol>``
   line from the raw continuation text.
2. ``detects_loose`` runs the regex bank — any expressed doubt anywhere
   in the continuation counts.
3. ``detects_strict`` further requires that a doubt marker co-occurs with
   an explicit reference to a specific step number (ideally the bad one).
4. ``classify_outcome`` maps (final, detection) against the two scoring
   targets (``ground_truth_final_output``, ``wrong_branch_final_output``)
   to produce one of six buckets.
5. ``score_prediction`` wraps the above and returns a
   ``ScoredContinuation`` row ready for JSONL emission.

The loose-detection regex bank is deliberately permissive — false
positives are easier to triage by hand than false negatives, and the
``detect_recover`` / ``detect_only`` buckets anchor the rate against
whether the final answer was actually corrected.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Literal


OutcomeBucket = Literal[
    "detect_recover",
    "silent_recovery",
    "detect_only",
    "blind_wrong_branch",
    "off_rails",
    "extraction_failed",
]

OUTCOME_BUCKETS: tuple[OutcomeBucket, ...] = (
    "detect_recover",
    "silent_recovery",
    "detect_only",
    "blind_wrong_branch",
    "off_rails",
    "extraction_failed",
)


# --- Loose-detection regex bank --------------------------------------------

# Each pattern matches a verbalized-doubt marker. Word boundaries and
# punctuation anchors keep common false positives (``waiting``, ``actually
# good``) out. Refinement notes:
# - ``\bwait\b``              "wait, that's wrong" (common)
# - ``actually,``             correction opener, but only with a comma so
#                              ``actually true`` doesn't fire
# - ``let me (re)?(check|verify|reconsider|reconfirm)``
#                              a standard reasoning-model hedge phrase
# - ``should (be|have been|actually)``
#                              the classic "Step N should be X" correction
# - ``\bhmm+\b``               hedge marker
# - ``(i (made|had) a|there('s| is) an?) (mistake|error)``
#                              explicit ownership of an error
_DETECT_LOOSE_PATTERNS: tuple[str, ...] = (
    r"\bwait\b",
    r"\bwrong\b",
    r"\bmistake\b",
    r"\berror\b",
    r"\bincorrect\b",
    r"\binvalid\b",
    r"\bcorrection\b",
    r"\bcorrect(ing|ed)?\b",
    r"\breconsider\b",
    r"\brecheck\b",
    r"\bre-?verify\b",
    r"\boops\b",
    r"\bhmm+\b",
    r"\bactually,",
    r"let me (re)?(check|verify|reconsider|reconfirm)\b",
    r"should (be|have been|actually)\b",
    r"(i (made|had) a|there('s| is) an?) (mistake|error|issue)",
    r"does not match",
    r"doesn't match",
    r"doesn't equal",
    r"does not equal",
    r"isn't correct",
    r"is not correct",
)

DETECT_LOOSE_REGEX: re.Pattern[str] = re.compile(
    "|".join(_DETECT_LOOSE_PATTERNS),
    re.IGNORECASE,
)

# Tokens in the continuation text that explicitly reference a numbered
# step. Used by the strict detector to ask "did the doubt co-occur with a
# reference to any specific step?" Upgrading to "the bad step specifically"
# is a Phase 6 refinement once we see enough real detections.
_STEP_REFERENCE_REGEX = re.compile(r"\bstep\s*(\d+)\b", re.IGNORECASE)

# Width of the co-occurrence window for strict detection.
_STRICT_WINDOW_CHARS: int = 120


# --- Final output extraction ----------------------------------------------


_FINAL_OUTPUT_LINE_REGEX = re.compile(
    r"Final\s*Output\s*[:\-]\s*(.+?)(?:\n|$)",
    re.IGNORECASE,
)


def extract_final_output(text: str) -> str | None:
    """Return the model's final output symbol, or ``None`` if missing.

    Strips whitespace and common Markdown emphasis (``**x**``, ``*x*``,
    backticks). If the continuation contains multiple ``Final Output:``
    lines we take the LAST one, on the assumption that a model that
    revises itself writes the corrected answer last.
    """
    matches = list(_FINAL_OUTPUT_LINE_REGEX.finditer(text))
    if not matches:
        return None
    raw = matches[-1].group(1).strip()
    raw = raw.strip("`")
    # Strip surrounding Markdown emphasis but only if symmetric so we don't
    # accidentally eat a leading `*` that's part of content.
    for wrap in ("**", "*", "_", '"', "'"):
        if raw.startswith(wrap) and raw.endswith(wrap) and len(raw) > 2 * len(wrap):
            raw = raw[len(wrap):-len(wrap)].strip()
    # If the model wrote "Final Output: 🪈." strip a trailing period.
    raw = raw.rstrip(".,")
    if not raw:
        return None
    # Take the first whitespace-delimited token — guards against trailing
    # commentary like "Final Output: 🪈 (the correct answer)".
    first_token = raw.split()[0] if raw.split() else raw
    return first_token or None


# --- Detection signals -----------------------------------------------------


def detects_loose(text: str) -> bool:
    """Any verbalized doubt anywhere in the continuation."""
    return DETECT_LOOSE_REGEX.search(text) is not None


def detects_strict(text: str, *, error_step: int | None = None) -> bool:
    """Verbalized doubt that co-occurs with an explicit step reference.

    If ``error_step`` is provided, require the referenced step to match it
    (within the co-occurrence window). Otherwise any numbered step counts.
    A judge-fallback path can upgrade ambiguous cases later without
    changing this regex.
    """
    for match in DETECT_LOOSE_REGEX.finditer(text):
        start = max(0, match.start() - _STRICT_WINDOW_CHARS)
        end = match.end() + _STRICT_WINDOW_CHARS
        window = text[start:end]
        step_hits = _STEP_REFERENCE_REGEX.findall(window)
        if not step_hits:
            continue
        if error_step is None:
            return True
        if any(int(n) == error_step for n in step_hits):
            return True
    return False


# --- Outcome classification ------------------------------------------------


def classify_outcome(
    *,
    final_output: str | None,
    detected_loose: bool,
    ground_truth_final_output: str,
    wrong_branch_final_output: str,
) -> OutcomeBucket:
    """Five-bucket outcome plus an extraction-failure sixth.

    Decision order:
        final is None                         -> extraction_failed
        detected and final == gt              -> detect_recover
        detected and final != gt              -> detect_only
        not detected and final == gt          -> silent_recovery
        not detected and final == wrong_branch -> blind_wrong_branch
        not detected (anything else)          -> off_rails

    ``detect_only`` deliberately subsumes "detected but ended at the
    wrong-branch symbol" — verbalizing doubt but still propagating is
    different enough from blind propagation that we want it tagged as a
    detection case, not a pure cascade.
    """
    if final_output is None:
        return "extraction_failed"

    if detected_loose:
        if final_output == ground_truth_final_output:
            return "detect_recover"
        return "detect_only"

    if final_output == ground_truth_final_output:
        return "silent_recovery"
    if final_output == wrong_branch_final_output:
        return "blind_wrong_branch"
    return "off_rails"


# --- Public scoring entry point -------------------------------------------


@dataclass(frozen=True)
class ScoredContinuation:
    example_id: str
    difficulty: str
    chain_length_x: int
    prefill_error_step: int
    ground_truth_final_output: str
    wrong_branch_final_output: str

    final_output: str | None
    detected_loose: bool
    detected_strict: bool
    outcome_bucket: OutcomeBucket
    matches_ground_truth: bool
    matches_wrong_branch: bool

    model: str
    provider: str
    mode: str
    raw_continuation_text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "difficulty": self.difficulty,
            "chain_length_x": self.chain_length_x,
            "prefill_error_step": self.prefill_error_step,
            "ground_truth_final_output": self.ground_truth_final_output,
            "wrong_branch_final_output": self.wrong_branch_final_output,
            "final_output": self.final_output,
            "detected_loose": self.detected_loose,
            "detected_strict": self.detected_strict,
            "outcome_bucket": self.outcome_bucket,
            "matches_ground_truth": self.matches_ground_truth,
            "matches_wrong_branch": self.matches_wrong_branch,
            "model": self.model,
            "provider": self.provider,
            "mode": self.mode,
            "raw_continuation_text": self.raw_continuation_text,
        }


_REQUIRED_PREDICTION_FIELDS: tuple[str, ...] = (
    "example_id",
    "difficulty",
    "chain_length_x",
    "prefill_error_step",
    "ground_truth_final_output",
    "wrong_branch_final_output",
    "raw_continuation_text",
    "model",
    "provider",
    "mode",
)


def score_prediction(row: dict[str, Any]) -> ScoredContinuation:
    """Score a single prediction row from ``evaluate_continuation.py``."""
    missing = [field for field in _REQUIRED_PREDICTION_FIELDS if field not in row]
    if missing:
        raise ValueError(
            f"prediction row {row.get('example_id')!r} missing fields: {missing}"
        )

    text = row["raw_continuation_text"]
    final = extract_final_output(text)
    detected_loose_flag = detects_loose(text)
    detected_strict_flag = detects_strict(text, error_step=row["prefill_error_step"])

    gt = row["ground_truth_final_output"]
    wb = row["wrong_branch_final_output"]
    bucket = classify_outcome(
        final_output=final,
        detected_loose=detected_loose_flag,
        ground_truth_final_output=gt,
        wrong_branch_final_output=wb,
    )

    return ScoredContinuation(
        example_id=row["example_id"],
        difficulty=row["difficulty"],
        chain_length_x=row["chain_length_x"],
        prefill_error_step=row["prefill_error_step"],
        ground_truth_final_output=gt,
        wrong_branch_final_output=wb,
        final_output=final,
        detected_loose=detected_loose_flag,
        detected_strict=detected_strict_flag,
        outcome_bucket=bucket,
        matches_ground_truth=(final == gt),
        matches_wrong_branch=(final == wb),
        model=row["model"],
        provider=row["provider"],
        mode=row["mode"],
        raw_continuation_text=text,
    )


# --- Nested judge-backed scoring ------------------------------------------
#
# New headline pipeline (Phase 5b). Three nested metrics that replace the
# regex-only bucket scheme as the primary report:
#
#     detected                    = judge.detected_error
#     detected_and_fixed          = detected AND judge.corrected_step_y
#     detected_fixed_and_right    = detected_and_fixed
#                                    AND validator.derivation_valid
#                                    AND validator.terminal_matches_gt
#
# Each row's nested flags are fully determined by (a) a ``JudgeVerdict`` from
# the judge pass and (b) a ``ValidationResult`` from the Python validator.
# No regex input is used for the nested metrics — the regex bucket system
# stays in place as an orthogonal diagnostic baseline.


@dataclass(frozen=True)
class NestedScoredContinuation:
    example_id: str
    difficulty: str
    chain_length_x: int
    prefill_error_step: int

    detected: bool
    detected_and_fixed: bool
    detected_fixed_and_right: bool

    # Underlying inputs, persisted alongside so downstream analysis can
    # re-derive the booleans without re-running the judge or validator.
    judge_detected_error: bool
    judge_corrected_step_y: bool
    judge_reasoning: str
    validator_parseable: bool
    validator_derivation_valid: bool
    validator_terminal_matches_gt: bool
    validator_first_invalid_step: int | None
    validator_first_discontinuity_step: int | None
    validator_parsed_step_count: int
    validator_reason: str | None
    final_output: str | None

    model: str
    provider: str
    mode: str
    turn_2_level: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "difficulty": self.difficulty,
            "chain_length_x": self.chain_length_x,
            "prefill_error_step": self.prefill_error_step,
            "detected": self.detected,
            "detected_and_fixed": self.detected_and_fixed,
            "detected_fixed_and_right": self.detected_fixed_and_right,
            "judge_detected_error": self.judge_detected_error,
            "judge_corrected_step_y": self.judge_corrected_step_y,
            "judge_reasoning": self.judge_reasoning,
            "validator_parseable": self.validator_parseable,
            "validator_derivation_valid": self.validator_derivation_valid,
            "validator_terminal_matches_gt": self.validator_terminal_matches_gt,
            "validator_first_invalid_step": self.validator_first_invalid_step,
            "validator_first_discontinuity_step": self.validator_first_discontinuity_step,
            "validator_parsed_step_count": self.validator_parsed_step_count,
            "validator_reason": self.validator_reason,
            "final_output": self.final_output,
            "model": self.model,
            "provider": self.provider,
            "mode": self.mode,
            "turn_2_level": self.turn_2_level,
        }


def score_prediction_nested(
    *,
    prediction_row: dict[str, Any],
    judge_verdict: Any,  # emoji_bench.continuation_judge.JudgeVerdict
    validation_result: Any,  # emoji_bench.continuation_validator.ValidationResult
    final_output: str | None,
) -> NestedScoredContinuation:
    """Combine judge + validator outputs into the three nested booleans."""
    detected = bool(judge_verdict.detected_error)
    detected_and_fixed = detected and bool(judge_verdict.corrected_step_y)
    detected_fixed_and_right = (
        detected_and_fixed
        and bool(validation_result.derivation_valid)
        and bool(validation_result.terminal_matches_gt)
    )

    return NestedScoredContinuation(
        example_id=prediction_row["example_id"],
        difficulty=prediction_row["difficulty"],
        chain_length_x=prediction_row["chain_length_x"],
        prefill_error_step=prediction_row["prefill_error_step"],
        detected=detected,
        detected_and_fixed=detected_and_fixed,
        detected_fixed_and_right=detected_fixed_and_right,
        judge_detected_error=bool(judge_verdict.detected_error),
        judge_corrected_step_y=bool(judge_verdict.corrected_step_y),
        judge_reasoning=str(judge_verdict.reasoning),
        validator_parseable=bool(validation_result.parseable),
        validator_derivation_valid=bool(validation_result.derivation_valid),
        validator_terminal_matches_gt=bool(validation_result.terminal_matches_gt),
        validator_first_invalid_step=validation_result.first_invalid_step,
        validator_first_discontinuity_step=validation_result.first_discontinuity_step,
        validator_parsed_step_count=validation_result.parsed_step_count,
        validator_reason=validation_result.reason,
        final_output=final_output,
        model=prediction_row.get("model", ""),
        provider=prediction_row.get("provider", ""),
        mode=prediction_row.get("mode", ""),
        turn_2_level=prediction_row.get("turn_2_level"),
    )


def summarize_nested(
    scored: list[NestedScoredContinuation],
) -> dict[str, Any]:
    """Aggregate the three nested rates overall + per difficulty."""
    total = len(scored)
    if total == 0:
        return {
            "total": 0,
            "detect_rate": 0.0,
            "detect_correct_rate": 0.0,
            "detect_correct_finaloutput_correct_rate": 0.0,
            "mechanical_correct_rate": 0.0,
            "by_difficulty": {},
        }

    n_detect = sum(1 for s in scored if s.detected)
    n_detect_fix = sum(1 for s in scored if s.detected_and_fixed)
    n_detect_fix_right = sum(1 for s in scored if s.detected_fixed_and_right)
    n_mechanical = sum(
        1 for s in scored
        if s.validator_derivation_valid and s.validator_terminal_matches_gt
    )

    by_difficulty: dict[str, dict[str, Any]] = {}
    diff_counts: dict[str, Counter] = {}
    for s in scored:
        d = s.difficulty
        diff_counts.setdefault(d, Counter())
        diff_counts[d]["_total"] += 1
        diff_counts[d]["detect"] += int(s.detected)
        diff_counts[d]["detect_fix"] += int(s.detected_and_fixed)
        diff_counts[d]["detect_fix_right"] += int(s.detected_fixed_and_right)
        diff_counts[d]["mechanical"] += int(
            s.validator_derivation_valid and s.validator_terminal_matches_gt
        )

    for d, c in diff_counts.items():
        n = c["_total"]
        by_difficulty[d] = {
            "total": n,
            "detect_rate": round(c["detect"] / n, 4) if n else 0.0,
            "detect_correct_rate": round(c["detect_fix"] / n, 4) if n else 0.0,
            "detect_correct_finaloutput_correct_rate": (
                round(c["detect_fix_right"] / n, 4) if n else 0.0
            ),
            "mechanical_correct_rate": (
                round(c["mechanical"] / n, 4) if n else 0.0
            ),
        }

    return {
        "total": total,
        "detect_rate": round(n_detect / total, 4),
        "detect_correct_rate": round(n_detect_fix / total, 4),
        "detect_correct_finaloutput_correct_rate": round(n_detect_fix_right / total, 4),
        "mechanical_correct_rate": round(n_mechanical / total, 4),
        "by_difficulty": by_difficulty,
    }
