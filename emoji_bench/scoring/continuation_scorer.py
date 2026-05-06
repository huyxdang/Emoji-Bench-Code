"""Deterministic scoring for E-CONTINUE predictions.

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
    This remains a regex-only diagnostic signal; the headline metric is
    final-answer correctness.
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


def summarize_final_answer_only(
    scored: list[ScoredContinuation],
) -> dict[str, Any]:
    """Aggregate final-answer correctness overall and per difficulty."""
    total = len(scored)
    if total == 0:
        return {
            "total": 0,
            "final_answer_correct_rate": 0.0,
            "by_difficulty": {},
        }

    n_final_correct = sum(1 for s in scored if s.matches_ground_truth)
    by_difficulty: dict[str, dict[str, Any]] = {}
    diff_counts: dict[str, Counter] = {}
    for s in scored:
        d = s.difficulty
        diff_counts.setdefault(d, Counter())
        diff_counts[d]["_total"] += 1
        diff_counts[d]["final_correct"] += int(s.matches_ground_truth)

    for d, c in diff_counts.items():
        n = c["_total"]
        by_difficulty[d] = {
            "total": n,
            "final_answer_correct_rate": round(c["final_correct"] / n, 4) if n else 0.0,
        }

    return {
        "total": total,
        "final_answer_correct_rate": round(n_final_correct / total, 4),
        "by_difficulty": by_difficulty,
    }
