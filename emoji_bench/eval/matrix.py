from __future__ import annotations

from pathlib import Path


def matrix_variant(mode: str) -> str:
    if mode == "prefill":
        return "B"
    if mode == "single_turn":
        return "C"
    raise ValueError(f"Unsupported continuation mode: {mode}")


def matrix_cell(mode: str, turn_2_level: int) -> str:
    return f"{matrix_variant(mode)}-L{turn_2_level}"


def model_output_slug(model_key: str, reasoning_effort: str | None) -> str:
    slug = model_key.replace("/", "-")
    if reasoning_effort is None:
        return slug
    return f"{slug}-reasoning-{reasoning_effort}"


def default_eval_output_dir(
    model_key: str,
    mode: str,
    *,
    reasoning_effort: str | None = None,
    turn_2_level: int = 0,
) -> Path:
    slug = model_output_slug(model_key, reasoning_effort)
    return Path("artifacts") / "evals" / f"{slug}-{matrix_cell(mode, turn_2_level)}"
