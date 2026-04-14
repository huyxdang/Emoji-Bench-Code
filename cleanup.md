# Cleanup Plan — E-CONTINUE-only repo

## Goal

Keep the repo scoped to the active `E-CONTINUE` benchmark. Legacy `E-RECONV` machinery, numeric-relabel experiments, and older evaluator scripts have value as history but clutter the day-to-day codebase.

## Strategy

Make `main` the clean, active-only codebase. Preserve the full pre-cleanup state on an **archive branch** so nothing is lost.

1. **Finish the in-flight xhigh runs** (`gpt-5.4` + `gpt-5.4-mini`, reasoning-effort=xhigh, Shape B L0 only). Judge + score each as it completes.
2. **Archive commit:** on `main`, stage every pending file (pilot artifacts, eval directories, summary JSONs) and land one commit titled something like `archive: full pilot artifacts + legacy pipeline before E-CONTINUE-only cleanup`. Push.
3. **Cut the archive branch from that commit:** `git branch archive-pre-cleanup && git push -u origin archive-pre-cleanup`. Also tag the same commit (`archive-2026-04-14`) so the snapshot is easy to find even if the branch is pruned later.
4. **Stay on `main`** and execute the three cleanup phases below directly on `main`.
5. **Verify:** `uv run pytest -x -q` must stay green, and each CLI (`generate_continuation_dataset`, `evaluate_continuation`, `judge_continuation`, `score_continuation`) must still `--help` and smoke-test.
6. **Push `main`.** The archive branch remains as a long-lived history snapshot; nothing on it will move forward.

_Rationale: future work (more runs, scoring changes, new prompts) keeps landing on `main` without needing a rebase or merge. Anyone who wants the old E-RECONV machinery or the pre-cleanup artifacts checks out `archive-pre-cleanup`._

## Phase 1 — Safe deletions (no import chain touched)

All of the following are NOT reachable from the `E-CONTINUE` transitive closure. Deleting them only drops tests tied to them. No refactor needed.

### `emoji_bench/`

- `eval_cli.py` — shared CLI base for the legacy single-prompt evaluators
- `metric_extract.py` — legacy answer extractor
- `numeric_labels.py` — numeric-relabeling helpers
- `reconvergent_dataset.py` — `E-RECONV` dataset builder
- `reporting.py` — legacy HTML/CSV report renderer

### `scripts/`

- `analyze_evals.py`
- `evaluate_anthropic.py`
- `evaluate_gemini.py`
- `evaluate_model.py`
- `evaluate_openai.py`
- `extract_key_metrics.py`
- `generate_dataset.py`
- `generate_reconvergent_dataset.py`
- `relabel_dataset_numeric.py`
- `run.sh`
- `run_numbers.sh`
- `run_reconv.sh`

Keep: `preview_dataset.py` (still useful for E-CONTINUE rows — verify imports first), the four `*_continuation.py` + `generate_continuation_dataset.py` scripts.

### `tests/`

Drop these (each tests a module that Phase 1 or 2 removes):

- `test_benchmark.py`
- `test_eval_cli.py`
- `test_metric_extract.py`
- `test_numeric_labels.py`
- `test_reconvergent_dataset.py`
- `test_reconvergent_error_injector.py`
- `test_relabel_dataset_numeric.py`
- `test_reporting.py`

Keep but audit after refactor: `test_evaluation.py`, `test_preview_dataset.py`, `test_prompt_formatter.py`, `test_provider_eval.py`. Their underlying modules stay (trimmed) — trim the tests to match if they exercise removed helpers.

Keep every `test_continuation_*.py`, plus the foundation tests: `test_chain_generator.py`, `test_dataset.py`, `test_error_injector.py`, `test_expressions.py`, `test_formatter.py`, `test_generator.py`, `test_integration.py`, `test_interpreter.py`, `test_model_registry.py`, `test_operations.py`, `test_score_prediction_nested.py`, `test_symbols.py`, `test_transforms.py`, `test_turn_2_prompt_levels.py`.

### Root-level cruft

- `index.html`, `index_og.html`, `index2.html` — old E-RECONV visualization pages. Drop from source; any published versions stay in git history under `main`.
- `continue.md` — resume instructions for the xhigh runs. Delete once those runs are done and scored.

## Phase 2 — Detach shared utilities from legacy callers

Currently reachable from the `E-CONTINUE` transitive closure only because `dataset.py` imports `benchmark.py` and `benchmark.py` imports `reconvergent_error_injector.py`:

- `emoji_bench/benchmark.py` — legacy single-instance generator (the E-RECONV equivalent of `continuation_benchmark.py`)
- `emoji_bench/reconvergent_error_injector.py` — the reconvergent injector

The chain to cut:

```
dataset.py  ->  benchmark.py  ->  reconvergent_error_injector.py
```

**Refactor:** open `emoji_bench/dataset.py` and remove anything that needs `generate_benchmark_instance` / `BenchmarkInstance`. The `E-CONTINUE` path uses only `DIFFICULTY_CONFIGS`, `DEFAULT_TARGET_LENGTHS`, `DatasetManifest`, `DatasetVariant`, and `write_dataset` / `push_dataset_to_hub`. The `generate_dataset_records` function and all of the variant-producing helpers (`_variant_supported_by_chain`, `_variant_seed_offsets`, `_error_seed_for_variant`, `_can_generate_variant_instance`) are E-RECONV-era code. Drop them.

Once `dataset.py` is clean, delete:

- `emoji_bench/benchmark.py`
- `emoji_bench/reconvergent_error_injector.py`

Shared-utility modules to keep but trim:

- `emoji_bench/prompt_formatter.py` — `continuation_formatter.py` imports `format_step` from here. Keep the file, verify nothing else in it is unused.
- `emoji_bench/provider_eval.py` — `continuation_provider.py` + `continuation_judge.py` import `make_client`, `resolve_api_key`, usage-extraction helpers, and `GEMINI_API_BASE_URL` / `MISTRAL_API_URL` / `_api_ssl_context`. **Aggressive option:** rename to `emoji_bench/provider_clients.py` and move just those utilities; drop `build_*_request_options`, `request_prediction`, `_request_openai_prediction` etc. **Lazy option:** leave the file as-is since it's only referenced internally.
- `emoji_bench/evaluation.py` — `evaluate_continuation.py` + `judge_continuation.py` use `append_jsonl` + `load_jsonl_records`. Keep those, drop `normalize_prediction`, `score_prediction`, `scored_prediction_to_dict`, `summarize_scores` (legacy regex + bucket scoring ≠ the new nested scorer).

## Phase 3 — Artifacts and docs

- `artifacts/emoji-bench-e-reconv-1000/` — legacy E-RECONV dataset. Still referenced in the README as the published HF dataset. **Keep.**
- `artifacts/emoji-bench-mixed-2000-numbers/` — already gitignored. Leave alone.
- `artifacts/emoji-bench-e-continue-pilot/` — current pilot data. **Keep.**
- `artifacts/evals/emoji-bench-e-continue-pilot-*` — current pilot evaluations. **Keep.**
- Any `artifacts/evals/emoji-bench-e-reconv-*` from older sessions — check if present; drop if any.

Docs:

- `README.md` — audit for lingering E-RECONV mentions after Phase 1/2. The current README already drops most of this but has a "Why E-CONTINUE?" section that references the legacy machinery; rewrite that paragraph if the machinery is gone.
- `continue.md` — delete once xhigh runs are fully scored.
- `cleanup.md` — delete after cleanup is complete.
- `public/` — check contents; keep the benchmark splash image, drop anything E-RECONV-specific.

## Execution order

1. Finish xhigh runs (in flight).
2. Judge + score xhigh runs.
3. On `main`: one archive commit covering all pending pilot artifacts, untracked eval directories, and any local unstaged work. Push.
4. From that commit, `git branch archive-pre-cleanup && git push -u origin archive-pre-cleanup`. Tag it too.
5. Back on `main`: Phase 1 deletions. Commit. Run tests. Push.
6. Phase 2 refactor + deletions. Commit. Run tests. Push.
7. Phase 3 doc + artifact pass. Commit. Run tests. Push.
8. Smoke-test each CLI (`--help`, plus a 2-row live run if budget allows).
9. Confirm `archive-pre-cleanup` on GitHub has the full pre-cleanup tree; `main` has the E-CONTINUE-only tree.

## Risks / things to watch

- Phase 2's `dataset.py` refactor is the only code-level risk. Any `test_dataset.py` test that exercises E-RECONV variant generation will break; trim those tests too rather than keeping a second dataset generator around.
- `scripts/preview_dataset.py` may import legacy modules — verify after Phase 1. If it does, either update it to use the E-CONTINUE row shape or drop the script.
- The `artifacts/emoji-bench-e-reconv-1000/README.md` references E-RECONV fields — harmless to leave since it's published dataset metadata, but note the path diverges from the active code.
- Committing the full set of `artifacts/evals/emoji-bench-e-continue-pilot-*` directories grows the repo; that's already locked in by the archive commit. Cleanup doesn't undo it.
