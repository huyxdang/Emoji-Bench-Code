<p align="center">
  <img src="public/emoji-bench.png" alt="Emoji-Bench" width="100%" />
</p>

# Emoji-Bench

Emoji-Bench tests whether a model continues a derivation correctly after its
prefilled prior answer contains a deliberately wrong step.

Each example is a 3-turn conversation:

1. User gives a generated formal system, an expression, and step-format rules.
2. Assistant prefill contains derivation steps ending on an injected error.
3. User says `Please continue.`

The current headline metric is simple:

`final_answer_correct_rate` = extracted `Final Output:` matches the stored
ground-truth final symbol.

## Setup

Requirements: Python 3.11+.

```bash
pip install -r requirements.txt
```

Or with `uv`:

```bash
uv pip install -e ".[dev,hf,openai,anthropic]"
```

Set provider keys as needed:

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GEMINI_API_KEY=...
export MISTRAL_API_KEY=...
export XAI_API_KEY=...
```

Run tests:

```bash
pytest
```

## Generate Dataset

```bash
python scripts/generate_continuation_dataset.py \
  --count 100 \
  --output-dir artifacts/emoji-bench-dataset-100
```

Output:

```text
artifacts/emoji-bench-dataset-100/
├── test.jsonl
├── manifest.json
└── README.md
```

Each row stores:

- `turn_1_user`
- `turn_1_assistant_prefill`
- `clean_derivation`
- `ground_truth_final_output`
- `wrong_branch_final_output`
- `system_json`
- `system_seed`, `chain_seed`, `error_seed`

## Run B Evals

`run.sh` runs the checked-in headline slice only:

- B: native 3-message prefill delivery
- L0: `Please continue.`
- L1: `Please continue. Double-check any step you're unsure about.`
- 13 configured models

```bash
./run.sh artifacts/emoji-bench-dataset-100 -- --max-concurrent 8
```

For each successful model, artifacts are written to:

```text
artifacts/evals/<model>-B-L0/
├── predictions.jsonl
├── scores.jsonl
├── score_summary.json
└── summary.json
```

## Run One Model

```bash
python scripts/evaluate_continuation.py \
  artifacts/emoji-bench-dataset-100 \
  --model claude-haiku-4-5 \
  --mode prefill \
  --turn-2-prompt-level 0

python scripts/score_continuation.py artifacts/evals/claude-haiku-4-5-B-L0
```

## Plot Results

```bash
python scripts/plot_b_final_answer.py
```

Plot output:

```text
artifacts/plots/b_final_answer_l0.png
```

## Repo Map

- `emoji_bench/domain/` - formal systems, expressions, derivation chains
- `emoji_bench/dataset/` - dataset generation and error injection
- `emoji_bench/eval/` - artifact paths, matrix naming, shared runner
- `emoji_bench/providers/` - OpenAI, Anthropic, Gemini, and Mistral transports
- `emoji_bench/scoring/` - deterministic scoring
- `scripts/` - dataset, eval, score, preview, and plot CLIs
- `run.sh` - B batch runner

## Notes

- Gemini artifacts checked into this repo were produced through OpenRouter,
  not the Google Gemini API.
- Results may differ from Kaggle Benchmark because this repo controls model
  reasoning settings and max output tokens.

## License

MIT. See `LICENSE`.
