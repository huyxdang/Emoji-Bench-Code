---
pretty_name: Emoji-Bench
task_categories:
- text-generation
language:
- en
---

# emoji-bench-e-continue

This dataset contains prompt-only benchmark instances for Emoji-Bench.

## Schema

- `example_id`: unique row id
- `base_id`: shared id across clean/error variants of the same underlying problem
- `split`: train / validation / test
- `difficulty`: easy / medium / hard / expert
- `condition`: clean or error_injected
- `error_type`: null or an injected error label such as E-RES, E-INV, E-CASC, or E-RECONV
- `has_error`: whether the prompt contains an injected error
- `prompt`: full benchmark prompt
- `actual_step_count`: realized number of derivation steps
- `target_step_count`: requested target length used during generation
- `expected_error_step`: ground-truth step with the injected error, or null on clean rows
- `system_json`: JSON serialization of the underlying formal system
- `system_seed` / `chain_seed` / `error_seed`: generation metadata for reproducibility

## Counts

- total_examples: 100
- split_counts: {"train": 0, "validation": 0, "test": 100}
- difficulty_counts: {"easy": 25, "medium": 25, "hard": 25, "expert": 25}
- condition_counts: {"error_injected": 100}
- error_type_counts: {"E-CONTINUE": 100}
- generator_commit: 033b20f56e9786d29b688e552063f306fee73c31

## Load

```python
from datasets import load_dataset

ds = load_dataset("emoji-bench-e-continue")
print(ds)
```
