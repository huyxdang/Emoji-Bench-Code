---
pretty_name: Emoji-Bench
task_categories:
- text-generation
language:
- en
---

# emoji-bench-e-continue

This dataset contains continuation-benchmark rows for Emoji-Bench.

Each row stores the first user turn, the prefilled assistant prefix ending on the bad step, the full clean derivation, and the scoring targets for the continuation task.

## Schema

- `example_id`: unique row id
- `base_id`: shared id for the underlying generated formal system
- `split`: dataset split (`test` in the current release)
- `difficulty`: easy / medium / hard / expert
- `error_type`: injected error label (`E-CONTINUE` in the current release)
- `turn_1_user`: rules + expression + formatting instructions
- `turn_1_assistant_prefill`: partial derivation ending on the injected error
- `clean_derivation`: full correct derivation through `Final Output:`
- `ground_truth_final_output`: correct final symbol from the clean chain
- `wrong_branch_final_output`: final symbol reached by blindly continuing from the bad state
- `chain_length_x`: realized clean derivation length
- `prefill_error_step`: step number of the injected error
- `target_step_count`: requested target length used during generation
- `system_json`: JSON serialization of the formal system
- `system_seed` / `chain_seed` / `error_seed`: generation metadata for reproducibility

The default Turn 2 user message is `Please continue.` and is applied at evaluation time; prompt-strength variants can be requested without regenerating the dataset.

## Counts

- total_examples: 100
- master_seed: 20260413
- split_counts: {"train": 0, "validation": 0, "test": 100}
- difficulty_counts: {"easy": 25, "medium": 25, "hard": 25, "expert": 25}
- error_type_counts: {"E-CONTINUE": 100}
- target_lengths: {"easy": 6, "medium": 8, "hard": 10, "expert": 14}
- difficulty_configs: {"easy": {"n_symbols": 3, "n_base_ops": 1, "n_derived_ops": 0, "n_transformations": 0}, "medium": {"n_symbols": 4, "n_base_ops": 1, "n_derived_ops": 1, "n_transformations": 1}, "hard": {"n_symbols": 5, "n_base_ops": 2, "n_derived_ops": 1, "n_transformations": 1}, "expert": {"n_symbols": 6, "n_base_ops": 2, "n_derived_ops": 2, "n_transformations": 2}}
- generator_commit: 02168e72d241b9cd805398244cb7edf2903e446e

## Load

```python
from datasets import load_dataset

ds = load_dataset("emoji-bench-e-continue")
print(ds)
```
