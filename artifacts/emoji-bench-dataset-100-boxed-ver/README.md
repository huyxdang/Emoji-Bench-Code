---
pretty_name: Emoji-Bench
task_categories:
- text-generation
language:
- en
---

# emoji-bench-e-continue

This dataset contains continuation-benchmark rows for Emoji-Bench.

## Schema

- `example_id`: unique row id
- `base_id`: shared id for the underlying generated formal system
- `split`: dataset split (`test` in the current release)
- `difficulty`: easy / medium / hard / expert
- `error_type`: injected error label (`E-CONTINUE` in the current release)
- `has_prefill_error`: whether the assistant prefill contains the injected error
- `turn_1_user`: rules + expression + formatting instructions
- `turn_1_assistant_prefill`: partial derivation ending on the injected error
- `turn_2_user`: continuation prompt used for the stored condition
- `clean_derivation`: full correct derivation through `Final Output:`
- `ground_truth_final_output`: correct final symbol from the clean chain
- `wrong_branch_final_output`: final symbol reached by blindly continuing from the bad state
- `chain_length_x`: realized clean derivation length
- `prefill_error_step`: step number of the injected error
- `prefill_cutoff_step`: prefill cutoff step for this artifact
- `target_step_count`: requested target length used during generation
- `system_json`: JSON serialization of the formal system
- `system_seed` / `chain_seed` / `error_seed`: generation metadata for reproducibility
- `condition`: benchmark condition label (`error_injected` in the current release)

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
