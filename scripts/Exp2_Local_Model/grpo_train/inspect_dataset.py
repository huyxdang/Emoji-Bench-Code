"""Quick inspection of the GRPO dataset at ../ft_dataset/SAD."""

from datasets import load_from_disk

ds = load_from_disk("../ft_dataset/SAD")

print(ds)
print()

for split in ds:
    print(f"=== split: {split} ===")
    print(f"  num_rows: {len(ds[split])}")
    print(f"  columns:  {ds[split].column_names}")
    print(f"  features: {ds[split].features}")
    print()
    print("  --- example[0] ---")
    ex = ds[split][0]
    for k, v in ex.items():
        print(f"  {k}: {v}")
    print()
