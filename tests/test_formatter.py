from emoji_bench.domain.formatter import (
    format_system_for_prompt,
    system_from_json,
    system_to_json,
)
from emoji_bench.domain.generator import generate_system


def test_json_round_trip_easy():
    system = generate_system(n_symbols=3, random_seed=42)
    json_str = system_to_json(system)
    restored = system_from_json(json_str)
    assert restored.name == system.name
    assert restored.seed == system.seed
    assert restored.symbols == system.symbols
    assert restored.base_operations == system.base_operations
    assert restored.transformations == system.transformations


def test_json_round_trip_medium():
    system = generate_system(
        n_symbols=4, n_base_ops=1, n_derived_ops=1, n_transformations=1, random_seed=77
    )
    json_str = system_to_json(system)
    restored = system_from_json(json_str)
    assert restored.name == system.name
    assert restored.symbols == system.symbols
    assert restored.derived_operations == system.derived_operations
    assert restored.transformations == system.transformations


def test_prompt_format_contains_symbols():
    system = generate_system(n_symbols=3, random_seed=42)
    prompt = format_system_for_prompt(system)
    for s in system.symbols:
        assert s.emoji in prompt


def test_prompt_format_contains_table():
    system = generate_system(n_symbols=3, random_seed=42)
    prompt = format_system_for_prompt(system)
    # Should contain Markdown table markers
    assert "|" in prompt
    assert "---" in prompt


def test_prompt_format_contains_op_symbol():
    system = generate_system(n_symbols=3, random_seed=42)
    prompt = format_system_for_prompt(system)
    for op in system.base_operations:
        assert op.symbol_id in prompt


def test_prompt_format_contains_transform():
    system = generate_system(
        n_symbols=4, n_base_ops=1, n_derived_ops=0, n_transformations=1, random_seed=77
    )
    prompt = format_system_for_prompt(system)
    for tr in system.transformations:
        assert tr.name in prompt
        assert "Distribution property" in prompt


def test_prompt_format_explains_table_operand_order():
    system = generate_system(n_symbols=3, random_seed=42)
    prompt = format_system_for_prompt(system)
    op_symbol = system.base_operations[0].symbol_id

    assert "row is the left operand" in prompt
    assert f"row a and column b means a {op_symbol} b" in prompt


def test_prompt_format_derived_ops_use_base_symbol():
    """Derived op definitions should use ⊕ not 'op0'."""
    system = generate_system(
        n_symbols=4, n_base_ops=1, n_derived_ops=1, n_transformations=1, random_seed=77
    )
    prompt = format_system_for_prompt(system)
    # The base op symbol should appear in derived op definitions
    base_sym = system.base_operations[0].symbol_id
    assert base_sym in prompt
    # Internal name should NOT appear in the prompt
    assert "op0" not in prompt


def test_prompt_format_golden_output_fixed_seed():
    system = generate_system(
        n_symbols=4, n_base_ops=1, n_derived_ops=1, n_transformations=1, random_seed=77
    )
    prompt = format_system_for_prompt(system)

    assert prompt == """Symbols: {🪸, 🪩, 🧄, 🍄}

Operation ⊕ (defined by table):
In the table below, the row is the left operand and the column is the right operand. For example, row a and column b means a ⊕ b.

| ⊕ | 🪸 | 🪩 | 🧄 | 🍄 |
|---|---|---|---|---|
| **🪸** | 🧄 | 🍄 | 🪩 | 🧄 |
| **🪩** | 🧄 | 🍄 | 🍄 | 🪸 |
| **🧄** | 🍄 | 🪩 | 🪸 | 🪸 |
| **🍄** | 🪸 | 🧄 | 🪩 | 🪩 |

Derived operation ⊗:
x ⊗ y = (x ⊕ x) ⊕ y

Transformation "inv":
  inv(🪸) = 🪩
  inv(🪩) = 🪸
  inv(🧄) = 🍄
  inv(🍄) = 🧄
  Distribution property: inv(x ⊕ y) = inv(x) ⊕ inv(y)"""
