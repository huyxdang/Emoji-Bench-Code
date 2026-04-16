from __future__ import annotations

import json

from emoji_bench.types import (
    DerivedOperation,
    FormalSystem,
    OperationTable,
    Symbol,
    TransformationRule,
)

# --- JSON Serialization ---


def system_to_dict(system: FormalSystem) -> dict:
    """Convert a FormalSystem to a JSON-serializable dict."""
    return {
        "name": system.name,
        "seed": system.seed,
        "symbols": [s.emoji for s in system.symbols],
        "base_operations": [_op_table_to_dict(op) for op in system.base_operations],
        "derived_operations": [_derived_op_to_dict(d) for d in system.derived_operations],
        "transformations": [_transform_to_dict(t) for t in system.transformations],
    }


def system_from_dict(data: dict) -> FormalSystem:
    """Reconstruct a FormalSystem from a dict."""
    symbols = tuple(Symbol(e) for e in data["symbols"])
    sym_map = {s.emoji: s for s in symbols}

    return FormalSystem(
        name=data["name"],
        seed=data["seed"],
        symbols=symbols,
        base_operations=tuple(
            _op_table_from_dict(d, symbols, sym_map) for d in data["base_operations"]
        ),
        derived_operations=tuple(
            _derived_op_from_dict(d) for d in data["derived_operations"]
        ),
        transformations=tuple(
            _transform_from_dict(d, sym_map) for d in data["transformations"]
        ),
    )


def system_to_json(system: FormalSystem) -> str:
    return json.dumps(system_to_dict(system), ensure_ascii=False, indent=2)


def system_from_json(json_str: str) -> FormalSystem:
    return system_from_dict(json.loads(json_str))


# --- Prompt Formatting ---


def format_system_for_prompt(system: FormalSystem) -> str:
    """Format a formal system as a readable prompt section (Markdown tables).

    This is the primary formatting function. It resolves all internal names
    (like 'op0') to their display symbols (like '⊕').
    """
    return format_system_for_prompt_full(system)


# --- Internal Helpers ---


def _op_table_to_dict(op: OperationTable) -> dict:
    return {
        "name": op.name,
        "symbol_id": op.symbol_id,
        "table": {
            f"{a.emoji},{b.emoji}": r.emoji
            for (a, b), r in op.table.items()
        },
    }


def _op_table_from_dict(
    data: dict,
    symbols: tuple[Symbol, ...],
    sym_map: dict[str, Symbol],
) -> OperationTable:
    table: dict[tuple[Symbol, Symbol], Symbol] = {}
    for key, val in data["table"].items():
        a_str, b_str = key.split(",")
        table[(sym_map[a_str], sym_map[b_str])] = sym_map[val]
    return OperationTable(
        name=data["name"],
        symbol_id=data["symbol_id"],
        symbols=symbols,
        table=table,
    )


def _derived_op_to_dict(dop: DerivedOperation) -> dict:
    return {
        "name": dop.name,
        "symbol_id": dop.symbol_id,
        "template_id": dop.template_id,
        "base_ops": list(dop.base_ops),
        "transform_name": dop.transform_name,
    }


def _derived_op_from_dict(data: dict) -> DerivedOperation:
    return DerivedOperation(
        name=data["name"],
        symbol_id=data["symbol_id"],
        template_id=data["template_id"],
        base_ops=tuple(data["base_ops"]),
        transform_name=data["transform_name"],
    )


def _transform_to_dict(tr: TransformationRule) -> dict:
    return {
        "name": tr.name,
        "mapping": {k.emoji: v.emoji for k, v in tr.mapping.items()},
        "distributes_over": list(tr.distributes_over),
    }


def _transform_from_dict(data: dict, sym_map: dict[str, Symbol]) -> TransformationRule:
    return TransformationRule(
        name=data["name"],
        mapping={sym_map[k]: sym_map[v] for k, v in data["mapping"].items()},
        distributes_over=tuple(data["distributes_over"]),
    )


def _format_op_table(op: OperationTable) -> str:
    """Format an operation table as a Markdown table."""
    syms = op.symbols
    # Header row
    header = f"| {op.symbol_id} | " + " | ".join(s.emoji for s in syms) + " |"
    separator = "|---|" + "|".join("---" for _ in syms) + "|"
    rows = []
    for row_sym in syms:
        cells = " | ".join(op.table[(row_sym, col_sym)].emoji for col_sym in syms)
        rows.append(f"| **{row_sym.emoji}** | {cells} |")

    note = (
        "In the table below, the row is the left operand and the column is the "
        f"right operand. For example, row a and column b means a {op.symbol_id} b."
    )
    return (
        f"Operation {op.symbol_id} (defined by table):\n"
        f"{note}\n\n{header}\n{separator}\n" + "\n".join(rows)
    )


def format_system_for_prompt_full(system: FormalSystem) -> str:
    """Format with resolved base-op symbols in derived operation definitions."""
    parts: list[str] = []

    sym_str = ", ".join(s.emoji for s in system.symbols)
    parts.append(f"Symbols: {{{sym_str}}}")
    parts.append("")

    # Build name->symbol_id mapping for resolving internal names to display symbols
    op_sym: dict[str, str] = {}
    for op in system.base_operations:
        op_sym[op.name] = op.symbol_id
    for op in system.derived_operations:
        op_sym[op.name] = op.symbol_id

    # Base operations
    for op in system.base_operations:
        parts.append(_format_op_table(op))
        parts.append("")

    # Derived operations (with resolved base op symbols)
    for dop in system.derived_operations:
        base_sym_id = op_sym.get(dop.base_ops[0], "[base]") if dop.base_ops else "[base]"
        parts.append(_format_derived_op_resolved(dop, base_sym_id))
        parts.append("")

    # Transformations (with resolved op symbols)
    for tr in system.transformations:
        parts.append(_format_transform_resolved(tr, op_sym))
        parts.append("")

    return "\n".join(parts).rstrip()


def _format_derived_op_resolved(dop: DerivedOperation, base_sym_id: str) -> str:
    """Format a derived operation with the actual base operation symbol."""
    match dop.template_id:
        case "compose_left":
            return (
                f"Derived operation {dop.symbol_id}:\n"
                f"x {dop.symbol_id} y = (x {base_sym_id} y) {base_sym_id} x"
            )
        case "inv_compose":
            return (
                f"Derived operation {dop.symbol_id}:\n"
                f"x {dop.symbol_id} y = {dop.transform_name}(x {base_sym_id} y)"
            )
        case "double_left":
            return (
                f"Derived operation {dop.symbol_id}:\n"
                f"x {dop.symbol_id} y = (x {base_sym_id} x) {base_sym_id} y"
            )
    return f"Derived operation {dop.symbol_id}: {dop.template_id}"


def _format_transform_resolved(
    tr: TransformationRule, op_sym: dict[str, str]
) -> str:
    """Format a transformation rule with resolved operator symbols."""
    lines = [f'Transformation "{tr.name}":']
    for src, dst in tr.mapping.items():
        lines.append(f"  {tr.name}({src.emoji}) = {dst.emoji}")
    if tr.distributes_over:
        for op_name in tr.distributes_over:
            display = op_sym.get(op_name, op_name)
            lines.append(
                f"  Distribution property: {tr.name}(x {display} y) = "
                f"{tr.name}(x) {display} {tr.name}(y)"
            )
    return "\n".join(lines)
