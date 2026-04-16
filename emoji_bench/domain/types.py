from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Symbol:
    emoji: str

    def __repr__(self) -> str:
        return self.emoji


@dataclass(frozen=True)
class OperationTable:
    name: str
    symbol_id: str  # e.g. "⊕", "⊗"
    symbols: tuple[Symbol, ...]
    table: dict[tuple[Symbol, Symbol], Symbol]

    def __repr__(self) -> str:
        header = f"OperationTable('{self.name}', {self.symbol_id})"
        syms = self.symbols
        # Column headers
        col_w = max(len(s.emoji) for s in syms) + 1
        hdr = self.symbol_id.ljust(col_w) + "".join(s.emoji.ljust(col_w) for s in syms)
        rows = []
        for row_sym in syms:
            cells = "".join(
                self.table[(row_sym, col_sym)].emoji.ljust(col_w)
                for col_sym in syms
            )
            rows.append(row_sym.emoji.ljust(col_w) + cells)
        return header + "\n" + hdr + "\n" + "\n".join(rows)


@dataclass(frozen=True)
class DerivedOperation:
    name: str
    symbol_id: str  # e.g. "⊗"
    template_id: str  # "compose_left", "inv_compose", "double_left"
    base_ops: tuple[str, ...]  # names of base operations referenced
    transform_name: str | None  # name of transform (for inv-based templates)

    def __repr__(self) -> str:
        return f"DerivedOperation('{self.name}', {self.symbol_id}, template={self.template_id})"


@dataclass(frozen=True)
class TransformationRule:
    name: str
    mapping: dict[Symbol, Symbol]
    distributes_over: tuple[str, ...]  # names of operations

    def __repr__(self) -> str:
        arrows = ", ".join(f"{k}→{v}" for k, v in self.mapping.items())
        return f"TransformationRule('{self.name}': {arrows})"


@dataclass(frozen=True)
class FormalSystem:
    name: str
    seed: int
    symbols: tuple[Symbol, ...]
    base_operations: tuple[OperationTable, ...]
    derived_operations: tuple[DerivedOperation, ...]
    transformations: tuple[TransformationRule, ...]

    def __repr__(self) -> str:
        syms = "{" + ", ".join(s.emoji for s in self.symbols) + "}"
        ops = [op.symbol_id for op in self.base_operations] + [
            op.symbol_id for op in self.derived_operations
        ]
        trans = [t.name for t in self.transformations]
        parts = [
            f"FormalSystem('{self.name}'",
            f"seed={self.seed}",
            f"symbols={syms}",
            f"ops=[{', '.join(ops)}]",
        ]
        if trans:
            parts.append(f"transforms=[{', '.join(trans)}]")
        return ", ".join(parts) + ")"
