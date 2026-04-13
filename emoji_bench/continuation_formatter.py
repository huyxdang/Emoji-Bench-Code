from __future__ import annotations

from emoji_bench.chain_types import DerivationChain
from emoji_bench.expressions import expr_to_str_with_system
from emoji_bench.formatter import format_system_for_prompt
from emoji_bench.prompt_formatter import format_step
from emoji_bench.types import FormalSystem


CONTINUATION_TASK_PROMPT = """\
Simplify the expression above step by step. Use this exact format for every step:

Step N: <full expression before> = <full expression after>    [by <rule name>]

where N is the sequential step number, each step rewrites the full expression (not just the changed subpart), and <rule name> is the rule you applied (e.g. "\u2295 table", "definition of \u2297", or a transformation name).

Continue producing steps until the expression is a single symbol. Then, on its own line, state:

Final Output: <single symbol>"""


TURN_2_USER: str = "Please continue."


def format_continuation_turn_1_user(
    system: FormalSystem,
    chain: DerivationChain,
) -> str:
    """Format the first user turn: rules + starting expression + task instructions.

    The chain's starting expression is shown; the derivation steps are NOT
    included here — they belong in the assistant prefill.
    """
    rules = format_system_for_prompt(system)
    start_str = expr_to_str_with_system(chain.starting_expression, system)
    return (
        f'Below is a formal system called "{system.name}".\n\n'
        f"=== RULES ===\n{rules}\n\n"
        f"=== EXPRESSION ===\n{start_str}\n\n"
        f"=== TASK ===\n{CONTINUATION_TASK_PROMPT}"
    )


def format_continuation_prefill(
    chain: DerivationChain,
    cutoff_step: int,
    system: FormalSystem,
) -> str:
    """Format the assistant prefill: steps 1..cutoff_step only.

    The prefill deliberately ends mid-derivation: no trailing "Result:" line,
    no trailing blank line, no "Final Output:" marker. The model's continuation
    must resume on a new step line.
    """
    if cutoff_step < 1:
        raise ValueError(f"cutoff_step must be >= 1, got {cutoff_step}")
    if cutoff_step > len(chain.steps):
        raise ValueError(
            f"cutoff_step {cutoff_step} exceeds chain length {len(chain.steps)}"
        )

    start_str = expr_to_str_with_system(chain.starting_expression, system)
    lines = [f"Start: {start_str}", ""]
    for step in chain.steps[:cutoff_step]:
        lines.append(format_step(step, system))
    return "\n".join(lines)
