# GPT-5.2 vs GPT-5.4 nano Check

Date: 2026-05-03

Both runs completed all 100 Emoji-Bench rows and used the same headline setup:

- mode: `prefill`
- matrix cell: `B-L0`
- turn-2 prompt: `Please continue.`
- scoring: deterministic final-answer-only via `--ignore-judge`
- max output tokens: `128000`
- OpenAI reasoning effort: `xhigh`

Results:

| Model | Final-answer correct | Easy | Medium | Hard | Expert |
|---|---:|---:|---:|---:|---:|
| `gpt-5.2-reasoning-xhigh` | 58% | 64% | 64% | 48% | 56% |
| `gpt-5.4-nano-reasoning-xhigh` | 80% | 80% | 92% | 68% | 80% |

Head-to-head on the same 100 examples:

- both correct: 43
- GPT-5.4 nano only correct: 37
- GPT-5.2 only correct: 15
- neither correct: 5

Token behavior was very different:

| Model | Total output tokens | Total reasoning tokens |
|---|---:|---:|
| `gpt-5.2-reasoning-xhigh` | 489,808 | 459,648 |
| `gpt-5.4-nano-reasoning-xhigh` | 1,349,567 | 1,315,466 |

Interpretation note: under the no-hint final-answer metric, GPT-5.4 nano outperformed GPT-5.2, but it also used much more reasoning/output budget in practice. Phrase this as a benchmark-specific result, not a general model-quality claim.
