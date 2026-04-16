from emoji_bench.domain import continuation_validator as _continuation_validator
from emoji_bench.domain.continuation_validator import *

_PRIVATE_EXPORTS = ("_parse_expression",)

for _name in _PRIVATE_EXPORTS:
    globals()[_name] = getattr(_continuation_validator, _name)
