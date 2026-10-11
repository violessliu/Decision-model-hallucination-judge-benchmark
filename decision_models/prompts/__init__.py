from __future__ import annotations

from . import (
    context_inconsistency,
    factual_contradiction,
    factual_fabrication,
    instruction_inconsistency,
    logical_inconsistency,
)


PROMPTS = {
    "factual_contradiction": factual_contradiction,
    "factual_fabrication": factual_fabrication,
    "context_inconsistency": context_inconsistency,
    "logical_inconsistency": logical_inconsistency,
    "instruction_inconsistency": instruction_inconsistency,
}
