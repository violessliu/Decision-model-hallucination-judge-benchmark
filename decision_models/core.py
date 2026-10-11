from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from .prompts import PROMPTS


@dataclass(frozen=True)
class CategorySpec:
    name: str
    question: str
    positive: str
    negative: str


CATEGORIES: dict[str, CategorySpec] = {
    name: CategorySpec(
        name=name,
        question=prompt.QUESTION,
        positive=prompt.POSITIVE,
        negative=prompt.NEGATIVE,
    )
    for name, prompt in PROMPTS.items()
}


MODEL_ALIASES = {
    "jev": "jev-latest",
    "openai-decisions": "gpt-6-luna",
    "pplx": "perplexity/pplx-decider-v1.1-27b",
    "d1": "d1:free",
    "clef": "clef",
    "clef-flash": "clef-flash",
    "cml": "cml",
    "luna": "gpt-6-luna",
    "sol": "gpt-6-sol",
}


@dataclass
class JudgeInput:
    category: str
    response: str
    instruction: str = ""
    context: Any = ""
    reference: Any = ""
    sample_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.category not in CATEGORIES:
            valid = ", ".join(CATEGORIES)
            raise ValueError(f"Unknown category {self.category!r}; choose one of: {valid}")
        if not self.response.strip():
            raise ValueError("response must not be empty")

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        category: str | None = None,
    ) -> "JudgeInput":
        chosen_category = category or str(payload.get("category", ""))
        instruction = payload.get("instruction", payload.get("input", ""))
        return cls(
            category=chosen_category,
            instruction=str(instruction or ""),
            context=payload.get("context", ""),
            reference=payload.get("reference", ""),
            response=str(payload.get("response", "")),
            sample_id=str(payload["id"]) if payload.get("id") is not None else None,
            metadata=dict(payload.get("metadata") or {}),
        )

    def state(self) -> dict[str, Any]:
        state: dict[str, Any] = {"candidate_response": self.response}
        if self.instruction:
            state["instruction"] = self.instruction
        if self.context not in (None, "", {}, []):
            state["context"] = self.context
        if self.reference not in (None, "", {}, []):
            state["reference"] = self.reference
        return state

    def render(self) -> str:
        parts = [f"CATEGORY:\n{self.category}"]
        if self.instruction:
            parts.append(f"INSTRUCTION:\n{self.instruction}")
        if self.context not in (None, "", {}, []):
            parts.append(f"CONTEXT:\n{_stringify(self.context)}")
        if self.reference not in (None, "", {}, []):
            parts.append(f"REFERENCE:\n{_stringify(self.reference)}")
        parts.append(f"CANDIDATE RESPONSE:\n{self.response}")
        return "\n\n".join(parts)


def build_choice_instrument(item: JudgeInput) -> dict[str, Any]:
    """Return the single source of truth shared by JEV and generative judges."""
    spec = CATEGORIES[item.category]
    return {
        "state": item.state(),
        "instructions": spec.question,
        "criteria": {
            "H": spec.positive,
            "N": spec.negative,
        },
    }


@dataclass
class Decision:
    backend: str
    model: str
    category: str
    label: int
    probability: float
    confidence: float
    latency_ms: float
    confidence_kind: str
    choice: str | None = None
    probabilities: dict[str, float] | None = None
    native_confidence: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    subtype: str | None = None
    rationale: str | None = None
    raw: dict[str, Any] | None = None

    @property
    def label_name(self) -> str:
        return self.category if self.label == 1 else "normal"

    def to_dict(self, include_raw: bool = False) -> dict[str, Any]:
        result = asdict(self)
        result["label_name"] = self.label_name
        if not include_raw:
            result.pop("raw", None)
        return result


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    import json

    return json.dumps(value, ensure_ascii=False, sort_keys=True)
