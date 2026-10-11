# Category prompts

Each hallucination category has one dataset-independent prompt module:

- `factual_contradiction.py`
- `factual_fabrication.py`
- `context_inconsistency.py`
- `logical_inconsistency.py`
- `instruction_inconsistency.py`

Every module exports `QUESTION`, `POSITIVE`, and `NEGATIVE`. `decision_models.core`
loads them through `PROMPTS`, preserving the shared `CategorySpec` interface used by
all model backends.
