"""The eight data views and six judges in the 2026-10-11 share package."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

MODELS = {
    "CLEF": ("run_clef.py", "clef"),
    "D1": ("run_d1.py", "d1"),
    "JEV": ("run_jev.py", "jev"),
    "OPENAI_DECISIONS": ("run_openai_decisions.py", "openai-decisions"),
    "PPLX": ("run_pplx.py", "pplx"),
    "SOL": ("run_sol.py", "sol"),
}

# name, category, source, rows, H, N, enriched output name template
VIEWS = (
    ("CI_RAGTruth", "context_inconsistency", "data/03_context_inconsistency/RAGTruth/binary.jsonl", 2675, 943, 1732, "CI__{model}__2675"),
    ("FC_FAVABench", "factual_contradiction", "data/01_factual_contradiction/FAVABench/binary_inclusive.jsonl", 377, 224, 153, "FC__{model}__FAVA_INCL__377"),
    ("FC_HADTest", "factual_contradiction", "data/01_factual_contradiction/HADTest/binary_pure.jsonl", 1244, 120, 1124, "FC__{model}__HADTEST__1244"),
    ("FF_FAVABench", "factual_fabrication", "data/02_factual_fabrication/FAVABench/binary_inclusive.jsonl", 351, 198, 153, "FF__{model}__FAVA_INCL__351"),
    ("FF_HADTest", "factual_fabrication", "data/02_factual_fabrication/HADTest/binary_pure.jsonl", 1265, 141, 1124, "FF__{model}__HADTEST__1265"),
    ("II_IFRewardBench", "instruction_inconsistency", "data/05_instruction_inconsistency/IFRewardBench/binary_english.jsonl", 2294, 1600, 694, "II__{model}__2294"),
    ("LI_ContraDoc", "logical_inconsistency", "data/04_logical_inconsistency/ContraDoc/binary.jsonl", 891, 449, 442, "LI__{model}__891"),
    ("LI_ROSCOE", "logical_inconsistency", "data/04_logical_inconsistency/ROSCOE/binary.jsonl", 755, 89, 666, "LI_ROSCOE__{model}__755"),
)


def result_path(view_name: str, model: str) -> Path:
    return ROOT / "results" / "full" / view_name / f"{model}.jsonl"


def enriched_name(template: str, model: str) -> str:
    return template.format(model=model) + ".enriched.jsonl"
