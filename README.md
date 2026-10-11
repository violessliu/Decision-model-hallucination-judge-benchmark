# Hallucination Judge Benchmark: code and evaluation data

This repository contains the runnable judge code and the **eight binary data views** in the current share package. It contains **no model results and no credentials**. The five hallucination categories are factual contradiction (FC), factual fabrication (FF), context inconsistency (CI), logical inconsistency (LI), and instruction inconsistency (II).

The eight views contain 9,852 benchmark instances. Each view is evaluated by six judges (CLEF, D1, JEV, gpt-6-luna, PPLX, and gpt-6-sol), giving **48 view/model runs**. The results of a fresh run may differ from previously shared judgments because hosted models and services can change.

## Dataset views

| View | Input JSONL | Rows | H | N |
|---|---|---:|---:|---:|
| CI / RAGTruth | `data/03_context_inconsistency/RAGTruth/binary.jsonl` | 2,675 | 943 | 1,732 |
| FC / FAVABench | `data/01_factual_contradiction/FAVABench/binary_inclusive.jsonl` | 377 | 224 | 153 |
| FC / HADTest | `data/01_factual_contradiction/HADTest/binary_pure.jsonl` | 1,244 | 120 | 1,124 |
| FF / FAVABench | `data/02_factual_fabrication/FAVABench/binary_inclusive.jsonl` | 351 | 198 | 153 |
| FF / HADTest | `data/02_factual_fabrication/HADTest/binary_pure.jsonl` | 1,265 | 141 | 1,124 |
| II / IF-RewardBench | `data/05_instruction_inconsistency/IFRewardBench/binary_english.jsonl` | 2,294 | 1,600 | 694 |
| LI / ContraDoc | `data/04_logical_inconsistency/ContraDoc/binary.jsonl` | 891 | 449 | 442 |
| LI / ROSCOE | `data/04_logical_inconsistency/ROSCOE/binary.jsonl` | 755 | 89 | 666 |

The current FAVABench views exclude responses annotated **only** with the other factual error type. FC excludes 83 such responses and FF excludes 109. Positive responses with both types remain positive in both views. HADTest uses the same `binary_pure.jsonl` views as the share package. See [data/README.md](data/README.md) for dataset details.

## Run the judges

Use Python 3.10 or newer. Install dependencies and inspect the 48-command plan without calling an API:

```bash
pip install -r requirements.txt
python scripts/run_all.py
```

Set the required API keys in your environment, or create a local `secrets.env` from your own credentials. `secrets.env` is ignored by Git. Then run all 48 evaluations:

```bash
python scripts/run_all.py --execute
```

Each result is written under `results/full/<view>/<model>.jsonl`, with a companion `.summary.json`. To run a single combination, for example:

```bash
python scripts/run_all.py --execute --view FF_FAVABench --model JEV
```

The runner refuses to replace an existing result. Use `--skip-existing` to resume a partially completed batch. If a result has failed decisions, use `python scripts/rerun_failed.py --result-file <path> --model <decision-key>` before enrichment. The six individual `run_*.py` entry points and `decision_models/` contain the underlying benchmark implementation and prompts.

## Enrich the results

After all 48 runs have valid judgments, join each result to its source data by ID:

```bash
python scripts/build_enriched.py
python scripts/jsonl_to_csv.py
```

This writes 48 enriched JSONL files and 48 CSV files under `results/enriched_with_source/`. Enrichment checks IDs, row counts, gold labels, and decision validity; it does not call a model API. It restores RAGTruth's `prompt` and `source_info` field names, matching the share package layout. All `results/` files are ignored by Git.
