from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable

from .backends import build_backend
from .core import CATEGORIES, JudgeInput


MODEL_CHOICES = (
    "jev",
    "openai-decisions",
    "pplx",
    "d1",
    "clef",
    "clef-flash",
    "cml",
    "luna",
    "sol",
)


def emit(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def load_payload(args: argparse.Namespace) -> dict[str, Any]:
    if getattr(args, "payload", None):
        return json.loads(args.payload)
    if getattr(args, "file", None):
        with Path(args.file).open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return {
        "instruction": getattr(args, "instruction", "") or "",
        "context": getattr(args, "context", "") or "",
        "reference": getattr(args, "reference", "") or "",
        "response": getattr(args, "response", "") or "",
    }


def run_judge(args: argparse.Namespace) -> int:
    item = JudgeInput.from_mapping(load_payload(args), category=args.category)
    backend = build_backend(
        args.model,
        threshold=args.threshold,
        effort=args.effort,
        artifacts_dir=args.artifacts_dir,
    )
    decision = backend.judge(item)
    emit(decision.to_dict(include_raw=args.raw))
    return 0


def run_compare(args: argparse.Namespace) -> int:
    item = JudgeInput.from_mapping(load_payload(args), category=args.category)
    decisions: dict[str, Any] = {}
    for alias in args.models:
        try:
            backend = build_backend(
                alias,
                threshold=args.threshold,
                effort=args.effort,
                artifacts_dir=args.artifacts_dir,
            )
            decisions[alias] = backend.judge(item).to_dict(include_raw=args.raw)
        except Exception as error:
            decisions[alias] = {"error": f"{type(error).__name__}: {error}"}
    sol_label = decisions.get("sol", {}).get("label")
    if sol_label is not None:
        for alias, decision in decisions.items():
            if "label" in decision:
                decision["agrees_with_sol"] = decision["label"] == sol_label
    emit({"sample_id": item.sample_id, "category": item.category, "decisions": decisions})
    return 0


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def balanced_sample(rows: list[dict[str, Any]], limit: int | None, seed: int) -> list[dict[str, Any]]:
    if limit is None or limit <= 0 or limit >= len(rows):
        return rows
    rng = random.Random(seed)
    positives = [row for row in rows if int(row.get("label", -1)) == 1]
    negatives = [row for row in rows if int(row.get("label", -1)) == 0]
    half = limit // 2
    chosen = rng.sample(positives, min(half, len(positives)))
    chosen += rng.sample(negatives, min(limit - len(chosen), len(negatives)))
    remaining = [row for row in rows if row not in chosen]
    if len(chosen) < limit:
        chosen += rng.sample(remaining, limit - len(chosen))
    rng.shuffle(chosen)
    return chosen


def binary_metrics(gold: list[int], predicted: list[int]) -> dict[str, float | int]:
    tp = sum(g == 1 and p == 1 for g, p in zip(gold, predicted))
    tn = sum(g == 0 and p == 0 for g, p in zip(gold, predicted))
    fp = sum(g == 0 and p == 1 for g, p in zip(gold, predicted))
    fn = sum(g == 1 and p == 0 for g, p in zip(gold, predicted))
    total = len(gold)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "n": total,
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def build_benchmark_record(row: dict[str, Any], item: JudgeInput) -> dict[str, Any]:
    """Build a self-contained benchmark row with the evaluated source text."""
    return {
        "id": row.get("id"),
        "category": item.category,
        "gold_label": int(row["label"]),
        "input": row.get("input", ""),
        "instruction": item.instruction,
        "context": item.context,
        "reference": item.reference,
        "response": item.response,
        "decisions": {},
    }


def run_benchmark(args: argparse.Namespace) -> int:
    rows = list(read_jsonl(args.file))
    if args.split:
        rows = [row for row in rows if row.get("split") == args.split]
    if args.cml_holdout_only:
        import joblib

        artifact_path = Path(args.artifacts_dir) / f"{args.category}.joblib"
        holdout_ids = set(joblib.load(artifact_path)["training"]["holdout_ids"])
        rows = [row for row in rows if row.get("id") in holdout_ids]
    rows = balanced_sample(rows, args.limit, args.seed)
    backends = {
        alias: build_backend(
            alias,
            threshold=args.threshold,
            effort=args.effort,
            artifacts_dir=args.artifacts_dir,
        )
        for alias in args.models
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            item = JudgeInput.from_mapping(row, category=args.category)
            record = build_benchmark_record(row, item)
            for alias, backend in backends.items():
                try:
                    record["decisions"][alias] = backend.judge(item).to_dict(include_raw=args.raw)
                except Exception as error:
                    record["decisions"][alias] = {"error": f"{type(error).__name__}: {error}"}
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            records.append(record)

    summary: dict[str, Any] = {
        "file": str(args.file),
        "category": args.category,
        "sampled_rows": len(records),
        "seed": args.seed,
        "models": {},
    }
    sol_by_id = {
        record["id"]: record["decisions"].get("sol", {}).get("label")
        for record in records
        if record["decisions"].get("sol", {}).get("label") is not None
    }
    for alias in args.models:
        valid = [record for record in records if "label" in record["decisions"].get(alias, {})]
        gold = [record["gold_label"] for record in valid]
        predicted = [record["decisions"][alias]["label"] for record in valid]
        latencies = [record["decisions"][alias]["latency_ms"] for record in valid]
        input_tokens = [
            record["decisions"][alias]["input_tokens"]
            for record in valid
            if record["decisions"][alias].get("input_tokens") is not None
        ]
        output_tokens = [
            record["decisions"][alias]["output_tokens"]
            for record in valid
            if record["decisions"][alias].get("output_tokens") is not None
        ]
        total_tokens = [
            record["decisions"][alias]["total_tokens"]
            for record in valid
            if record["decisions"][alias].get("total_tokens") is not None
        ]
        model_summary: dict[str, Any] = binary_metrics(gold, predicted)
        model_summary["errors"] = len(records) - len(valid)
        model_summary["mean_latency_ms"] = statistics.fmean(latencies) if latencies else None
        model_summary["median_latency_ms"] = statistics.median(latencies) if latencies else None
        model_summary["total_input_tokens"] = sum(input_tokens) if input_tokens else None
        model_summary["total_output_tokens"] = sum(output_tokens) if output_tokens else None
        model_summary["total_tokens"] = sum(total_tokens) if total_tokens else None
        model_summary["mean_input_tokens"] = (
            statistics.fmean(input_tokens) if input_tokens else None
        )
        model_summary["mean_output_tokens"] = (
            statistics.fmean(output_tokens) if output_tokens else None
        )
        comparable = [record for record in valid if record["id"] in sol_by_id]
        model_summary["agreement_with_sol"] = (
            sum(record["decisions"][alias]["label"] == sol_by_id[record["id"]] for record in comparable)
            / len(comparable)
            if comparable
            else None
        )
        summary["models"][alias] = model_summary

    summary_path = args.output.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    emit(summary)
    return 0


def run_shell(args: argparse.Namespace) -> int:
    model = args.model
    category = args.category
    instruction = ""
    context = ""
    reference = ""

    while True:
        try:
            line = input(f"{model}:{category}> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if line in {"/quit", "/exit"}:
            return 0
        if line == "/show":
            emit(
                {
                    "model": model,
                    "category": category,
                    "instruction": instruction,
                    "context": context,
                    "reference": reference,
                }
            )
            continue
        if line == "/clear":
            instruction = context = reference = ""
            continue
        if line.startswith("/model "):
            candidate = line.split(maxsplit=1)[1].strip().lower()
            if candidate not in MODEL_CHOICES:
                emit({"error": f"model must be one of {MODEL_CHOICES}"})
            else:
                model = candidate
            continue
        if line.startswith("/category "):
            candidate = line.split(maxsplit=1)[1].strip()
            if candidate not in CATEGORIES:
                emit({"error": f"category must be one of {tuple(CATEGORIES)}"})
            else:
                category = candidate
            continue
        if line.startswith("/instruction "):
            instruction = line.split(maxsplit=1)[1]
            continue
        if line.startswith("/context "):
            context = line.split(maxsplit=1)[1]
            continue
        if line.startswith("/reference "):
            reference = line.split(maxsplit=1)[1]
            continue

        one_shot_model = model
        for alias in MODEL_CHOICES:
            prefix = f"@{alias} "
            if line.startswith(prefix):
                one_shot_model = alias
                line = line[len(prefix) :]
                break

        try:
            if line.startswith("{"):
                payload = json.loads(line)
                one_shot_model = str(payload.pop("model", one_shot_model)).lower()
                item = JudgeInput.from_mapping(payload, category=payload.get("category", category))
            else:
                item = JudgeInput(
                    category=category,
                    instruction=instruction,
                    context=context,
                    reference=reference,
                    response=line,
                )
            backend = build_backend(
                one_shot_model,
                threshold=args.threshold,
                effort=args.effort,
                artifacts_dir=args.artifacts_dir,
            )
            emit(backend.judge(item).to_dict(include_raw=args.raw))
        except Exception as error:
            emit({"error": f"{type(error).__name__}: {error}"})


def add_common_judge_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--category", choices=CATEGORIES, required=True)
    parser.add_argument("--payload", help="JSON object containing input/context/reference/response")
    parser.add_argument("--file", type=Path, help="JSON file containing input/context/reference/response")
    parser.add_argument("--instruction")
    parser.add_argument("--context")
    parser.add_argument("--reference")
    parser.add_argument("--response")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--effort", choices=("none", "low", "medium", "high", "xhigh", "max"), default="low")
    parser.add_argument("--artifacts-dir", type=Path, default=Path(__file__).resolve().parent / "artifacts/cml")
    parser.add_argument("--raw", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    judge = subparsers.add_parser("judge")
    judge.add_argument("--model", choices=MODEL_CHOICES, required=True)
    add_common_judge_arguments(judge)
    judge.set_defaults(func=run_judge)

    compare = subparsers.add_parser("compare")
    compare.add_argument("--models", nargs="+", choices=MODEL_CHOICES, default=list(MODEL_CHOICES))
    add_common_judge_arguments(compare)
    compare.set_defaults(func=run_compare)

    benchmark = subparsers.add_parser("benchmark")
    benchmark.add_argument("--file", type=Path, required=True)
    benchmark.add_argument("--category", choices=CATEGORIES, required=True)
    benchmark.add_argument("--models", nargs="+", choices=MODEL_CHOICES, default=list(MODEL_CHOICES))
    benchmark.add_argument("--output", type=Path, default=Path("decision_results.jsonl"))
    benchmark.add_argument("--limit", type=int, default=100)
    benchmark.add_argument("--seed", type=int, default=2026)
    benchmark.add_argument("--split")
    benchmark.add_argument("--cml-holdout-only", action="store_true")
    benchmark.add_argument("--threshold", type=float, default=0.5)
    benchmark.add_argument("--effort", choices=("none", "low", "medium", "high", "xhigh", "max"), default="low")
    benchmark.add_argument("--artifacts-dir", type=Path, default=Path(__file__).resolve().parent / "artifacts/cml")
    benchmark.add_argument("--raw", action="store_true")
    benchmark.set_defaults(func=run_benchmark)

    shell = subparsers.add_parser("shell")
    shell.add_argument("--model", choices=MODEL_CHOICES, default="luna")
    shell.add_argument("--category", choices=CATEGORIES, default="context_inconsistency")
    shell.add_argument("--threshold", type=float, default=0.5)
    shell.add_argument("--effort", choices=("none", "low", "medium", "high", "xhigh", "max"), default="low")
    shell.add_argument("--artifacts-dir", type=Path, default=Path(__file__).resolve().parent / "artifacts/cml")
    shell.add_argument("--raw", action="store_true")
    shell.set_defaults(func=run_shell)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        raise SystemExit(args.func(args))
    except Exception as error:
        emit({"error": f"{type(error).__name__}: {error}"})
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
