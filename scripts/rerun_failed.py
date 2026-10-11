"""Re-judge rows whose decision failed in a raw result file, patch them in place, and refresh its summary.

Usage (from the repository root):
    python scripts/rerun_failed.py --result-file results/full/.../xxx.jsonl --model sol

--model is the decision key inside the result file: clef, d1, jev, openai-decisions, pplx, sol.
The original file and summary are backed up once to <name>.bak before writing.
API keys are read from secrets.env in the repository root.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from decision_models.backends import build_backend  # noqa: E402
from decision_models.cli import binary_metrics  # noqa: E402
from decision_models.core import JudgeInput  # noqa: E402

ATTEMPTS = 3


def load_secrets() -> None:
    path = ROOT / "secrets.env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            value = value.strip().strip('"').strip("'")
            if key.strip() and value:
                os.environ.setdefault(key.strip(), value)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def backup_once(path: Path) -> None:
    backup = path.with_name(path.name + ".bak")
    if not backup.exists():
        shutil.copyfile(path, backup)


def failed(row: dict, model: str) -> bool:
    decision = row.get("decisions", {}).get(model, {})
    return bool(decision.get("error")) or decision.get("label") not in (0, 1)


def judge(backend, row: dict) -> dict:
    item = JudgeInput.from_mapping(row, category=row["category"])
    last = None
    for attempt in range(1, ATTEMPTS + 1):
        try:
            return backend.judge(item).to_dict()
        except Exception as error:  # retry transient API failures
            last = f"{type(error).__name__}: {error}"
            print(f"    attempt {attempt} failed: {last[:120]}", flush=True)
            time.sleep(5 * attempt)
    return {"error": last}


def refresh_summary(result_path: Path, model: str) -> None:
    summary_path = result_path.with_suffix(".summary.json")
    if not summary_path.exists():
        return
    records = read_jsonl(result_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    valid = [r for r in records if "label" in r["decisions"].get(model, {})]
    decisions = [r["decisions"][model] for r in valid]

    def values(key: str) -> list:
        return [d[key] for d in decisions if d.get(key) is not None]

    stats = binary_metrics([r["gold_label"] for r in valid], [d["label"] for d in decisions])
    stats["errors"] = len(records) - len(valid)
    latencies = values("latency_ms")
    stats["mean_latency_ms"] = statistics.fmean(latencies) if latencies else None
    stats["median_latency_ms"] = statistics.median(latencies) if latencies else None
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        tokens = values(key)
        stats[f"total_{key}"] = sum(tokens) if tokens else None
    for key in ("input_tokens", "output_tokens"):
        tokens = values(key)
        stats[f"mean_{key}"] = statistics.fmean(tokens) if tokens else None
    previous = summary["models"].get(model, {})
    summary["models"][model] = {**previous, **stats}
    backup_once(summary_path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"summary refreshed: {summary_path.name} errors={stats['errors']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-file", type=Path, required=True)
    parser.add_argument("--model", required=True, help="decision key in the result file, e.g. sol or openai-decisions")
    args = parser.parse_args()

    load_secrets()
    path = args.result_file if args.result_file.is_absolute() else ROOT / args.result_file
    rows = read_jsonl(path)
    todo = [r for r in rows if failed(r, args.model)]
    print(f"{path.name}: {len(todo)} failed rows")
    if not todo:
        return
    backend = build_backend(args.model)
    for row in todo:
        row["decisions"][args.model] = judge(backend, row)
        decision = row["decisions"][args.model]
        print(f"  {row['id']}: {decision.get('error') or 'label=%s gold=%s' % (decision['label'], row['gold_label'])}", flush=True)

    backup_once(path)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    refresh_summary(path, args.model)
    print("remaining failures:", sum(failed(r, args.model) for r in rows))


if __name__ == "__main__":
    main()
