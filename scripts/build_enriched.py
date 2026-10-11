"""Join completed judgments to the eight evaluation views by sample id."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import MODELS, ROOT, VIEWS, enriched_name, result_path


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def unique_index(rows: list[dict], path: Path) -> dict[str, dict]:
    indexed = {str(row["id"]): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"duplicate ids: {path}")
    return indexed


def ragtruth_layout(row: dict) -> dict:
    """Restore the original RAGTruth field names in the shared results."""
    row = dict(row)
    prompt, source_info = row.pop("input"), row.pop("context")
    row.pop("instruction", None)
    row.pop("reference", None)
    row["source_info"] = source_info
    row["prompt"] = prompt
    metadata = dict(row["metadata"])
    metadata["source"] = metadata.pop("source_dataset")
    metadata["model"] = metadata.pop("generator_model")
    metadata["labels"] = metadata.pop("span_labels")
    row["metadata"] = metadata
    return row


def build_one(view: tuple, model: str, output_dir: Path) -> dict:
    name, _, source_relative, count, h_count, n_count, template = view
    source_path = ROOT / source_relative
    raw_path = result_path(name, model)
    source_rows = read_jsonl(source_path)
    result_rows = read_jsonl(raw_path)
    source_ids = unique_index(source_rows, source_path)
    results = unique_index(result_rows, raw_path)
    if len(source_rows) != count or sum(int(r["label"]) for r in source_rows) != h_count or count - h_count != n_count:
        raise ValueError(f"source count mismatch: {source_path}")
    if set(source_ids) != set(results):
        raise ValueError(f"id mismatch: {raw_path}; missing={len(set(source_ids)-set(results))}, extra={len(set(results)-set(source_ids))}")

    enriched = []
    decision_key = MODELS[model][1]
    for source_row in source_rows:
        result_row = results[str(source_row["id"])]
        if int(source_row["label"]) != int(result_row["gold_label"]):
            raise ValueError(f"gold-label mismatch: {source_row['id']}")
        decision = result_row.get("decisions", {}).get(decision_key, {})
        if decision.get("error") or decision.get("label") not in (0, 1):
            raise ValueError(f"failed decision: {source_row['id']} in {raw_path}")
        merged = dict(source_row)
        merged.update(result_row)
        enriched.append(ragtruth_layout(merged) if name == "CI_RAGTruth" else merged)

    output = output_dir / enriched_name(template, model)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in enriched:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return {"view": name, "model": model, "rows": len(enriched), "H": h_count, "N": n_count,
            "source_file": source_relative, "result_file": str(raw_path.relative_to(ROOT)).replace("\\", "/"), "output": output.name}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "enriched_with_source")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = [build_one(view, model, output_dir) for view in VIEWS for model in MODELS]
    (output_dir / "manifest.json").write_text(json.dumps({"files": entries}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(entries)} enriched JSONL files to {output_dir}")


if __name__ == "__main__":
    main()
