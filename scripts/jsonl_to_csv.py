"""Convert every *.jsonl in a folder to CSV under <folder>/csv (UTF-8 with BOM, opens directly in Excel).

Nested objects are flattened into dotted column names (metadata.task_type, decisions.sol.label);
lists and any deeper dicts are written as JSON strings.

Usage (from the repository root):
    python scripts/jsonl_to_csv.py                                   # results/enriched_with_source
    python scripts/jsonl_to_csv.py --input-dir some/other/folder
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def flatten(value, prefix: str, out: dict) -> None:
    if isinstance(value, dict) and value and prefix.count(".") < 3:
        for key, child in value.items():
            flatten(child, f"{prefix}.{key}" if prefix else key, out)
    elif isinstance(value, (dict, list)):
        out[prefix] = json.dumps(value, ensure_ascii=False)
    else:
        out[prefix] = value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=ROOT / "results/enriched_with_source")
    args = parser.parse_args()
    out_dir = args.input_dir / "csv"
    out_dir.mkdir(exist_ok=True)
    for path in sorted(args.input_dir.glob("*.jsonl")):
        rows, columns = [], {}
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    flat: dict = {}
                    flatten(json.loads(line), "", flat)
                    rows.append(flat)
                    columns.update(dict.fromkeys(flat))
        target = out_dir / (path.name.removesuffix(".jsonl") + ".csv")
        with target.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(columns))
            writer.writeheader()
            writer.writerows(rows)
        print(f"{target.name}: {len(rows)} rows, {len(columns)} columns")


if __name__ == "__main__":
    main()
