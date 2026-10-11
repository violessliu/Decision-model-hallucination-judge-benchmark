"""Plan or run the 48 view/model evaluations in the share package."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

from config import MODELS, ROOT, VIEWS, result_path


def validate_source(view: tuple) -> None:
    name, _, relative, expected_rows, expected_h, expected_n, _ = view
    path = ROOT / relative
    ids: set[str] = set()
    rows = positives = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row["id"])
            if sample_id in ids:
                raise ValueError(f"duplicate id in {path}: {sample_id}")
            ids.add(sample_id)
            label = int(row["label"])
            if label not in (0, 1):
                raise ValueError(f"non-binary label in {path}: {label}")
            rows += 1
            positives += label
    if (rows, positives, rows - positives) != (expected_rows, expected_h, expected_n):
        raise ValueError(f"{name}: expected {(expected_rows, expected_h, expected_n)}, got {(rows, positives, rows - positives)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="call model APIs; otherwise print the plan only")
    parser.add_argument("--model", action="append", choices=MODELS, help="repeat to select models")
    parser.add_argument("--view", action="append", choices=[v[0] for v in VIEWS], help="repeat to select views")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    selected_views = [v for v in VIEWS if args.view is None or v[0] in args.view]
    selected_models = [m for m in MODELS if args.model is None or m in args.model]
    for view in selected_views:
        validate_source(view)
    print(f"{len(selected_views)} views x {len(selected_models)} models = {len(selected_views) * len(selected_models)} runs")

    for view in selected_views:
        name, category, source, *_ = view
        for model in selected_models:
            wrapper, _ = MODELS[model]
            output = result_path(name, model)
            command = [
                sys.executable, str(ROOT / wrapper), "benchmark",
                "--file", str(ROOT / source), "--category", category,
                "--output", str(output), "--limit", "0", "--seed", "2026",
            ]
            if output.exists():
                if args.skip_existing:
                    print(f"SKIP {name} {model}: {output}")
                    continue
                raise FileExistsError(f"result exists: {output}; use --skip-existing or move it first")
            print(f"{name} {model}: {' '.join(command)}", flush=True)
            if args.execute:
                subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
