from __future__ import annotations

import getpass
import os
import sys
from pathlib import Path

from decision_models.cli import main as decision_models_main


SECRETS_FILE = Path(__file__).resolve().with_name("secrets.env")


def _load_secrets(path: Path = SECRETS_FILE) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value:
            os.environ.setdefault(key, value)


def _with_sol_model(arguments: list[str]) -> list[str]:
    if not arguments:
        return ["--help"]

    command, *rest = arguments
    if command == "judge":
        return [command, "--model", "sol", *rest]
    if command in {"compare", "benchmark"}:
        return [command, "--models", "sol", *rest]
    if command == "shell":
        return [command, "--model", "sol", *rest]
    return arguments


def _ensure_api_key() -> None:
    if os.getenv("OPENAI_API_KEY"):
        return
    api_key = getpass.getpass("OpenAI API key: ").strip()
    if not api_key:
        raise SystemExit("An OPENAI_API_KEY is required.")
    os.environ["OPENAI_API_KEY"] = api_key


def main() -> None:
    _load_secrets()
    arguments = _with_sol_model(sys.argv[1:])
    if "--help" not in arguments and "-h" not in arguments:
        _ensure_api_key()
    sys.argv = [sys.argv[0], *arguments]
    decision_models_main()


if __name__ == "__main__":
    main()
