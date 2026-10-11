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


def _with_clef_model(arguments: list[str]) -> list[str]:
    if not arguments:
        return ["--help"]

    command, *rest = arguments
    if command == "judge":
        return [command, "--model", "clef", *rest]
    if command in {"compare", "benchmark"}:
        return [command, "--models", "clef", *rest]
    if command == "shell":
        return [command, "--model", "clef", *rest]
    return arguments


def _ensure_credentials() -> None:
    if not os.getenv("CLOUDFLARE_ACCOUNT_ID"):
        account_id = input("Cloudflare account ID: ").strip()
        if not account_id:
            raise SystemExit("CLOUDFLARE_ACCOUNT_ID is required.")
        os.environ["CLOUDFLARE_ACCOUNT_ID"] = account_id

    if os.getenv("CLOUDFLARE_AUTH_TOKEN") or os.getenv("CLOUDFLARE_API_TOKEN"):
        return
    api_token = getpass.getpass("Cloudflare Workers AI API token: ").strip()
    if not api_token:
        raise SystemExit("CLOUDFLARE_AUTH_TOKEN is required.")
    os.environ["CLOUDFLARE_AUTH_TOKEN"] = api_token


def main() -> None:
    _load_secrets()
    arguments = _with_clef_model(sys.argv[1:])
    if "--help" not in arguments and "-h" not in arguments:
        _ensure_credentials()
    sys.argv = [sys.argv[0], *arguments]
    decision_models_main()


if __name__ == "__main__":
    main()
