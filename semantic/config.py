"""Local, server-side configuration for future Gemini functionality."""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import os


ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


def _read_env_file(path: Path) -> Optional[str]:
    """Read GEMINI_API_KEY from a small project-local .env file."""

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return None
    except OSError as error:
        raise RuntimeError("Could not read Gemini configuration from .env.") from error

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        name, separator, raw_value = line.partition("=")
        if separator and name.strip() == "GEMINI_API_KEY":
            value = raw_value.strip()
            if (
                len(value) >= 2
                and value[0] == value[-1]
                and value[0] in {"'", '"'}
            ):
                value = value[1:-1].strip()
            return value or None

    return None


def get_gemini_api_key() -> str:
    """Return the configured Gemini key or raise a safe, actionable error."""

    environment_value = os.environ.get("GEMINI_API_KEY", "").strip()
    if environment_value:
        return environment_value

    file_value = _read_env_file(ENV_FILE)
    if file_value:
        return file_value

    raise RuntimeError(
        "GEMINI_API_KEY is not configured. Set it in the environment "
        "or in the project .env file."
    )
