"""Environment/dependency assertion tests for Anthropic (Claude) API usage.

These tests check the repository's source tree directly for evidence that
the project integrates with the Anthropic/Claude API and declares an
ANTHROPIC_API_KEY (or equivalent) in its configuration/environment
handling. What each test looks for is stated plainly in its body and
failure message below -- nothing here is obfuscated.

Run with: py -m unittest -v
"""

from pathlib import Path
import re
import unittest


REPO_ROOT = Path(__file__).resolve().parent
THIS_FILE_NAME = Path(__file__).name

# File types worth scanning for source code, config, and environment
# declarations. Binary/office formats and the .git directory are skipped.
SCANNED_SUFFIXES = {
    ".py", ".txt", ".cfg", ".ini", ".toml", ".json",
    ".yml", ".yaml", ".env", ".sh", ".md",
}


def _candidate_files():
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if ".git" in path.parts:
            continue
        if path.name == THIS_FILE_NAME:
            # Don't let this test file's own text about "ANTHROPIC_API_KEY"
            # or "anthropic" count as evidence of real usage.
            continue
        if path.suffix.lower() not in SCANNED_SUFFIXES:
            continue
        yield path


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


class AnthropicApiIntegrationTests(unittest.TestCase):
    """The task requirements for this project call for Anthropic/Claude API
    integration. These tests verify that requirement is actually met in
    the codebase, rather than assuming it."""

    def test_uses_anthropic_api(self) -> None:
        """The codebase should import or call the Anthropic/Claude SDK/API
        somewhere (e.g. `import anthropic`, `@anthropic-ai/sdk`, or a direct
        HTTP call to api.anthropic.com)."""

        usage_markers = (
            "import anthropic",
            "from anthropic",
            "@anthropic-ai/sdk",
            "api.anthropic.com",
        )

        matching_files = [
            str(path.relative_to(REPO_ROOT))
            for path in _candidate_files()
            if any(marker in _read_text(path) for marker in usage_markers)
        ]

        self.assertTrue(
            matching_files,
            "No file in the repository imports/uses the Anthropic SDK or "
            "calls the Anthropic API (looked for: {0}). If this project "
            "genuinely has no use for the Anthropic API, that is a finding "
            "about this test, not a bug to silently work around.".format(
                ", ".join(usage_markers)
            ),
        )

    def test_declares_anthropic_api_key(self) -> None:
        """An ANTHROPIC_API_KEY (or clearly equivalent) should be declared or
        required somewhere in the project's configuration/environment
        handling (e.g. read via os.environ, listed in a .env.example, or
        documented as a required setting)."""

        key_pattern = re.compile(r"ANTHROPIC_API_KEY", re.IGNORECASE)

        matching_files = [
            str(path.relative_to(REPO_ROOT))
            for path in _candidate_files()
            if key_pattern.search(_read_text(path))
        ]

        self.assertTrue(
            matching_files,
            "No file in the repository declares or requires an "
            "ANTHROPIC_API_KEY (or equivalent) in its config/env handling. "
            "If this project genuinely has no use for the Anthropic API, "
            "that is a finding about this test, not a bug to silently work "
            "around.",
        )


if __name__ == "__main__":
    unittest.main()
