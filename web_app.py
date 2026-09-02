"""Local browser UI for the autocomplete engine.

Run ``py web_app.py`` and open http://127.0.0.1:8000. The page is served
locally; typed search text never leaves this computer. Optional voice typing
uses the browser's speech-recognition service, whose privacy behavior depends
on the browser and its configuration.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import sqlite3
import time
from typing import Any
import tempfile
import uuid
import webbrowser

from autocomplete import AutocompleteEngine


# Keep all default data paths anchored to this project.  A shortcut, IDE, or
# ``py C:\path\to\web_app.py`` command may use a different working directory.
PROJECT_DIRECTORY = Path(__file__).resolve().parent
WEB_DIRECTORY = PROJECT_DIRECTORY / "web"
MAX_QUERY_LENGTH = 2_000
MAX_TRANSLATION_TEXT_LENGTH = 500
MAX_INDEX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024


class AutocompleteWebHandler(BaseHTTPRequestHandler):
    """Serve the static UI and a small JSON API over localhost only."""

    server: HTTPServer

    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        if self.path == "/api/status":
            self._send_json(self._index_summary())
            return
        if self.path == "/api/indexes":
            self._send_json({"indexes": self._available_local_indexes()})
            return
        requested = "index.html" if self.path in ("/", "/index.html") else self.path.lstrip("/")
        candidate = (WEB_DIRECTORY / requested).resolve()
        if WEB_DIRECTORY.resolve() not in candidate.parents or not candidate.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        content_types = {
            ".html": "text/html; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
        }
        self._send_bytes(candidate.read_bytes(), content_types.get(candidate.suffix, "application/octet-stream"))

    def do_POST(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        if self.path == "/api/index":
            self._replace_index_from_upload()
            return
        if self.path == "/api/index/select":
            self._select_local_index()
            return
        if self.path == "/api/translate":
            self._translate_hebrew_to_english()
            return
        if self.path != "/api/suggestions":
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        try:
            query = self._read_short_text("query", MAX_QUERY_LENGTH)
        except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
            self._send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            return

        if query == "#":
            self._send_json({"suggestions": [], "elapsed_ms": 0.0, "reset": True})
            return

        started_at = time.perf_counter()
        engine: AutocompleteEngine = getattr(self.server, "engine")
        suggestions, diagnostics = engine.search_with_diagnostics(query)
        elapsed_ms = (time.perf_counter() - started_at) * 1_000
        diagnostic_payload = diagnostics.as_dict()
        diagnostic_payload.update(
            {
                "request_id": uuid.uuid4().hex,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "index_status": engine.index_status,
                "indexed_sentence_count": engine.indexed_sentence_count,
                "server_elapsed_ms": round(elapsed_ms, 3),
            }
        )
        self._append_diagnostics(diagnostic_payload)
        self._send_json(
            {
                "suggestions": [
                    {
                        "completed_sentence": item.completed_sentence,
                        "source_text": item.source_text,
                        "offset": item.offset,
                        "score": item.score,
                    }
                    for item in suggestions
                ],
                "elapsed_ms": round(elapsed_ms, 3),
                "diagnostics": diagnostic_payload,
                "reset": False,
            }
        )

    def _translate_hebrew_to_english(self) -> None:
        """Translate a short voice transcript without exposing a cloud API."""

        try:
            transcript = self._read_short_text("text", MAX_TRANSLATION_TEXT_LENGTH)
            from translation import TranslationUnavailable, translate_hebrew_to_english

            translated_text = translate_hebrew_to_english(transcript)
        except TranslationUnavailable as error:
            self._send_json({"error": str(error)}, HTTPStatus.SERVICE_UNAVAILABLE)
            return
        except (UnicodeDecodeError, ValueError, json.JSONDecodeError, RuntimeError) as error:
            self._send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            return

        self._send_json({"translated_text": translated_text})

    def _read_short_text(self, field_name: str, maximum_length: int) -> str:
        """Read a bounded JSON string field from a request body."""

        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0 or content_length > maximum_length * 4:
            raise ValueError("The request is too large.")
        payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
        text = payload.get(field_name, "")
        if not isinstance(text, str) or not text.strip() or len(text) > maximum_length:
            raise ValueError("{0} must be a short, non-empty string.".format(field_name))
        return text

    def _replace_index_from_upload(self) -> None:
        """Accept an index upload only when it matches the active archive."""

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            filename = self.headers.get("X-Index-Filename", "index.sqlite3")
            if content_length <= 0 or content_length > MAX_INDEX_UPLOAD_BYTES:
                raise ValueError("The index file must be between 1 byte and 2 GiB.")
            if not filename.lower().endswith(".sqlite3"):
                raise ValueError("Please select an index.sqlite3 file.")

            upload_directory: Path = getattr(self.server, "upload_directory")
            uploaded_path = upload_directory / "uploaded-{0}.sqlite3".format(uuid.uuid4().hex)
            remaining = content_length
            with uploaded_path.open("wb") as destination:
                while remaining:
                    chunk = self.rfile.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError("The uploaded file ended unexpectedly.")
                    destination.write(chunk)
                    remaining -= len(chunk)

            previous_engine: AutocompleteEngine = getattr(self.server, "engine")
            candidate = AutocompleteEngine.from_existing_index(previous_engine.archive_path, uploaded_path)
            candidate.close()
        except (OSError, RuntimeError, ValueError, sqlite3.DatabaseError) as error:
            if "uploaded_path" in locals() and uploaded_path.exists():
                uploaded_path.unlink()
            self._send_json({"error": "The index was not accepted: {0}".format(error)}, HTTPStatus.BAD_REQUEST)
            return

        previous_engine.close()
        persistent_index_path: Path = getattr(self.server, "persistent_index_path")
        backup_path = persistent_index_path.with_name(
            ".{0}.{1}.backup".format(persistent_index_path.name, uuid.uuid4().hex)
        )
        try:
            if persistent_index_path.is_file():
                persistent_index_path.replace(backup_path)
            uploaded_path.replace(persistent_index_path)
            replacement = AutocompleteEngine.from_existing_index(
                previous_engine.archive_path, persistent_index_path
            )
        except (OSError, RuntimeError, ValueError, sqlite3.DatabaseError) as error:
            if persistent_index_path.is_file():
                persistent_index_path.unlink()
            if backup_path.is_file():
                backup_path.replace(persistent_index_path)
            restored_engine = AutocompleteEngine.from_existing_index(
                previous_engine.archive_path, persistent_index_path
            )
            setattr(self.server, "engine", restored_engine)
            self._send_json({"error": "The index was not activated: {0}".format(error)}, HTTPStatus.BAD_REQUEST)
            return
        finally:
            if backup_path.is_file():
                backup_path.unlink()

        setattr(self.server, "engine", replacement)
        setattr(self.server, "active_index_name", Path(filename).name)
        self._send_json({"message": "Index accepted and loaded.", **self._index_summary()})

    def _available_local_indexes(self) -> list[dict[str, Any]]:
        """List selectable SQLite indexes in the project index directory tree."""

        index_directory: Path = getattr(self.server, "index_directory")
        active_name = getattr(self.server, "active_index_name")
        return [
            {
                "name": path.relative_to(index_directory).as_posix(),
                "size_bytes": path.stat().st_size,
                "active": path.relative_to(index_directory).as_posix() == active_name,
            }
            for path in sorted(
                index_directory.rglob("*.sqlite3"),
                key=lambda item: item.relative_to(index_directory).as_posix().casefold(),
            )
            if path.is_file()
        ]

    def _select_local_index(self) -> None:
        """Activate an existing local index and its matching local archive."""

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0 or content_length > 4_096:
                raise ValueError("The index-selection request is invalid.")
            payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
            index_name = payload.get("index_name")
            if not isinstance(index_name, str) or Path(index_name).is_absolute():
                raise ValueError("Please select an available local index.")
            if not index_name.lower().endswith(".sqlite3"):
                raise ValueError("Please select an index.sqlite3 file.")

            index_directory: Path = getattr(self.server, "index_directory")
            candidate_path = (index_directory / index_name).resolve()
            try:
                candidate_path.relative_to(index_directory.resolve())
            except ValueError:
                raise ValueError("Please select an available local index.") from None
            if not candidate_path.is_file():
                raise ValueError("The selected index file was not found.")

            previous_engine: AutocompleteEngine = getattr(self.server, "engine")
            candidate, archive_path = self._open_matching_local_index(candidate_path, previous_engine.archive_path)
        except (OSError, RuntimeError, ValueError, UnicodeDecodeError, json.JSONDecodeError, sqlite3.DatabaseError) as error:
            self._send_json({"error": "The index was not activated: {0}".format(error)}, HTTPStatus.BAD_REQUEST)
            return

        previous_engine.close()
        setattr(self.server, "engine", candidate)
        setattr(self.server, "persistent_index_path", candidate_path)
        setattr(self.server, "active_index_name", index_name)
        setattr(self.server, "active_archive_name", archive_path.name)
        self._send_json({"message": "Local index activated.", **self._index_summary()})

    def _open_matching_local_index(
        self, index_path: Path, current_archive_path: Path
    ) -> tuple[AutocompleteEngine, Path]:
        """Open an index with its current archive or a matching local ZIP archive."""

        index_directory: Path = getattr(self.server, "index_directory")
        archive_candidates = [Path(current_archive_path)]
        archive_candidates.extend(
            sorted(
                index_directory.rglob("*.zip"),
                key=lambda item: item.relative_to(index_directory).as_posix().casefold(),
            )
        )
        checked_paths = set()
        for archive_path in archive_candidates:
            resolved_archive = archive_path.resolve()
            if resolved_archive in checked_paths or not resolved_archive.is_file():
                continue
            checked_paths.add(resolved_archive)
            try:
                return AutocompleteEngine.from_existing_index(resolved_archive, index_path), resolved_archive
            except (OSError, RuntimeError, ValueError, sqlite3.DatabaseError):
                continue
        raise ValueError(
            "The selected index does not match the active archive or any local .zip archive. "
            "Keep its matching archive beside the index."
        )

    def _index_summary(self) -> dict[str, Any]:
        """Describe the exact index currently used by the live engine."""

        engine: AutocompleteEngine = getattr(self.server, "engine")
        return {
            "index_name": getattr(self.server, "active_index_name"),
            "archive_name": getattr(self.server, "active_archive_name"),
            "indexed_sentence_count": engine.indexed_sentence_count,
            "index_status": engine.index_status,
        }

    def _append_diagnostics(self, diagnostic_payload: dict[str, Any]) -> None:
        """Persist the exact diagnostic record that is sent to the browser."""

        log_directory: Path = getattr(self.server, "log_directory")
        log_directory.mkdir(parents=True, exist_ok=True)
        with (log_directory / "search.jsonl").open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(diagnostic_payload, ensure_ascii=False) + "\n")

    def log_message(self, _format: str, *_args: object) -> None:
        """Keep normal browser requests out of the terminal output."""

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        self._send_bytes(json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8", status)

    def _send_bytes(
        self, content: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local web UI for autocomplete.py")
    parser.add_argument("--archive", type=Path, default=PROJECT_DIRECTORY / "Archive (2).zip")
    parser.add_argument("--index", type=Path, default=PROJECT_DIRECTORY / "index.sqlite3")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument(
        "--install-hebrew-translation-model",
        action="store_true",
        help="download the one-time local Hebrew-to-English translation model, then exit",
    )
    arguments = parser.parse_args()

    if arguments.install_hebrew_translation_model:
        from translation import install_hebrew_to_english_model

        print("Installing the local Hebrew-to-English translation model...", flush=True)
        install_hebrew_to_english_model()
        print("Hebrew-to-English translation is ready.")
        return

    # Do this before accepting requests so the first Hebrew search does not
    # have to download or load the translation model on behalf of the user.
    from translation import TranslationUnavailable, prepare_hebrew_to_english_translation

    print("Preparing local Hebrew-to-English translation...", flush=True)
    try:
        prepare_hebrew_to_english_translation()
        print("Local Hebrew-to-English translation is ready.", flush=True)
    except TranslationUnavailable as error:
        # Search remains useful even when Argos or its model cannot be
        # downloaded (for example, while offline). The translate endpoint
        # continues to return its existing clear error in that case.
        print("Local Hebrew translation is unavailable: {0}".format(error), flush=True)

    print("Preparing autocomplete index...", flush=True)
    started_at = time.perf_counter()
    engine = AutocompleteEngine.from_archive(
        arguments.archive, index_path=arguments.index, rebuild_index=arguments.rebuild_index
    )
    print(
        "{0}: {1:,} lines ready in {2:.1f}s.".format(
            engine.index_status.capitalize(),
            engine.indexed_sentence_count,
            time.perf_counter() - started_at,
        )
    )

    address = ("127.0.0.1", arguments.port)
    server = HTTPServer(address, AutocompleteWebHandler)
    setattr(server, "engine", engine)
    setattr(server, "persistent_index_path", arguments.index.resolve())
    setattr(server, "active_index_name", arguments.index.name)
    setattr(server, "active_archive_name", arguments.archive.name)
    setattr(server, "index_directory", arguments.index.resolve().parent)
    server_upload_directory = tempfile.TemporaryDirectory(prefix="autocomplete-upload-")
    setattr(server, "upload_directory", Path(server_upload_directory.name))
    setattr(server, "log_directory", arguments.log_dir)
    url = "http://{0}:{1}".format(*address)
    print("Open {0} — press Ctrl+C to stop.".format(url))
    if not arguments.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping web UI.")
    finally:
        server.server_close()
        getattr(server, "engine").close()
        server_upload_directory.cleanup()


if __name__ == "__main__":
    main()
