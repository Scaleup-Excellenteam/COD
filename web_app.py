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
import threading
import time
from typing import Any, Optional
import tempfile
import uuid
import webbrowser

from autocomplete import AutocompleteEngine
from snapshot_store import read_current_snapshot, snapshot_index_path


WEB_DIRECTORY = Path(__file__).with_name("web")
MAX_QUERY_LENGTH = 2_000
MAX_TRANSLATION_TEXT_LENGTH = 500
MAX_INDEX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024
DEFAULT_POLL_INTERVAL_SECONDS = 2.0


def _activate_engine(
    server: HTTPServer, new_engine: AutocompleteEngine, active_name: str, engine_lock: threading.Lock
) -> None:
    """Atomically swap the serving engine, then close the retired one.

    The reference swap happens while holding ``engine_lock`` -- the same lock
    a request holds for its entire query -- so a request already in flight
    against the old engine always finishes before the old engine is closed.
    No request ever sees a torn state, and none are dropped.
    """

    with engine_lock:
        old_engine: AutocompleteEngine = getattr(server, "engine")
        setattr(server, "engine", new_engine)
        setattr(server, "active_index_name", active_name)
    old_engine.close()


def _is_manual_upload_allowed(server: HTTPServer) -> bool:
    """Manual browser upload only makes sense against a single fixed index.

    In snapshot mode there is no single "the archive" a manually uploaded
    index could be validated against -- snapshots are versioned and may come
    from different data sources over time -- so that path is disabled in
    favor of the ``build_snapshot.py`` -> pointer -> watcher hand-off.
    """

    return getattr(server, "snapshots_dir", None) is None


class SnapshotWatcher:
    """Poll the snapshot pointer in the background and hot-swap the serving engine.

    Offline builds land in their own versioned directory under
    ``snapshots_dir`` and only flip the ``CURRENT`` pointer once a build is
    complete and validated (see ``snapshot_store.build_and_activate_snapshot``).
    This watcher notices that the pointer changed, loads the new snapshot
    into memory on its own background thread, and atomically swaps the
    engine the server queries. Because the load happens off the
    request-handling thread and the swap is a single locked reference
    assignment, adding a data source never requires stopping the service.
    """

    def __init__(
        self,
        server: HTTPServer,
        snapshots_dir: Path,
        engine_lock: threading.Lock,
        poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
    ) -> None:
        self._server = server
        self._snapshots_dir = Path(snapshots_dir)
        self._engine_lock = engine_lock
        self._poll_interval_seconds = poll_interval_seconds
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name="snapshot-watcher", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=self._poll_interval_seconds + 5)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self.poll_once()
            self._stop_event.wait(self._poll_interval_seconds)

    def poll_once(self) -> bool:
        """Activate the pointed-to snapshot if it changed. Returns whether it swapped."""

        try:
            version_id = read_current_snapshot(self._snapshots_dir)
        except OSError:
            return False
        if version_id is None or version_id == getattr(self._server, "active_index_name", None):
            return False

        index_path = snapshot_index_path(self._snapshots_dir, version_id)
        try:
            new_engine = AutocompleteEngine.open_snapshot(index_path)
        except (OSError, RuntimeError, ValueError, sqlite3.DatabaseError):
            # The pointer names a snapshot that is missing, still being
            # written, or corrupt. Keep serving the current snapshot and try
            # again on the next poll -- a bad or half-finished build must
            # never take a live service down.
            return False

        _activate_engine(self._server, new_engine, version_id, self._engine_lock)
        return True


class AutocompleteWebHandler(BaseHTTPRequestHandler):
    """Serve the static UI and a small JSON API over localhost only."""

    server: HTTPServer

    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        if self.path == "/api/status":
            self._send_json(self._index_summary())
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
        engine_lock: threading.Lock = getattr(self.server, "engine_lock")
        # Held for the whole query, not just the reference read: this is what
        # lets the zero-downtime watcher swap and retire an old snapshot
        # without ever closing it out from under a request already using it.
        with engine_lock:
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
        """Accept an index upload only when it matches the active archive.

        Disabled in snapshot mode: see ``_is_manual_upload_allowed``. That
        mode's zero-downtime watcher thread is the only thing besides a
        request that ever touches ``self.server.engine`` outside this
        request-handling thread, and this codepath predates -- and does not
        take -- the shared ``engine_lock``, so the two must stay mutually
        exclusive.
        """

        if not _is_manual_upload_allowed(self.server):
            self._send_json(
                {
                    "error": (
                        "Manual index upload is disabled while running in snapshot mode. "
                        "Build a new snapshot with build_snapshot.py instead; the running "
                        "service picks it up automatically."
                    )
                },
                HTTPStatus.BAD_REQUEST,
            )
            return

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

    def _index_summary(self) -> dict[str, Any]:
        """Describe the exact index currently used by the live engine."""

        engine: AutocompleteEngine = getattr(self.server, "engine")
        return {
            "index_name": getattr(self.server, "active_index_name"),
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


def _load_initial_snapshot_engine(snapshots_dir: Path) -> tuple[AutocompleteEngine, str]:
    """Load whatever snapshot CURRENT names at startup, in zero-downtime mode."""

    version_id = read_current_snapshot(snapshots_dir)
    if version_id is None:
        raise SystemExit(
            "No snapshot is activated under {0}. Run build_snapshot.py first, e.g.:\n"
            "  py build_snapshot.py --archive \"Archive (2).zip\" --snapshots-dir {0}".format(
                snapshots_dir
            )
        )
    index_path = snapshot_index_path(snapshots_dir, version_id)
    engine = AutocompleteEngine.open_snapshot(index_path)
    return engine, version_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Local web UI for autocomplete.py")
    parser.add_argument("--archive", type=Path, default=Path("Archive (2).zip"))
    parser.add_argument("--index", type=Path, default=Path("index.sqlite3"))
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument(
        "--snapshots-dir",
        type=Path,
        default=None,
        help=(
            "Run in zero-downtime mode: serve whichever snapshot CURRENT names under this "
            "directory, and hot-swap live whenever build_snapshot.py activates a new one. "
            "Overrides --index and --rebuild-index."
        ),
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=DEFAULT_POLL_INTERVAL_SECONDS,
        help="How often to check for a newly activated snapshot in --snapshots-dir mode.",
    )
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

    print("Preparing autocomplete index...", flush=True)
    started_at = time.perf_counter()
    watcher: Optional[SnapshotWatcher] = None
    if arguments.snapshots_dir is not None:
        engine, active_index_name = _load_initial_snapshot_engine(arguments.snapshots_dir)
        print(
            "Loaded snapshot {0}: {1:,} lines ready in {2:.1f}s. Watching {3} for updates.".format(
                active_index_name,
                engine.indexed_sentence_count,
                time.perf_counter() - started_at,
                arguments.snapshots_dir,
            )
        )
    else:
        engine = AutocompleteEngine.from_archive(
            arguments.archive, index_path=arguments.index, rebuild_index=arguments.rebuild_index
        )
        active_index_name = arguments.index.name
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
    setattr(server, "engine_lock", threading.Lock())
    setattr(server, "persistent_index_path", arguments.index.resolve())
    setattr(server, "active_index_name", active_index_name)
    setattr(server, "snapshots_dir", arguments.snapshots_dir)
    server_upload_directory = tempfile.TemporaryDirectory(prefix="autocomplete-upload-")
    setattr(server, "upload_directory", Path(server_upload_directory.name))
    setattr(server, "log_directory", arguments.log_dir)

    if arguments.snapshots_dir is not None:
        watcher = SnapshotWatcher(
            server, arguments.snapshots_dir, getattr(server, "engine_lock"), arguments.poll_interval_seconds
        )
        watcher.start()

    url = "http://{0}:{1}".format(*address)
    print("Open {0} — press Ctrl+C to stop.".format(url))
    if not arguments.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping web UI.")
    finally:
        if watcher is not None:
            watcher.stop()
        server.server_close()
        getattr(server, "engine").close()
        server_upload_directory.cleanup()


if __name__ == "__main__":
    main()
