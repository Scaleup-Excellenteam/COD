"""Local browser UI for the autocomplete engine.

Run ``py web_app.py`` and open http://127.0.0.1:8000.  The page is served
locally; text typed into it never leaves this computer.
"""

from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import time
from typing import Any
import webbrowser

from autocomplete import AutocompleteEngine


WEB_DIRECTORY = Path(__file__).with_name("web")
MAX_QUERY_LENGTH = 2_000


class AutocompleteWebHandler(BaseHTTPRequestHandler):
    """Serve the static UI and a small JSON API over localhost only."""

    server: HTTPServer

    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
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
        if self.path != "/api/suggestions":
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length > MAX_QUERY_LENGTH * 4:
                raise ValueError("The request is too large.")
            payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
            query = payload.get("query", "")
            if not isinstance(query, str) or len(query) > MAX_QUERY_LENGTH:
                raise ValueError("query must be a short string.")
        except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
            self._send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            return

        if query == "#":
            self._send_json({"suggestions": [], "elapsed_ms": 0.0, "reset": True})
            return

        started_at = time.perf_counter()
        engine: AutocompleteEngine = getattr(self.server, "engine")
        suggestions = engine.get_best_k_completions(query)
        elapsed_ms = (time.perf_counter() - started_at) * 1_000
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
                "reset": False,
            }
        )

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
    parser.add_argument("--archive", type=Path, default=Path("Archive (2).zip"))
    parser.add_argument("--index", type=Path, default=Path("index.sqlite3"))
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--no-browser", action="store_true")
    arguments = parser.parse_args()

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
        engine.close()


if __name__ == "__main__":
    main()
