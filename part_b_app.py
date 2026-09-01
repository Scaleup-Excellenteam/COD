"""Small local Part B demonstration app. Run: python part_b_app.py"""

from __future__ import annotations

import json
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from autocomplete import AutocompleteEngine
from semantic.gemini_embeddings import create_client
from semantic.search import SemanticSearchEngine
from semantic.semantic_query import SemanticQueryService


ROOT = Path(__file__).resolve().parent
MAX_QUERY_LENGTH = 2_000

PAGE = """<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Google Autocomplete — Part B</title>
<style>body{font:16px system-ui,sans-serif;max-width:900px;margin:40px auto;padding:0 20px;color:#222}input{width:min(560px,100%);padding:9px}button{padding:9px 16px;margin:4px}section{margin-top:32px;border-top:1px solid #bbb}article{padding:12px 0;border-bottom:1px solid #ddd}.meta{color:#555;font-size:.9em;margin-top:5px}.error{color:#a00}</style>
<main><h1>Google Autocomplete — Part B</h1><label>Query<br><input id="query" autofocus></label>
<p>Search mode<br><label><input type="radio" name="mode" value="classic" checked> Classic</label><label><input type="radio" name="mode" value="semantic"> Semantic</label><label><input type="radio" name="mode" value="both"> Both</label></p>
<button id="search">Search</button><div id="results" aria-live="polite"></div></main>
<script>
const q=document.querySelector('#query'), out=document.querySelector('#results');
const esc=v=>{const e=document.createElement('span');e.textContent=v;return e.innerHTML};
function list(title,items,score){return `<section><h2>${title}</h2>${items.length?items.map((x,i)=>`<article><strong>${i+1}. ${esc(x.sentence)}</strong><div class="meta">Source: ${esc(x.source)} &nbsp; Offset: ${x.offset} &nbsp; ${score}: ${Number(x.score).toFixed(score==='Semantic Similarity'?4:0)}</div></article>`).join(''):'<p>No results.</p>'}</section>`}
async function search(){const query=q.value.trim();if(!query){out.innerHTML='<p>Please enter a query.</p>';return}const mode=document.querySelector('input[name=mode]:checked').value;out.textContent='Searching...';try{const r=await fetch('/api/search',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query,mode})});const d=await r.json();if(!r.ok)throw Error(d.error||'Search failed.');let html='';if(d.classic)html+=list('Classic Results',d.classic,'Part A Score');if(d.semantic)html+=list('Semantic Results',d.semantic,'Semantic Similarity');if(d.semantic_error)html+=`<section><h2>Semantic Results</h2><p class="error">Semantic search unavailable: ${esc(d.semantic_error)}</p></section>`;out.innerHTML=html}catch(e){out.innerHTML=`<p class="error">${esc(e.message)}</p>`}}
document.querySelector('#search').onclick=search;q.onkeydown=e=>{if(e.key==='Enter')search()};
</script></html>"""


def safe_semantic_error(error: Exception) -> str:
    """Never return provider details or configuration values to the browser."""

    if isinstance(error, ValueError):
        return "Please enter a valid query."
    return "Gemini query embedding failed."


class Handler(BaseHTTPRequestHandler):
    server: HTTPServer

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self._send(PAGE.encode(), "text/html; charset=utf-8")

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/search":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(size).decode())
            query, mode = payload.get("query"), payload.get("mode")
            if not isinstance(query, str) or not query.strip() or len(query) > MAX_QUERY_LENGTH:
                raise ValueError
            if mode not in {"classic", "semantic", "both"}:
                raise ValueError
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
            self._json({"error": "Enter a valid query and mode."}, HTTPStatus.BAD_REQUEST)
            return

        answer: dict[str, Any] = {}
        if mode in {"classic", "both"}:
            engine: AutocompleteEngine = getattr(self.server, "classic_engine")
            answer["classic"] = [{"sentence": item.completed_sentence, "source": item.source_text,
                                  "offset": item.offset, "score": item.score}
                                 for item in engine.get_best_k_completions(query)]
        if mode in {"semantic", "both"}:
            service: SemanticQueryService | None = getattr(self.server, "semantic_service")
            if service is None:
                answer["semantic_error"] = getattr(self.server, "semantic_startup_error")
            else:
                try:
                    answer["semantic"] = [{"sentence": item.sentence, "source": item.source_text,
                                           "offset": item.offset, "score": item.semantic_score}
                                          for item in service.search(query)]
                except Exception as error:  # Provider errors must not affect Classic mode.
                    answer["semantic_error"] = safe_semantic_error(error)
        self._json(answer)

    def log_message(self, _format: str, *_args: object) -> None:
        pass

    def _json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        self._send(json.dumps(payload).encode(), "application/json; charset=utf-8", status)

    def _send(self, content: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)


def main() -> None:
    print("Loading existing Part A index...", flush=True)
    classic = AutocompleteEngine.from_archive(ROOT / "Archive (2).zip", index_path=ROOT / "index.sqlite3")
    semantic_service: SemanticQueryService | None = None
    semantic_startup_error = "Semantic artifacts or Gemini configuration are unavailable."
    try:
        # Satellite: load existing FAISS artifacts once. Ground: create Gemini client.
        satellite = SemanticSearchEngine.from_files(ROOT / "data/semantic.faiss", ROOT / "data/semantic_metadata.jsonl")
        semantic_service = SemanticQueryService(satellite, create_client())
    except Exception:
        print("Semantic search unavailable: " + semantic_startup_error, flush=True)

    server = HTTPServer(("127.0.0.1", 8001), Handler)
    setattr(server, "classic_engine", classic)
    setattr(server, "semantic_service", semantic_service)
    setattr(server, "semantic_startup_error", semantic_startup_error)
    url = "http://127.0.0.1:8001"
    print("Open " + url + " — press Ctrl+C to stop.", flush=True)
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        classic.close()


if __name__ == "__main__":
    main()
