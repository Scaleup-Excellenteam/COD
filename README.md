# Google Autocomplete — Part A

## Run

Use Python 3.12 (or another Python build with SQLite FTS5 trigram support):

```powershell
py autocomplete.py
```

The supplied archive name is already the default. To specify another archive:

```powershell
py autocomplete.py --archive "Archive (2).zip"
```

The program performs the required offline phase first. By default it stores a
persistent SQLite FTS5 index in `index.sqlite3`: if it matches the current ZIP,
the next run loads it immediately; otherwise it is created or rebuilt.

```powershell
# Force a fresh index build.
py autocomplete.py --rebuild-index

# Use a named persistent index file.
py autocomplete.py --index "my-index.sqlite3"

# Keep the former behavior: use a temporary index and delete it on exit.
py autocomplete.py --temporary-index
```

During the online phase, each Enter appends text to the active query. Enter `#`
by itself to reset the query completely.

## Local web interface

For a clean browser interface that uses the same engine and persistent index:

```powershell
py web_app.py
```

The browser opens `http://127.0.0.1:8000`. The server listens on this computer
only, and typed search text is never sent to an external service. Press `Ctrl+C`
in the terminal to stop it. Use `--rebuild-index` to refresh its index, or
`--port 8080` to select another local port.

## Zero-downtime indexing (adding a data source live)

The offline build and the online server are decoupled through the
filesystem, so a new or updated data source can be indexed and published
while the server keeps answering requests -- no restart, no dropped
suggestions.

1. Build and activate a new snapshot from an archive. Each run writes into
   its own versioned directory under `--snapshots-dir` and only then
   atomically flips a `CURRENT` pointer file to it:

   ```powershell
   py build_snapshot.py --archive "Archive (2).zip" --snapshots-dir snapshots
   ```

2. Run the web server in snapshot mode, pointed at the same directory:

   ```powershell
   py web_app.py --snapshots-dir snapshots
   ```

   On a background thread it polls `CURRENT` every `--poll-interval-seconds`
   (default 2s). While the server is running, add a new data source by
   building another snapshot from a new or updated archive (step 1 again,
   pointed at the same `--snapshots-dir`) -- locally, or copied in remotely
   by whatever means already gets a new archive onto the machine. As soon as
   the build finishes and validates, the watcher notices the pointer moved,
   loads the new snapshot into memory, and atomically swaps the engine the
   server queries. In-flight requests finish against the snapshot they
   started with; every request after the swap sees the new one.

   A failed or half-finished build never affects the running service: the
   `CURRENT` pointer only ever moves after a build fully succeeds, so a bad
   archive just leaves the previous, good snapshot serving.

Manual index upload from the browser (`--index`, no `--snapshots-dir`) is
still available for a single fixed archive, but is disabled while running in
snapshot mode -- see `build_snapshot.py` instead.

### Voice typing and Hebrew translation

In a browser that supports the Web Speech API (commonly Chrome or Edge), select
**Voice** beside the search field, allow microphone access, and speak a search
phrase. Interim text appears in the field, and one local search runs after the
browser returns a final transcript. It uses no API key or server dependency.

When typed text or a voice transcript is Hebrew, the local server translates it
to English before searching. Choose **English** or **עברית** from **Voice
language** before recording; the browser's preferred language is selected
initially. The translation model is a one-time download and uses no cloud
translation API:

```powershell
py -m pip install -r requirements.txt
py web_app.py --install-hebrew-translation-model
```

Speech recognition is supplied by the browser, not by this local Python server.
Depending on the browser and its configuration, audio may be processed by that
browser's speech service. The Argos Hebrew-to-English translation step itself
runs locally after installation.

## Why queries are fast

The archive contains more than three million lines. Scanning every line after
every Enter would be slow. The index stores overlapping three-character parts
of every normalized line. For a query of six or more characters the engine
searches two non-overlapping anchors: one from the beginning and one from the
end of the query.

With at most one edit, at least one of those anchors remains unchanged inside a
legal matching sentence. SQLite therefore returns only the lines that contain
one of the anchors. The Python matcher then verifies each returned candidate
exactly. This preserves correctness while avoiding a full-corpus scan for
normal-length input.

If five exact matches exist, they are returned immediately: exact matches always
outscore a result containing one edit. For two- to five-character input, the
engine directly generates the legal corrected substrings and looks them up. A
one- or two-character corrected substring is inherently non-selective, so only
these rare low-information cases may use a fallback scan.

## Required API

```python
from pathlib import Path
from autocomplete import initialize, get_best_k_completions

initialize(Path("Archive (2).zip"))
results = get_best_k_completions("to pe")
```

Each value in `results` is an `AutoCompleteData` object with
`completed_sentence`, `source_text`, `offset`, and `score`.

## Test

```powershell
py -m unittest -v
```

The test suite covers normalization, the official substitution/deletion/
insertion scoring examples, original source/line reporting, alphabetical tie
breaking, and empty input.

## Benchmark against the supplied corpus

```powershell
py benchmark.py
```

This builds a fresh temporary index, then reports the separate build time and
the response time for exact, normalized, typo, and short-query cases.
