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

The program performs the required offline phase first: it reads every `.txt`
file inside the ZIP, normalizes its non-empty lines, and builds temporary
SQLite FTS5 indexes. The indexes are deleted when the program closes, so they
are rebuilt at the start of every run.

During the online phase, each Enter appends text to the active query. Enter `#`
by itself to reset the query completely.

## Debug profiling

To find a performance bottleneck without changing search results, run:

```powershell
py autocomplete.py --debug-profile
```

The debug output separates database setup, ZIP reading/normalization/insertion,
alphabetical-index creation, and FTS-index construction. For every Enter it
also reports normalization time, exact-SQL time, the selected query path,
candidate rows examined, matcher time, and total response time.

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
