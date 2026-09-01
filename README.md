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
only, and the search text is never sent to an external service. Press `Ctrl+C`
in the terminal to stop it. Use `--rebuild-index` to refresh its index, or
`--port 8080` to select another local port.

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

## Semantic embeddings: Earth-side preparation and query search

Gemini is used only by offline/pre-deployment tooling on Earth. Part A remains
fully local and usable without Gemini.

Create a local environment file:

```powershell
Copy-Item .env.example .env
```

Set the Gemini API key in `.env`:

```text
GEMINI_API_KEY=<my-key>
```

Build an initial 10,000-row dataset from the existing SQLite index:

```powershell
py -m semantic.build_dataset --index index.sqlite3 --limit 10000
```

The output is `data/semantic_dataset.jsonl`. Dataset generation reads only the
local SQLite index and does **not** contact Gemini.

Create a tiny corpus-embedding sample:

```powershell
py -m semantic.build_embeddings --limit 3
```

This reads the first three dataset records and writes
`data/semantic_embeddings.jsonl`, preserving each original sentence, source
file, and physical line offset while adding a 768-dimensional vector. In V1,
one sentence makes one Gemini request; there is no batching or retry behavior.

Embedding selection is intentionally explicit:

```powershell
# Safe experiment: embed only the first N records.
py -m semantic.build_embeddings --limit 3

# Deliberate full-dataset run: may consume substantial Gemini quota.
py -m semantic.build_embeddings --all
```

Exactly one of `--limit` or `--all` is required, so the complete corpus cannot
be embedded accidentally. Both generated JSONL files are local artifacts and
are ignored by Git. Convert the completed corpus embeddings into the two
satellite deployment artifacts:

```powershell
py -m semantic.build_faiss_index `
    --input data/semantic_embeddings.jsonl `
    --index-output data/semantic.faiss `
    --metadata-output data/semantic_metadata.jsonl
```

This creates an exact FAISS `IndexFlatIP` containing L2-normalized `float32`
vectors and a positionally aligned metadata JSONL without embedding vectors.
Both deployment artifacts are also ignored by Git.

The completed architecture is:

```text
EARTH — offline

Corpus
→ semantic_dataset.jsonl
→ Gemini 768D embeddings
→ semantic_embeddings.jsonl
→ FAISS IndexFlatIP + metadata
→ upload deployment artifacts


GROUND — online

User query
→ Gemini query embedding
→ 768D vector
→ local demonstration boundary


SATELLITE — local runtime

text
→ Part A

query vector
→ local FAISS index
→ cosine similarity
→ Semantic Top 5
```

Satellite semantic retrieval receives an already-generated compatible 768D
query vector and loads its deployment files once:

```python
from semantic.search import SemanticSearchEngine

engine = SemanticSearchEngine.from_files(
    "data/semantic.faiss",
    "data/semantic_metadata.jsonl",
)
results = engine.search(query_embedding)
```

## Completed Earth / Ground / Satellite flow

```text
EARTH OFFLINE
Corpus -> Gemini document embeddings -> FAISS deployment artifact

GROUND ONLINE
User query -> Gemini query embedding -> 768D vector

SATELLITE ONLINE
768D vector -> local FAISS cosine search -> Semantic Top 5
```

The local demo makes that boundary explicit while running both sides in one
process; it does not implement networking:

```powershell
python -m semantic.semantic_query "how do computers learn?"
```

Ground sends exactly one `gemini-embedding-2` query request at 768 dimensions.
Embedding 2 retrieval inputs are compatible: corpus records use
`title: none | text: …`, while queries use `task: search result | query: …`.
The original query text is preserved and vectors are never displayed.

FAISS search runs entirely locally on the satellite: it makes zero Gemini calls
and needs neither a Gemini key nor network access. Corpus sentence, source,
and offset are real preserved dataset values. A semantic result contains
`sentence`, `source_text`, `offset`, and cosine `semantic_score`; it is never
combined with Part A's independent character/edit-match score.

Evaluate a deliberate paraphrased query against real deployment artifacts:

```powershell
python -m semantic.evaluate_semantic "the file for chapter one" `
    --expected-sentence "ch01.qxd" `
    --expected-idea "chapter one document file"
```

The evaluator prints the query, expected corpus idea, local Top-5 results,
rank of the expected sentence, and its semantic score. Automated tests mock
Gemini and consume no quota. Networking, an admin toggle, web UI integration,
and classic-plus-semantic result combination remain future work.

## Benchmark against the supplied corpus

```powershell
py benchmark.py
```

This builds a fresh temporary index, then reports the separate build time and
the response time for exact, normalized, typo, and short-query cases.
