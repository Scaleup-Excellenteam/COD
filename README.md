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

## Semantic embeddings: Earth-side preparation

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

The complete architecture is:

```text
EARTH — offline

Corpus
→ semantic_dataset.jsonl
→ Gemini 768D embeddings
→ semantic_embeddings.jsonl
→ FAISS IndexFlatIP + metadata
→ upload deployment artifacts


GROUND — future online

User query
→ optional Gemini query embedding
→ original text + 768D vector
→ satellite


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

FAISS search runs entirely locally on the satellite; Gemini never runs there
and no API key or network access is required. Corpus and query embeddings must
use compatible Gemini embedding configuration (currently
`gemini-embedding-2`, 768 dimensions). Part A remains independent and retains
its own matching and score. A semantic result contains `sentence`,
`source_text`, `offset`, and a cosine `semantic_score`, which is distinct from
the Part A score.

Ground-side query embedding, the admin toggle, networking, and UI integration
remain future work and are not part of the local semantic runtime.

## Benchmark against the supplied corpus

```powershell
py benchmark.py
```

This builds a fresh temporary index, then reports the separate build time and
the response time for exact, normalized, typo, and short-query cases.
