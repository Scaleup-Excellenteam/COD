# PRD: Gemini Semantic Embeddings — Earth Preparation and Satellite Retrieval

## 1. Product Summary

Part A implements classic autocomplete over a large text corpus using the existing SQLite/FTS5 index and `OneEditMatcher`.

Part B adds a separate semantic-search capability using Gemini embeddings.

The system represents a data center running on a satellite. Because Gemini is an external Earth-side service, corpus embeddings are generated before deployment and uploaded to the satellite as a prepared data artifact.

The final architecture separates three concerns:

```text
1. EARTH — Offline corpus preparation

Corpus
→ Part A SQLite index
→ semantic_dataset.jsonl
→ Gemini document embeddings
→ semantic_embeddings.jsonl
→ upload semantic data to satellite


2. EARTH / GROUND — Future request preparation

User query
→ semantic mode enabled?
   ├─ No  → send original text
   └─ Yes → Gemini query embedding
             ↓
           send original text + query vector


3. SATELLITE — Runtime

Original text
→ Part A search
→ Classic Top 5

Query vector
→ local semantic embedding index
→ vector similarity
→ Semantic Top 5
```

The satellite itself does not call Gemini.

---

# 2. Assignment Goal

This work implements the Gemini Embeddings semantic-search feature.

The complete feature must eventually:

* create vector representations for corpus sentences
* create a compatible vector representation for the user query
* retrieve corpus sentences according to semantic similarity
* return the original corpus sentence
* return the real source file and line number
* display semantic ranking separately from Part A scoring
* demonstrate queries expressing the same idea using different wording

This PRD focuses on building the corpus embedding infrastructure.

Query embeddings and semantic Top-5 retrieval are intentionally implemented in later PRDs.

---

# 3. Existing Architecture

## Part A

Part A remains unchanged:

```text
Corpus ZIP
    ↓
SQLite sentence index
    ↓
FTS5 candidate retrieval
    ↓
OneEditMatcher
    ↓
Part A edit score
    ↓
Top 5
```

Part A is fully local and independent from Gemini.

The following must not be changed as part of this PRD:

* `autocomplete.py`
* `AutocompleteEngine`
* `get_best_k_completions()`
* Part A SQLite schema
* Part A scoring
* Part A matching behavior

---

# 4. Existing Part B Foundation

The first semantic preparation stage already creates:

```text
data/semantic_dataset.jsonl
```

from the Part A SQLite index.

Flow:

```text
Existing SQLite index
        ↓
semantic.build_dataset
        ↓
semantic_dataset.jsonl
```

Each record contains:

```json
{
  "id": 1,
  "sentence": "Original corpus sentence.",
  "source_text": "source/file.txt",
  "offset": 25
}
```

The semantic dataset deliberately preserves the original sentence rather than Part A's normalized search representation.

---

# 5. Current PRD Goal

Generate one Gemini embedding for every selected corpus sentence and store the result locally.

Input:

```text
data/semantic_dataset.jsonl
```

Output:

```text
data/semantic_embeddings.jsonl
```

Conceptually:

```text
semantic_dataset.jsonl
        ↓
one sentence
        ↓
Gemini Embedding API
        ↓
768-dimensional vector
        ↓
semantic_embeddings.jsonl
```

This is an Earth-side offline/pre-deployment pipeline.

It is not satellite runtime code.

---

# 6. Deployment Architecture

## 6.1 Earth — Offline Corpus Preparation

Corpus processing happens before deployment:

```text
EARTH

Corpus
    ↓
Part A index
    ↓
semantic_dataset.jsonl
    ↓
Gemini
    ↓
corpus embeddings
    ↓
semantic_embeddings.jsonl
    ↓
future semantic index
    ↓
upload to satellite
```

The expensive operation of embedding millions of corpus sentences happens on Earth.

The satellite must never regenerate the entire corpus embedding dataset.

---

## 6.2 Ground — Future Runtime Request Preparation

The user request reaches the ground system before it reaches the satellite.

An admin/configuration setting controls whether semantic search is enabled.

### Classic mode

```text
semantic_enabled = false

User query
→ Ground
→ original text sent to satellite
→ Part A
```

No Gemini request is necessary.

### Semantic mode

```text
semantic_enabled = true

User query
      ├────────────────→ original text
      │
      └→ Gemini
           ↓
        query embedding
           ↓

original text + query embedding
           ↓
       Satellite
```

Conceptual request:

```json
{
  "query": "how do computers learn?",
  "semantic_enabled": true,
  "query_embedding": [
    "... 768 numeric values ..."
  ]
}
```

This request format is future work and is not implemented by this PRD.

---

## 6.3 Satellite — Future Runtime

The satellite will eventually contain:

```text
Part A index
+
precomputed corpus embeddings / semantic index
```

Runtime:

```text
SATELLITE

original query
    ↓
Part A
    ↓
Classic Top 5


query embedding
    ↓
local corpus embedding index
    ↓
similarity calculation
    ↓
Semantic Top 5
```

Similarity computation happens locally on the satellite.

Gemini does not run on the satellite.

---

# 7. Failure Architecture

Part A must remain operational independently from Gemini.

Future expected behavior:

```text
Gemini available
→ Classic mode works
→ Semantic mode works


Gemini unavailable
→ query embedding cannot be created
→ Semantic mode unavailable for that request
→ Classic Part A still works
```

This separation is an important architectural requirement.

---

# 8. Embedding Compatibility Invariant

Corpus embeddings and future query embeddings must exist in the same vector space.

The project therefore fixes:

```text
Model:      gemini-embedding-2
Dimensions: 768
```

Future query embedding generation must use a compatible model and dimensionality.

Changing the embedding model later requires rebuilding corpus embeddings.

The model and dimensions must therefore remain obvious constants in the implementation.

---

# 9. Gemini Embedding Configuration

Use:

```python
MODEL = "gemini-embedding-2"
EMBEDDING_DIMENSIONS = 768
```

Use the official Google GenAI Python SDK.

Gemini credentials must come from the existing:

```python
semantic.config.get_gemini_api_key()
```

The API key must never:

* be hard-coded
* be committed
* appear in JSONL
* appear in terminal output
* appear in logs
* appear in errors
* be sent to browser/client JavaScript

`.env` remains ignored by Git.

`.env.example` remains tracked.

---

# 10. Corpus Embedding Format

Each output record must contain exactly:

```text
id
sentence
source_text
offset
embedding
```

Example:

```json
{
  "id": 123,
  "sentence": "Machine learning can identify patterns in data.",
  "source_text": "books/example.txt",
  "offset": 84,
  "embedding": [
    0.012,
    -0.041,
    0.083
  ]
}
```

The real vector contains exactly:

```text
768 dimensions
```

Requirements:

* preserve `id`
* preserve the original `sentence`
* preserve `source_text`
* preserve physical `offset`
* preserve input order
* embedding contains exactly 768 numeric values
* do not include Part A `normalized`
* UTF-8 JSONL
* readable Unicode

---

# 11. Retrieval Document Formatting

Corpus sentences are the document side of future semantic retrieval.

The text passed to Gemini should therefore use the retrieval-document formatting chosen for the project:

```text
title: none | text: {sentence}
```

Example original sentence:

```text
Python is a programming language.
```

API input:

```text
title: none | text: Python is a programming language.
```

Stored output remains:

```json
{
  "sentence": "Python is a programming language."
}
```

The retrieval prefix is API input only.

Never modify the original stored sentence.

Future user queries will use the appropriate query-side retrieval formatting. That is outside this PRD.

---

# 12. One Sentence = One Request

V1 intentionally uses the simplest possible model:

```text
one corpus sentence
        ↓
one Gemini request
        ↓
one embedding
```

Do not batch sentences.

This is intentionally not optimized yet.

Batching can be introduced later without changing the semantic dataset format.

---

# 13. Input

Default input:

```text
data/semantic_dataset.jsonl
```

Each selected input record must contain:

```text
id
sentence
source_text
offset
```

The embedding builder must use this JSONL.

Do not:

* reopen the corpus ZIP
* reread SQLite
* duplicate corpus extraction
* regenerate the semantic dataset internally

The pipeline boundary must remain:

```text
SQLite
→ semantic_dataset.jsonl
→ semantic_embeddings.jsonl
```

---

# 14. CLI

Add:

```text
semantic/build_embeddings.py
```

Example:

```powershell
python -m semantic.build_embeddings --limit 10
```

Support:

```text
--input PATH
--output PATH
--limit POSITIVE_INTEGER
--all
```

Defaults:

```text
--input  data/semantic_dataset.jsonl
--output data/semantic_embeddings.jsonl
```

`--limit` and `--all` are mutually exclusive.

One of them must be explicitly supplied.

This is an intentional safety mechanism.

The command must never accidentally embed millions of sentences merely because no limit was provided.

Example small experiment:

```powershell
python -m semantic.build_embeddings --limit 10
```

Explicit full-corpus run:

```powershell
python -m semantic.build_embeddings --all
```

---

# 15. Free-Tier / Quota Safety

This first implementation is intended for very small experiments.

Initial real smoke tests should use:

```text
1–3 sentences
```

not thousands.

The implementation must:

* require explicit `--limit` or `--all`
* display how many records will be embedded
* never automatically switch to the full corpus
* make one request per sentence
* fail normally when Gemini rejects a request

V1 intentionally does not implement:

* retry
* throttling
* quota recovery
* batching
* parallel requests

Before running large experiments, the developer is responsible for checking the currently active Gemini project quota.

---

# 16. Proposed Modules

```text
semantic/
├── __init__.py
├── config.py
├── build_dataset.py
├── gemini_embeddings.py
└── build_embeddings.py
```

---

# 17. `gemini_embeddings.py`

This module owns the Gemini API boundary.

It is explicitly:

```text
EARTH-SIDE
OFFLINE
PRE-DEPLOYMENT
```

It must not contain satellite runtime behavior.

Important constants:

```python
MODEL = "gemini-embedding-2"
EMBEDDING_DIMENSIONS = 768
```

Suggested responsibilities:

```python
prepare_document_text(sentence: str) -> str
```

and:

```python
embed_sentence(client, sentence: str) -> list[float]
```

---

## `prepare_document_text()`

Transform:

```text
Python is a programming language.
```

into:

```text
title: none | text: Python is a programming language.
```

Requirements:

* do not alter the caller's sentence
* no normalization
* no punctuation removal
* no lowercase transformation
* no truncation

---

## `embed_sentence()`

Flow:

```text
sentence
    ↓
prepare_document_text
    ↓
Gemini embed request
    ↓
response
    ↓
strict validation
    ↓
list[float]
```

One call to `embed_sentence()` must result in exactly one Gemini embedding request.

Request:

```text
model = gemini-embedding-2
output dimensions = 768
```

Return:

```text
list[float]
```

with exactly 768 values.

---

# 18. Gemini Response Validation

For every request validate:

```text
exactly one embedding
```

and:

```text
len(embedding) == 768
```

Every value must be numeric.

Boolean values must not be accepted as numeric embedding values.

Reject:

* empty responses
* missing embedding
* unexpected multiple embeddings
* wrong dimensionality
* malformed response
* non-numeric values

Never silently repair malformed Gemini responses.

---

# 19. `build_embeddings.py`

This module owns dataset orchestration.

Flow:

```text
semantic_dataset.jsonl
        ↓
validate selected records
        ↓
create Gemini client
        ↓
read record
        ↓
embed sentence
        ↓
write output record
        ↓
next record
        ↓
successful completion
        ↓
atomic final output
```

Gemini SDK details should remain isolated in `gemini_embeddings.py`.

---

# 20. Input Validation

Before making real Gemini requests, validate the selected records whenever practical.

Each selected record must:

* be valid JSON
* be a JSON object
* contain `id`
* contain `sentence`
* contain `source_text`
* contain `offset`
* contain a nonblank string `sentence`

Do not silently:

* skip malformed records
* replace malformed values
* normalize text
* invent provenance

The implementation must remain streaming-friendly and must not load millions of corpus records into memory.

A validation/count pass followed by the embedding pass is acceptable.

---

# 21. Progress Reporting

During embedding show simple progress:

```text
Embedding: 1 / 10
Embedding: 2 / 10
...
Embedding: 10 / 10
```

Avoid producing unnecessary terminal noise if the progress can be updated cleanly.

On success:

```text
Created semantic embeddings: 10 sentences
Model: gemini-embedding-2
Dimensions: 768
Output: data/semantic_embeddings.jsonl
```

Never display:

* API key
* environment contents
* full vectors during normal operation

---

# 22. Failure Behavior

V1 deliberately uses simple fail-fast behavior.

There are:

```text
NO RETRIES
NO RESUME
NO BATCHING
NO PARALLELISM
```

Example:

```text
100 selected

1  ✓
2  ✓
...
37 ✓
38 ✗
```

Expected behavior:

```text
request 38 fails
        ↓
entire build fails
        ↓
temporary output removed
        ↓
previous valid output remains unchanged
```

The partial 37-record dataset must not replace the existing final output.

A future run starts again from the beginning.

Resume support is deferred.

---

# 23. Atomic Output

Write into a temporary file beside:

```text
data/semantic_embeddings.jsonl
```

Only after every selected record succeeds:

```text
temporary file
      ↓
atomic replace
      ↓
semantic_embeddings.jsonl
```

If:

* Gemini fails
* validation fails
* input is malformed
* output writing fails
* another handled pipeline error occurs

then:

```text
temporary output → delete
existing final output → preserve
```

---

# 24. Long Sentences

Do not automatically truncate or split corpus sentences in V1.

The original physical corpus sentence remains authoritative.

If a sentence cannot be embedded by the selected Gemini model, the build should fail using the normal failure behavior.

Future versions may introduce:

* token checking
* truncation
* chunking

but not this PRD.

---

# 25. Generated Data and Git

Generated embedding data is a local/pre-deployment artifact.

Ignore:

```text
data/semantic_embeddings.jsonl
```

Do not commit it.

Also continue ignoring:

```text
.env
data/semantic_dataset.jsonl
```

Tracked configuration examples and source code remain committed.

---

# 26. Testing Strategy

Automated tests must not consume Gemini quota.

Mock/fake the Gemini API boundary.

Real API calls should only occur in explicit tiny smoke tests.

---

# 27. Required Tests

Focused tests should cover:

### Gemini client

* document retrieval formatting
* original sentence remains unchanged
* one sentence causes exactly one API request
* model is correct
* 768 dimensions requested
* valid 768-dimensional response succeeds
* wrong dimension fails
* malformed response fails
* non-numeric vector fails
* boolean vector value fails

### Embedding dataset builder

* original metadata preserved
* original sentence preserved
* embedding added
* correct JSONL keys
* input order preserved
* deterministic `--limit`
* explicit `--all`
* `--limit` and `--all` mutually exclusive
* missing both rejected
* malformed input rejected

### Failure safety

* Gemini failure stops immediately
* no retry occurs
* previous output survives failed build
* temporary output cleaned after failure
* API key never appears in output

### Regression

All existing tests must remain green:

```text
Part A tests
semantic dataset/config tests
Gemini embedding tests
```

Part A must not be modified to make new tests pass.

---

# 28. Real Smoke Tests

Real Gemini tests should remain intentionally tiny.

## Phase 1

One sentence:

```text
Requests: 1
Expected embeddings: 1
Dimensions: 768
```

Do not print the vector itself.

Verify:

* authentication works
* model works
* one request is made
* vector has 768 values
* secret is not exposed

## Phase 2

Example:

```powershell
python -m semantic.build_embeddings --limit 3
```

Verify:

* 3 API requests
* 3 output records
* 768 values per embedding
* metadata preserved
* JSONL parses successfully
* API key absent

---

# 29. Success Criteria for This PRD

The implementation is complete when:

```powershell
python -m semantic.build_embeddings --limit 3
```

successfully:

1. reads three records from `semantic_dataset.jsonl`
2. preserves original sentences
3. preserves `source_text`
4. preserves `offset`
5. sends each sentence independently to Gemini
6. generates one 768-dimensional vector per sentence
7. stores each vector with its original metadata
8. creates output atomically
9. reports progress
10. does not expose the API key
11. does not modify Part A
12. does not implement satellite runtime search

All automated tests must also pass without real Gemini requests.

---

# 30. Explicitly Deferred Work

Do NOT implement during this PRD:

* query embeddings
* cosine similarity
* semantic Top 5
* vector database
* FAISS
* SQLite vector extensions
* satellite search service
* satellite upload mechanism
* ground request protocol
* semantic-mode admin toggle
* query request serialization
* UI integration
* retries
* exponential backoff
* rate-limit recovery
* batching
* async requests
* parallel requests
* resume/checkpoints
* automatic truncation
* sentence chunking

---

# 31. Next PRD — Satellite Semantic Retrieval

The next major implementation stage will operate on the satellite.

Input:

```text
precomputed corpus embeddings
+
768-dimensional query vector
```

Expected flow:

```text
query vector
    ↓
compare against local corpus vectors
    ↓
semantic similarity
    ↓
rank
    ↓
Top 5
```

Results must include:

```text
sentence
source_text
offset
semantic_score
```

The semantic score must remain distinct from Part A's edit score.

The next PRD should also test paraphrases where the query expresses the same idea using different words.

---

# 32. Later PRD — Ground Query Preparation

After satellite local retrieval works, implement the ground-side request path.

Conceptually:

```text
User query
    ↓
Admin/config semantic toggle
    ↓

semantic disabled
→ send text only


semantic enabled
→ create Gemini query embedding
→ send:
   original text
   + semantic_enabled
   + 768-dimensional vector
→ satellite
```

The satellite should never need direct Gemini credentials.

---

# 33. Final Target Architecture

```text
                         EARTH / GROUND


       OFFLINE                           ONLINE

Corpus sentences                     User query
      ↓                                  ↓
Part A SQLite                      Semantic enabled?
      ↓                            /             \
semantic_dataset.jsonl           no             yes
      ↓                            |              ↓
Gemini document embeddings        |            Gemini
      ↓                            |              ↓
corpus vectors                     |         query vector
      ↓                            |              |
deployment artifact                └──────┬───────┘
      ↓                                   ↓
upload                              request to satellite
      │                                   │
      └────────────────┬──────────────────┘
                       ↓


                    SATELLITE

        ┌───────────────────────────────┐
        │                               │
        │ original text                 │
        │      ↓                        │
        │ Part A                        │
        │      ↓                        │
        │ Classic Top 5                 │
        │                               │
        │ query vector                  │
        │      ↓                        │
        │ local corpus vectors          │
        │      ↓                        │
        │ semantic similarity           │
        │      ↓                        │
        │ Semantic Top 5                │
        │                               │
        └───────────────────────────────┘
```

---

# 34. Phased Implementation Workflow

This PRD must be implemented incrementally.

Do not automatically move between phases.

At the end of every phase:

1. run relevant verification
2. explain the important code
3. explain important design decisions
4. show `git status`
5. STOP
6. wait for explicit approval

After approval:

1. create the phase Git commit
2. show the commit hash and files
3. then begin the next approved phase

---

## Phase 1 — Gemini Embedding Client

### Goal

Prove that one corpus sentence can become one valid 768-dimensional Gemini embedding.

### Implement

```text
semantic/gemini_embeddings.py
Gemini SDK dependency configuration
focused Phase 1 tests
```

### Responsibilities

Implement:

```python
prepare_document_text(sentence: str) -> str
```

and:

```python
embed_sentence(client, sentence: str) -> list[float]
```

Use:

```text
MODEL = gemini-embedding-2
EMBEDDING_DIMENSIONS = 768
```

Use the existing secure Gemini configuration.

No JSONL embedding builder yet.

### Verification

Run automated tests without quota.

Then run one real harmless sentence through Gemini.

Report only:

```text
Model: gemini-embedding-2
Dimensions: 768
Requests: 1
```

Do not print the vector or API key.

Confirm Part A remains unchanged.

### Review gate

STOP after Phase 1.

After approval commit:

```text
feat: add Gemini embedding client
```

---

## Phase 2 — Corpus Embedding Builder

### Goal

Convert selected records from:

```text
semantic_dataset.jsonl
```

into:

```text
semantic_embeddings.jsonl
```

### Implement

```text
semantic/build_embeddings.py
.gitignore update
```

Implement:

* input JSONL reading
* input validation
* explicit `--limit`
* explicit `--all`
* one API request per sentence
* progress display
* metadata preservation
* embedding output
* atomic write
* fail immediately
* no retries
* no resume
* no batching

### Real smoke test

Run:

```powershell
python -m semantic.build_embeddings --limit 3
```

Inspect all three records.

Verify each embedding has 768 values.

### Review gate

STOP after Phase 2.

After approval commit:

```text
feat: build semantic embeddings dataset
```

---

## Phase 3 — Full Focused Tests

### Goal

Cover the complete corpus-embedding pipeline without consuming Gemini quota.

Add focused automated tests for all requirements in this PRD.

Use mocks/fakes only at the external Gemini boundary.

Run the full repository test suite.

Report separately:

```text
Gemini embedding tests
Existing semantic tests
Existing Part A tests
Total
```

Do not alter Part A.

### Review gate

STOP after Phase 3.

After approval commit:

```text
test: add Gemini embedding pipeline tests
```

---

## Phase 4 — Documentation and Final Verification

### Goal

Document the Earth/satellite architecture and embedding workflow.

Update README with concise instructions:

```text
1. configure GEMINI_API_KEY
2. generate semantic_dataset.jsonl
3. generate a tiny embedding sample
4. explain output
5. explain --limit vs --all
6. explain one sentence = one request in V1
7. explain Earth preprocessing
8. explain that corpus embeddings become satellite deployment data
```

Document clearly:

```text
Gemini does not run on the satellite.
```

And:

```text
Part A remains usable without Gemini.
```

Run:

* full automated suite
* one final tiny real smoke test
* Git ignore/security checks

Confirm:

* generated embeddings ignored
* `.env` ignored
* `.env.example` tracked
* `autocomplete.py` unchanged
* no semantic similarity implemented yet
* no satellite runtime implementation yet

### Review gate

STOP after Phase 4.

After approval commit:

```text
docs: document Earth-satellite semantic architecture
```

---

# 35. Status Tracking

New Codex sessions should read this PRD before making changes.

At the start of a new session:

1. read this file completely
2. inspect `git log --oneline -10`
3. inspect `git status`
4. inspect `semantic/`
5. inspect semantic tests
6. determine which phases are already committed
7. continue only from the next unfinished phase

Do not reimplement completed phases.

Do not infer progress only from this PRD; Git history and repository state are the authoritative implementation status.

If a previous phase is implemented but not committed, inspect and review it rather than rebuilding it.


Corpus sentence:
"Children should receive proper education."

Query:
"Kids deserve access to schooling."