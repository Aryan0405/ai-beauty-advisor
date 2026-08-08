# AI Beauty Advisor

A semantic-search skincare recommendation system. Describe a skin concern or
goal in plain language; the backend embeds the query, retrieves the closest
products from a ~1,472-item Kaggle cosmetics dataset via FAISS, re-ranks them
with a small hybrid (semantic + keyword + filter) score, and asks Gemini 2.5
Flash for one grounded, per-product explanation in a single structured-output
call.

Full functional/non-functional requirements live in [PROJECT_SPEC.md](PROJECT_SPEC.md);
this file is the practical "how do I run/use it" reference.

## Architecture

```
┌───────────────────────────┐
│   React + Vite SPA         │   :5173 (dev)  ·  :8080 behind nginx (Docker)
│   search box, result       │
│   cards, per-card           │
│   explanation, filters      │
└──────────────┬──────────────┘
               │ JSON over HTTP
               │   POST /api/v1/recommend
               │   GET  /api/v1/products/{id}
               │   GET  /api/v1/health
┌──────────────▼──────────────────────────────────────────┐
│                     FastAPI backend  :8000                │
│                                                            │
│  RequestIDMiddleware ──▶ CORSMiddleware ──▶ router         │
│  (assigns/propagates          structured error envelope    │
│   X-Request-ID, logs           for every non-2xx response   │
│   request start/finish)        {"error":{"code","message"}} │
│                                                            │
│  ┌──────────────────────────────────────────────────┐    │
│  │ services/recommendation.py  (hybrid ranking)        │    │
│  │   embed query → FAISS search → SQLite fetch →       │    │
│  │   rank by 0.9·similarity + 0.075·keyword + 0.025·filter │
│  └───────────┬───────────────────────┬────────────────┘    │
│              │                       │                     │
│     ┌────────▼────────┐    ┌─────────▼─────────┐           │
│     │ FAISS index       │    │ SQLite             │           │
│     │ IndexFlatIP,       │    │ products table     │           │
│     │ cosine similarity  │    │ (metadata, filters) │           │
│     └───────────────────┘    └─────────────────────┘           │
│                                                            │
│  ┌──────────────────────────────────────────────────┐    │
│  │ services/explanation.py                             │    │
│  │   one Gemini 2.5 Flash call, structured JSON output, │    │
│  │   → {product_id: explanation} for the whole batch    │    │
│  │   (null per-product on failure/timeout/omission --    │    │
│  │   never fails the request)                            │    │
│  └──────────────────────────────────────────────────┘    │
└────────────────────────────────────────────────────────────┘

Offline, run once (or whenever the source CSV changes):
  data/cosmetics.csv
    → backend/app/db/ingest.py              clean rows, load SQLite
    → backend/ingestion/build_embeddings.py Sentence-Transformers embeddings
    → backend/ingestion/build_index.py      build + persist the FAISS index,
                                             sync SQLite embedding_id <-> vector
```

**Stack:** FastAPI (Python 3.11) · SQLite · FAISS (`faiss-cpu`, `IndexFlatIP`)
· Sentence-Transformers (`all-MiniLM-L6-v2`) · Gemini 2.5 Flash
(`google-genai`) · React 19 + Vite · Docker / docker-compose.

## Features

- **Natural-language search** — free-text query, no need to pick filters first.
- **Hybrid ranking** — FAISS cosine similarity blended with a lexical keyword
  overlap score and an optional filter-match bonus, not similarity alone.
- **Optional structured filters** — `category`, `brand`, `skin_type` (exact,
  case-insensitive match; `skin_type` may be a comma-separated list, all of
  which must be present on the product).
- **Per-product LLM explanations** — one Gemini call per search (not one per
  product) returns a structured batch of grounded, per-product explanations;
  a product's `explanation` is `null` rather than blocking the response if
  Gemini fails, times out, or omits it.
- **Graceful degradation everywhere** — retrieval failures return `503`,
  Gemini failures degrade to `null` explanations with a `200`, validation
  failures return `400` — never a raw crash or stack trace to the client.
- **Structured JSON logging with request correlation** — every request gets
  a UUID4 (or client-supplied) ID, echoed back as `X-Request-ID` and attached
  to every log line for that request.
- **Dockerized** — non-root backend and frontend containers, healthchecked,
  orchestrated by a single `docker-compose.yml`.
- **94 automated tests**, all offline (FAISS/DB/Gemini mocked or run
  against small real fixtures) — see [Testing](#testing).

## Quickstart

### Option A: Docker (recommended)

```bash
cp .env.example .env   # then edit .env and set a real GEMINI_API_KEY
docker compose up -d --build
```

- Frontend: http://localhost:5173
- Backend: http://localhost:8000 (health: http://localhost:8000/api/v1/health)

```bash
docker compose logs -f backend   # structured JSON logs, one line per event
docker compose down
```

This is the only path that needs no manual setup beyond `.env` — both images
build from scratch, the backend waits to be marked healthy before the
frontend starts, and `data/` is bind-mounted in (see [Data](#data) below).

### Option B: Local development (no Docker)

```bash
# Backend
pip install -r backend/requirements-dev.txt   # prod deps + pytest/httpx for testing
cp .env.example .env   # set a real GEMINI_API_KEY
uvicorn backend.app.main:app --reload --port 8000

# Frontend (separate terminal)
cd frontend
npm install
npm run dev
```

`backend/requirements.txt` alone is enough to just run the app; use
`backend/requirements-dev.txt` (which includes it) if you also want to run
`pytest`.

### Data

Both paths need a populated `data/` directory: `data/beauty_advisor.db`
(SQLite) and `data/index/products.faiss` + `data/index/product_ids.json`
(FAISS). If you're starting from the raw Kaggle CSV instead of a pre-built
`data/`, run the offline ingestion pipeline once from the repo root:

```bash
python -m backend.app.db.ingest              # CSV -> SQLite
python -m backend.ingestion.build_embeddings # SQLite -> embeddings
python -m backend.ingestion.build_index      # embeddings -> FAISS index
```

Each step is idempotent and safe to re-run after the source CSV changes.

## API

Base path: `/api/v1`. Every non-2xx response is
`{"error": {"code": "string", "message": "string"}}`.

### `POST /recommend`

```json
{
  "query": "lightweight moisturizer for oily acne-prone skin",
  "top_k": 5,
  "filters": { "category": "Moisturizer", "skin_type": "Oily" }
}
```

`filters` is optional (`category` / `brand` / `skin_type`). `top_k` is
optional, defaults to 5, must be 1-20.

```json
{
  "query": "lightweight moisturizer for oily acne-prone skin",
  "count": 1,
  "recommendations": [
    {
      "product_id": "cosmetics-000123",
      "name": "string", "brand": "string", "category": "string",
      "skin_type": "string", "ingredients": "string", "description": "string",
      "price": 24.99, "source_rank": 4.5,
      "similarity_score": 0.87, "match_score": 0.83,
      "explanation": "string | null"
    }
  ]
}
```

Errors: `400 invalid_request` (empty/whitespace query, `top_k` out of
range) · `503 service_unavailable` (FAISS/index unavailable) ·
`500 internal_error` (unhandled).

### `GET /products/{product_id}`

Returns the same product fields as above, minus the ranking-specific ones
(`similarity_score`, `match_score`, `explanation`). `404 not_found` if the ID
doesn't exist.

### `GET /health`

```json
{ "status": "ok", "index_loaded": true, "db_connected": true }
```

## Testing

```bash
pytest              # 94 tests, offline, ~2s -- see backend/tests/
```

No network calls, no live model downloads, no real Gemini calls: FAISS/DB
use small real fixtures per-test, Gemini is a fake `google.genai.Client`.
Covers repository queries, FAISS build/search/persistence, hybrid ranking
and filter logic, Gemini prompt construction/parsing/timeout/fallback, every
API endpoint's status codes and error envelope, the request-ID middleware,
and startup config failure handling (e.g. a missing `GEMINI_API_KEY` fails
fast with one clear line instead of a stack trace).

### Evaluation scripts (manual/QA, not part of `pytest`)

These need the real embedding model, FAISS index, and SQLite DB (`data/`),
so they're deliberately not named `test_*.py` and are never collected by
pytest / run in CI.

```bash
python -m backend.tests.eval.run_latency_eval     # retrieval latency vs. NFR budget
python -m backend.tests.eval.run_precision_eval   # precision@5 relevance spot-check
```

**Latency** (`run_latency_eval.py`, 20 curated queries, local machine, warm
cache): retrieval-only (embed query → FAISS search → SQLite fetch → hybrid
rank, excluding the LLM call) — **P50 9.5ms / P95 13.0ms**, against a spec
budget of P95 ≤ 150ms. Pass `--with-llm` to also measure the full
search+explanation path (budget P95 ≤ 3.5s) — off by default since it makes
a real Gemini call.

**Precision@5** (`run_precision_eval.py`): each curated query is paired with
a hand-authored expected `category` (+ optional `skin_type`) — see
`backend/tests/eval/relevance_cases.py` for the reasoning behind each one,
including a real data quirk it surfaced (the dataset has no dedicated
"Serum" category; products literally named "... Serum" are filed under
either `Treatment` or `Moisturizer` inconsistently). A retrieved product
counts as relevant if its category matches and its `skin_type` tags
intersect the expected ones. This is a heuristic proxy for the spec's
"manual spot-check", not a substitute for a human judging result quality —
**current result: 66.0% precision@5** across 20 queries. Full per-query
breakdown is printed by the script and written to
`backend/tests/eval/results/precision_eval.json`.

## Project layout

```
backend/
  app/
    api/v1/endpoints/   health.py, recommendations.py, products.py
                        (thin routers -- delegate to services/, no direct
                        db/vectorstore imports)
    services/           recommendation.py (ranking), explanation.py (Gemini),
                         product_service.py, health_service.py
    db/                 session.py, repository.py, ingest.py
    vectorstore/        faiss_index.py (build/search/persist)
    core/               config.py (env vars), logging.py, middleware.py,
                         error_handlers.py
    main.py
  ingestion/            build_embeddings.py, build_index.py (offline pipeline)
  tests/                pytest suite + tests/eval/ (manual/QA scripts)
  requirements.txt      production deps
  requirements-dev.txt  + pytest/httpx, for running the test suite
  Dockerfile
frontend/
  src/App.jsx           the whole SPA (search form, results, explanations)
  Dockerfile            multi-stage: node build -> nginx (non-root, :8080)
data/                   SQLite DB + FAISS index (bind-mounted into Docker,
                         not baked into the image)
docker-compose.yml
.env.example
PROJECT_SPEC.md
```

## Configuration

All variables are documented with placeholder values in
[.env.example](.env.example); copy it to `.env` and fill in a real
`GEMINI_API_KEY`. Only `GEMINI_API_KEY` is required — everything else has a
sane default. If it's missing, the backend fails fast at startup with a
single clear line to stderr rather than a stack trace or a silent partial
start.

## Known limitations

- Endpoint is `POST /recommend`, not `POST /search` as an earlier draft of
  the spec named it — `PROJECT_SPEC.md` §11 has since been corrected to match.
- No live GitHub Actions / CI pipeline file is included; "CI-ready" here
  means the test suite runs cleanly offline with a standard `pytest`
  invocation, not that a pipeline is wired up.
- Precision@5 is a structural heuristic (category + skin type), not human
  relevance judgment — see the Evaluation section above.
