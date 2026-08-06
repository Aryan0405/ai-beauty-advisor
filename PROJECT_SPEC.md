# PROJECT_SPEC.md — AI Beauty Advisor

**Version:** 1.0
**Status:** Draft — Source of Truth for Implementation
**Owner:** Engineering

---

## 1. Project Overview

AI Beauty Advisor is a semantic search-based skincare/beauty product recommendation system. Users describe their skin concerns, preferences, or goals in natural language; the system retrieves the most relevant products from the Kaggle Cosmetics Dataset using vector similarity search, then generates a natural-language explanation of why each product was recommended using an LLM (Gemini 2.5 Flash).

The system consists of a Python/FastAPI backend performing embedding-based retrieval and LLM-backed explanation generation, and a React + Vite frontend for user interaction. Data is stored in SQLite; vector search is handled by FAISS.

## 2. Objectives

- O1: Enable users to search skincare products using free-text natural language queries.
- O2: Return semantically relevant product matches ranked by similarity.
- O3: Generate concise, grounded LLM explanations for each recommended product.
- O4: Provide a clean, modular, testable backend architecture suitable for future extension.
- O5: Ship a working, containerized, end-to-end demo (frontend + backend + data).

## 3. Scope

### In Scope
- One-time ingestion and preprocessing of the Kaggle Cosmetics Dataset.
- Embedding generation for product metadata (name, brand, ingredients, description, category, skin type).
- FAISS-based vector index for similarity search.
- FastAPI backend exposing search/recommendation endpoints.
- Gemini 2.5 Flash integration for explanation generation, grounded in retrieved product data only.
- React + Vite single-page frontend: search input, results list, explanation display.
- SQLite as the system of record for product metadata.
- Dockerized deployment (backend + frontend, single `docker-compose`).

### Out of Scope
- User accounts, authentication, personalization history, or profiles.
- Real-time price/inventory syncing with external retailers.
- Image-based skin analysis or computer vision features.
- Multi-language support.
- Mobile app.
- A/B testing infrastructure or analytics dashboards.

## 4. Functional Requirements

| ID | Requirement |
|----|-------------|
| FR1 | User submits a free-text query describing skin type, concern, or desired product attributes. |
| FR2 | System returns top-K (default K=5, configurable) semantically ranked products. |
| FR3 | Each result includes product metadata: name, brand, category, ingredients, price (if available), rank/similarity score. |
| FR4 | System generates one LLM explanation per result, grounded strictly in that product's retrieved metadata. |
| FR5 | System supports optional filters: category and skin type (applied pre- or post-retrieval). |
| FR6 | System exposes a health-check endpoint for deployment monitoring. |
| FR7 | Frontend displays loading, empty, and error states distinctly. |
| FR8 | Ingestion pipeline is re-runnable (idempotent) to rebuild the index from the raw dataset. |

## 5. Non-Functional Requirements

| Category | Requirement |
|----------|-------------|
| Performance | P95 search+explanation latency ≤ 3.5s for K=5 under local/dev load. |
| Performance | Vector search alone (excluding LLM call) ≤ 150ms for K=5. |
| Scalability | Architecture supports swapping FAISS for a hosted vector DB without touching API contracts (interface-based design). |
| Maintainability | Backend organized by layer (API, service, retrieval, LLM, data access); no cross-layer leakage. |
| Reliability | LLM failures degrade gracefully — system still returns ranked results without explanations. |
| Portability | Entire system runs via `docker-compose up` with no manual setup steps beyond `.env` configuration. |
| Observability | All requests, retrieval latency, and LLM call outcomes are logged with correlation IDs. |
| Cost Control | LLM explanation calls are batched/limited to top-K only; no speculative or background LLM calls. |
| Testability | Core retrieval and API logic covered by unit/integration tests independent of live LLM calls (mockable). |

## 6. System Architecture

```
                     ┌─────────────────────┐
                     │   React + Vite SPA   │
                     │  (Search UI, Results)│
                     └──────────┬───────────┘
                                │ HTTPS/JSON (REST)
                     ┌──────────▼───────────┐
                     │     FastAPI App       │
                     │  (API Layer)          │
                     └──────────┬───────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
┌───────▼────────┐   ┌──────────▼──────────┐   ┌─────────▼─────────┐
│ Retrieval       │   │ Explanation Service  │   │ Product Service    │
│ Service         │   │ (Gemini 2.5 Flash)   │   │ (metadata lookups)  │
│ (FAISS + ST)    │   └──────────────────────┘   └─────────┬─────────┘
└───────┬─────────┘                                        │
        │                                          ┌────────▼────────┐
┌───────▼─────────┐                                │     SQLite       │
│ FAISS Index      │                                │  (product data)  │
│ (in-memory/disk) │                                └───────────────────┘
└──────────────────┘

Offline / one-time:
  Kaggle CSV → Ingestion Pipeline → SQLite + Embeddings → FAISS Index (persisted to disk)
```

**Key architectural decisions:**
- Ingestion (data prep, embedding generation, index build) is a separate offline pipeline, not part of the live request path.
- The FAISS index and SQLite DB are both build artifacts checked into a `data/` volume, loaded at API startup.
- Retrieval and explanation are decoupled services; explanation generation failure never blocks retrieval results.

## 7. Folder Structure

```
ai-beauty-advisor/
├── backend/
│   ├── app/
│   │   ├── api/                # FastAPI routers (HTTP layer only)
│   │   │   └── v1/
│   │   │       ├── search.py
│   │   │       └── health.py
│   │   ├── services/            # Business logic
│   │   │   ├── retrieval_service.py
│   │   │   ├── explanation_service.py
│   │   │   └── product_service.py
│   │   ├── core/                # Config, logging, startup
│   │   │   ├── config.py
│   │   │   └── logging.py
│   │   ├── models/               # Pydantic schemas
│   │   │   └── schemas.py
│   │   ├── db/                   # SQLite access layer
│   │   │   ├── database.py
│   │   │   └── repository.py
│   │   ├── llm/                  # Gemini client wrapper
│   │   │   └── gemini_client.py
│   │   ├── vectorstore/          # FAISS wrapper
│   │   │   └── faiss_index.py
│   │   └── main.py
│   ├── ingestion/                # Offline data pipeline
│   │   ├── preprocess.py
│   │   ├── build_embeddings.py
│   │   └── build_index.py
│   ├── tests/
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── api/                  # API client
│   │   └── App.tsx
│   ├── Dockerfile
│   └── package.json
├── data/
│   ├── raw/                      # Original Kaggle CSV
│   ├── processed/                # Cleaned dataset
│   └── index/                    # Persisted FAISS index + id mapping
├── docker-compose.yml
├── .env.example
└── PROJECT_SPEC.md
```

## 8. Backend Architecture

- **API Layer** (`api/`): Thin FastAPI routers. Validates input via Pydantic, delegates to services, formats responses. No business logic.
- **Service Layer** (`services/`): Orchestrates retrieval → filtering → explanation generation. Contains all business rules.
- **Data Access Layer** (`db/`): SQLite repository pattern; only place raw SQL/ORM calls live.
- **Vector Store Layer** (`vectorstore/`): Wraps FAISS index load/query behind an interface (`search(query_embedding, k) -> List[ProductId, score]`), enabling future backend swap.
- **LLM Layer** (`llm/`): Wraps Gemini 2.5 Flash API calls, prompt construction, timeout/retry handling, and response parsing. Isolated so it can be mocked in tests.
- **Core** (`core/`): Centralized configuration (pydantic-settings) and structured logging setup, loaded once at startup.

Dependency direction is strictly one-way: `api → services → (db | vectorstore | llm)`. No layer below `services` imports from `api`.

## 9. Frontend Architecture

- Single-page React + Vite app, functional components with hooks only.
- **Pages:** `SearchPage` (default/only route for v1).
- **Components:** `SearchBar`, `ResultCard`, `ExplanationText`, `FilterPanel`, `LoadingState`, `ErrorState`.
- **API client:** thin fetch wrapper (`src/api/`) encapsulating the single search endpoint; no business logic in components.
- State managed with local component state / `useState`; no global state library needed given single-page scope.
- Styling: any lightweight utility framework (e.g., Tailwind) — implementation detail left to build phase, not specified further here.

## 10. Recommendation Pipeline

**Offline (ingestion, run once or on data refresh):**
1. Load raw Kaggle CSV.
2. Clean/normalize fields (drop nulls in required fields, normalize text casing, dedupe products).
3. Construct a composite text representation per product (e.g., name + brand + category + ingredients + description).
4. Generate embeddings for each product's composite text using Sentence Transformers.
5. Build and persist a FAISS index (vectors + product-id mapping) to `data/index/`.
6. Load cleaned product metadata into SQLite.

**Online (per request):**
1. Receive user query text (+ optional filters).
2. Embed the query using the same Sentence Transformers model used at ingestion.
3. Query FAISS index for top-K nearest neighbors (over-fetch if filters applied, then filter).
4. Fetch full product metadata for returned IDs from SQLite.
5. For each of the top-K products, construct a prompt containing only that product's metadata and the user's query; call Gemini 2.5 Flash for a short grounded explanation.
6. Assemble and return ranked results with explanations (explanation optional per-item on LLM failure).

## 11. API Specification

Base path: `/api/v1`

### `POST /search`
**Request:**
```json
{
  "query": "lightweight moisturizer for oily acne-prone skin",
  "top_k": 5,
  "filters": {
    "category": "Moisturizer",
    "skin_type": "Oily"
  }
}
```
**Response (200):**
```json
{
  "results": [
    {
      "product_id": "string",
      "name": "string",
      "brand": "string",
      "category": "string",
      "price": 24.99,
      "score": 0.87,
      "explanation": "string | null"
    }
  ],
  "query": "string",
  "count": 5
}
```
**Errors:** `400` invalid input; `503` retrieval subsystem unavailable; `500` unhandled.

### `GET /health`
Returns `{ "status": "ok", "index_loaded": true, "db_connected": true }`.

## 12. Data Flow

```
Kaggle CSV → Preprocess → SQLite (metadata) + FAISS Index (vectors)
                                        │
User Query → Embed → FAISS search → Product IDs
                                        │
                            SQLite lookup (metadata)
                                        │
                        Gemini prompt per product → Explanation
                                        │
                              JSON response → Frontend render
```

## 13. Database Design

**SQLite — single table for v1, normalization deferred as premature for this scope.**

`products`

| Column | Type | Notes |
|--------|------|-------|
| product_id | TEXT (PK) | Stable ID derived from source dataset |
| name | TEXT | Not null |
| brand | TEXT | Not null |
| category | TEXT | Indexed |
| skin_type | TEXT | Indexed; comma-separated or normalized tag |
| ingredients | TEXT | Raw text |
| description | TEXT | Composite source text used for embedding |
| price | REAL | Nullable |
| embedding_id | INTEGER | Maps to FAISS index position |

Indexes: `category`, `skin_type` for filter performance. FAISS index and SQLite `embedding_id` must stay in sync — rebuilt together by the ingestion pipeline as a single atomic step.

## 14. Configuration & Environment Variables

| Variable | Description | Required |
|----------|--------------|----------|
| `GEMINI_API_KEY` | API key for Gemini 2.5 Flash | Yes |
| `GEMINI_MODEL` | Model identifier (default: `gemini-2.5-flash`) | No |
| `EMBEDDING_MODEL_NAME` | Sentence Transformers model name | Yes |
| `SQLITE_DB_PATH` | Path to SQLite file | Yes |
| `FAISS_INDEX_PATH` | Path to persisted FAISS index | Yes |
| `DEFAULT_TOP_K` | Default number of results | No (default 5) |
| `LLM_TIMEOUT_SECONDS` | Timeout for explanation calls | No (default 5) |
| `LOG_LEVEL` | Logging verbosity | No (default INFO) |
| `CORS_ALLOWED_ORIGINS` | Allowed frontend origins | Yes |
| `VITE_API_BASE_URL` | Backend URL for frontend build | Yes |

All secrets provided via `.env`, never committed. `.env.example` checked in with placeholder values.

## 15. Error Handling Strategy

- API layer returns structured error responses: `{ "error": { "code": "string", "message": "string" } }`.
- Retrieval failures (index not loaded, corrupt index) → `503`, logged as critical at startup health check.
- LLM explanation failures (timeout, rate limit, malformed response) are caught per-item; that item's `explanation` is set to `null` and the request still returns `200` with results — never fail the whole request due to LLM issues.
- Input validation errors (empty query, invalid `top_k`) → `400` with field-level detail via Pydantic.
- Unhandled exceptions → generic `500`, full stack trace logged server-side only, generic message returned to client.

## 16. Logging Strategy

- Structured JSON logging (one log line per event) via Python `logging` + a JSON formatter.
- Every incoming request assigned a correlation/request ID, propagated through service and LLM calls, included in every log line for that request.
- Logged events: request received, retrieval latency, FAISS result count, LLM call start/end/failure, response sent, total latency.
- No PII is present in this system (no user accounts); query text may be logged at INFO level.
- LLM prompts/responses logged at DEBUG only, disabled by default in production config.

## 17. Security Considerations

- API keys (`GEMINI_API_KEY`) loaded only from environment, never logged, never returned in responses.
- CORS restricted to explicitly configured frontend origin(s); wildcard not used in production config.
- Input length limits enforced on `query` field to prevent prompt-injection-via-length or excessive token cost.
- Basic prompt-injection mitigation: LLM prompt template treats product metadata and user query as clearly delimited, untrusted data blocks; system instructs the model to only describe the given product, not follow embedded instructions in the query text.
- Rate limiting on `/search` (e.g., per-IP token bucket) recommended at reverse-proxy or middleware level to control LLM cost exposure.
- No authentication in v1 scope (explicitly out of scope) — this is a demo-grade constraint and should be called out clearly if ever deployed publicly.
- Docker images run as non-root user; no secrets baked into images.

## 18. Testing Strategy

| Layer | Approach |
|-------|----------|
| Unit — Services | Mock FAISS/DB/LLM clients; test retrieval ranking logic, filter logic, explanation-failure fallback behavior. |
| Unit — LLM wrapper | Test prompt construction and response parsing with mocked API responses (success, timeout, malformed). |
| Integration — API | `TestClient` against real SQLite (test fixture DB) + real small FAISS index (test fixture), mocked Gemini client. |
| Integration — Ingestion | Run pipeline against a small sample CSV; assert row counts, index size, and SQLite/FAISS ID alignment. |
| Frontend | Component tests for `ResultCard`, `SearchBar` states (loading/empty/error); one end-to-end smoke test hitting a mocked API. |
| Manual/QA | Full docker-compose run with real dataset and real Gemini key before each release; verify P95 latency target. |

CI gate: unit + integration tests must pass; live LLM/network calls excluded from CI (mocked only).

## 19. Evaluation Metrics

- **Retrieval relevance:** manual spot-check precision@5 against a curated set of ~20 representative queries (documented in `tests/eval/`).
- **Latency:** P50/P95 for retrieval-only and full request (retrieval + explanation), tracked via logs.
- **Explanation groundedness:** manual review checklist — explanation must not reference attributes absent from the retrieved product's metadata.
- **Availability of explanations:** % of requests where all K explanations succeeded vs. degraded (LLM fallback triggered).
- **System uptime (health check):** trivial pass/fail via `/health`.

## 20. Project Milestones (Implementation Order)

1. **M1 — Data pipeline foundation:** Acquire dataset, build preprocessing + cleaning script, load into SQLite.
2. **M2 — Embedding & index build:** Generate embeddings, build FAISS index, persist to disk; validate with a manual query script.
3. **M3 — Backend core:** Implement `db`, `vectorstore`, `services` layers with unit tests; no API yet.
4. **M4 — API layer:** Implement `/search` and `/health` FastAPI endpoints wired to services; integration tests.
5. **M5 — LLM integration:** Implement Gemini client wrapper, prompt template, per-item failure handling.
6. **M6 — Frontend:** Build SearchPage + components, wire to API, handle loading/empty/error states.
7. **M7 — Dockerization:** Backend + frontend Dockerfiles, `docker-compose.yml`, `.env.example`, end-to-end local run.
8. **M8 — Testing & evaluation:** Complete test suite, run relevance/latency evaluation, fix gaps.
9. **M9 — Polish & submission:** Logging cleanup, README, final QA pass against checklist (Section 22).

## 21. Future Improvements

- Swap FAISS for a managed/hosted vector DB (e.g., pgvector, Pinecone) for scale beyond single-node.
- Add caching layer for repeated/similar queries to reduce LLM cost and latency.
- Add lightweight user feedback capture (thumbs up/down) to inform future ranking tuning.
- Support hybrid search (keyword + semantic) for exact brand/ingredient matches.
- Add authentication and per-user history if the product moves beyond demo scope.
- Add automated relevance evaluation (e.g., embedding-based ground-truth set) beyond manual spot-checks.

## 22. Submission Checklist

- [ ] Dataset ingested and cleaned; row counts documented.
- [ ] FAISS index built and persisted; SQLite/FAISS IDs verified in sync.
- [ ] `/search` and `/health` endpoints implemented and tested.
- [ ] LLM explanation generation working with graceful degradation on failure.
- [ ] Frontend implements search, results, loading/empty/error states.
- [ ] All environment variables documented in `.env.example`.
- [ ] Unit + integration tests passing in CI.
- [ ] `docker-compose up` runs the full system with no manual steps beyond `.env` setup.
- [ ] Logging in place with request correlation IDs.
- [ ] Evaluation results (relevance + latency) documented.
- [ ] README with setup, run, and architecture summary.
- [ ] No secrets committed to the repository.
