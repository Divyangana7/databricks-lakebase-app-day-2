# Weather Intelligence — Unstructured Data → Lakebase Vector Search → REST API

This module extends the `databricks-lakebase-app-day-2` app with a weather
pipeline: harvest free-text weather data, embed it, store it in Lakebase
(Postgres + pgvector), and retrieve it by semantic similarity through the Flask
API.

## Data source and why

**National Weather Service API (`api.weather.gov`).** It is free, needs no API
key, and returns rich narrative text that is well suited to embedding:

- `GET /alerts/active?area={ST}` — active alerts with free-text `description`
  and `instruction` fields.
- `GET /gridpoints/.../forecast` (reached via the `forecast` URL from
  `GET /points/{lat},{lon}`) — multi-period forecasts, each with a
  `detailedForecast` narrative.

No API key means the work stays focused on harvesting, vectorization, and
retrieval rather than auth plumbing. The one requirement is a descriptive
`User-Agent` header on every request; without it NWS returns HTTP 403. This is
set via the `WEATHER_USER_AGENT` environment variable, so no contact email is
hard-coded.

Locations are accepted either as `"lat,lon"` (used directly) or as `"City, ST"`.
City/state is resolved to coordinates with a small built-in table for common
cities, falling back to the free OpenStreetMap Nominatim geocoder.

## Schema decisions

Two tables mirror the `ticker_news_documents` / `ticker_news_embeddings`
pattern.

`weather_documents` (raw, one row per alert or forecast period):

| column | type | notes |
| --- | --- | --- |
| `id` | TEXT PK | alert id, or `forecast:<sha1>` for forecasts (stable dedup key) |
| `location` | TEXT | `City, ST` or `areaDesc` |
| `source_type` | TEXT | `alert` or `forecast` |
| `headline` | TEXT | event name, e.g. `Flash Flood Warning` |
| `narrative_text` | TEXT | the free text that gets embedded |
| `issued_at` | TIMESTAMPTZ | effective / start time |
| `payload` | JSONB | raw JSON for provenance |
| `synced_at` | TIMESTAMPTZ | `DEFAULT now()` |

`weather_embeddings` (one row per chunk):

| column | type | notes |
| --- | --- | --- |
| `id` | TEXT PK | `<document_id>:<chunk_index>` |
| `document_id` | TEXT FK | references `weather_documents(id)` |
| `chunk_index` | INT | |
| `chunk_text` | TEXT | |
| `embedding` | `vector(384)` | pgvector column |
| `model_name` | TEXT | |
| `created_at` | TIMESTAMPTZ | `DEFAULT now()` |

- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2` (384 dims), the
  same model as the news pipeline, so both stay queryable with the same
  distance conventions.
- **Chunking:** sliding window, `CHUNK_SIZE=800`, `CHUNK_OVERLAP=100` (character
  based), matching the news pipeline. Most NWS text fits in a single chunk;
  chunking only matters for long combined alert + instruction bodies.
- **Index:** `CREATE INDEX ... USING hnsw (embedding vector_cosine_ops)` for
  cosine retrieval via the `<=>` operator.

## Files

- `weather_client.py` — NWS client (mirrors `massive_client.py`).
- `app.py` — adds `POST /weather/sync` and `POST /weather/search` (see
  `app_weather_additions.py` for the exact blocks).
- `sql/weather_tables.sql` — DDL for both tables (the app also creates them
  automatically).
- `notebooks/ingest_weather_embeddings.py` — psycopg2 embedding job.
- `requirements.txt` — add `sentence-transformers`.

## Run the pipeline end to end

1. Install deps: `pip install -r requirements.txt`.
2. Sync documents:
   ```
   curl -X POST "$APP_URL/weather/sync" \
     -H "Content-Type: application/json" \
     -d '{"locations": ["Chicago, IL", "Austin, TX"], "limit": 50}'
   ```
3. Embed: run `notebooks/ingest_weather_embeddings.py` in Databricks (Run All)
   or as a script wherever `lakebase.get_connection()` works.
4. Search:
   ```
   curl -X POST "$APP_URL/weather/search" \
     -H "Content-Type: application/json" \
     -d '{"query": "flash flood risk this weekend", "top_k": 5}'
   ```

## Known limitations / future improvements

- Alerts are fetched per state, so two cities in one state share the same alert
  set (deduplicated on `id`).
- Geocoding beyond the built-in city list depends on Nominatim availability and
  its 1 request/second policy.
- Re-running `/weather/sync` upserts on `id`, so it will not create duplicates.
- Possible extensions: an LLM summary of the top results (basic RAG), a
  `source_type` filter on retrieval, a scheduled re-sync job, and an
  HNSW-vs-no-index latency benchmark.
