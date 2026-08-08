CREATE EXTENSION IF NOT EXISTS vector;

-- Raw normalized weather documents (mirror of ticker_news_documents).
CREATE TABLE IF NOT EXISTS weather_documents (
    id             TEXT PRIMARY KEY,          -- stable dedup key (alert id / forecast hash)
    location       TEXT NOT NULL,             -- "City, ST" or areaDesc
    source_type    TEXT NOT NULL,             -- 'alert' or 'forecast'
    headline       TEXT,                      -- event name, e.g. "Flash Flood Warning"
    narrative_text TEXT NOT NULL,             -- free-text body that gets embedded
    issued_at      TIMESTAMPTZ,               -- effective / start time
    payload        JSONB NOT NULL,            -- raw JSON for provenance
    synced_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_weather_documents_source_type
    ON weather_documents (source_type);

-- Embeddings, one row per chunk (mirror of ticker_news_embeddings).
CREATE TABLE IF NOT EXISTS weather_embeddings (
    id          TEXT PRIMARY KEY,             -- "<document_id>:<chunk_index>"
    document_id TEXT NOT NULL REFERENCES weather_documents(id) ON DELETE CASCADE,
    chunk_index INT  NOT NULL,
    chunk_text  TEXT NOT NULL,
    embedding   vector(384) NOT NULL,         -- all-MiniLM-L6-v2 is 384-dim
    model_name  TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Cosine-distance index for retrieval (the <=> operator in /weather/search).
CREATE INDEX IF NOT EXISTS idx_weather_embeddings_hnsw
    ON weather_embeddings USING hnsw (embedding vector_cosine_ops);
