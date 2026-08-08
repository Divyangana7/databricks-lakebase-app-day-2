# Databricks notebook source
"""
Ingest weather embeddings.

Self-contained ETL that mirrors notebooks/ingest_ticker_news_embeddings.py, but
written as a plain psycopg2 script (no spark.write.jdbc, which is not reliable
against this Lakebase instance):

  1. Ensure the weather_embeddings table + HNSW index exist.
  2. Read documents from weather_documents that have no embeddings yet.
  3. Chunk narrative_text (sliding window, char-based, CHUNK_SIZE / CHUNK_OVERLAP).
  4. Embed each chunk with sentence-transformers/all-MiniLM-L6-v2 (384-dim).
  5. Upsert rows into weather_embeddings via execute_values, casting the vector
     to %s::vector.

Run it as a Databricks notebook (Run All) attached to a cluster, or as a plain
script anywhere lakebase.get_connection() works.

In a Databricks notebook, install deps first in a %pip cell:
    %pip install sentence-transformers psycopg2-binary
    dbutils.library.restartPython()
"""

import os
import sys

from psycopg2.extras import execute_values

# When run from a notebook inside the repo's Git folder, make sure the repo root
# (where lakebase.py lives) is importable.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import lakebase  # noqa: E402

# ---- configuration --------------------------------------------------------

DOCS_TABLE = os.environ.get("WEATHER_DOCS_TABLE", "weather_documents")
EMB_TABLE = os.environ.get("WEATHER_EMB_TABLE", "weather_embeddings")
MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_DIM = 384                     # all-MiniLM-L6-v2 output dimensionality
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 100))
BATCH_ROWS = 200                   # rows per execute_values call


def ensure_weather_embeddings_table() -> None:
    """Create the embeddings table (vector(384)) and its HNSW index if missing."""
    lakebase.run_write("CREATE EXTENSION IF NOT EXISTS vector")
    lakebase.run_write(
        f"""
        CREATE TABLE IF NOT EXISTS {EMB_TABLE} (
            id          TEXT PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES {DOCS_TABLE}(id) ON DELETE CASCADE,
            chunk_index INT  NOT NULL,
            chunk_text  TEXT NOT NULL,
            embedding   vector({EMBED_DIM}) NOT NULL,
            model_name  TEXT NOT NULL,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    lakebase.run_write(
        f"CREATE INDEX IF NOT EXISTS idx_{EMB_TABLE}_hnsw "
        f"ON {EMB_TABLE} USING hnsw (embedding vector_cosine_ops)"
    )


def fetch_unembedded_documents() -> list[dict]:
    """Return documents that do not yet have any rows in the embeddings table."""
    return lakebase.run_query(
        f"""
        SELECT d.id, d.narrative_text
        FROM {DOCS_TABLE} d
        WHERE NOT EXISTS (
            SELECT 1 FROM {EMB_TABLE} e WHERE e.document_id = d.id
        )
        ORDER BY d.synced_at
        """
    )


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Sliding-window character chunking. Most NWS text is short (one chunk)."""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    step = max(1, size - overlap)
    return [text[i : i + size] for i in range(0, len(text), step) if text[i : i + size].strip()]


def _vec_to_str(vec) -> str:
    """pgvector text literal, e.g. '[0.1,0.2,...]'."""
    return "[" + ",".join(str(float(x)) for x in vec) + "]"


def main() -> None:
    ensure_weather_embeddings_table()
    docs = fetch_unembedded_documents()
    print(f"Documents to embed: {len(docs)}")
    if not docs:
        print("Nothing to embed. Run /weather/sync first.")
        return

    # Load the model once for the whole job.
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME)

    rows = []
    for doc in docs:
        chunks = chunk_text(doc["narrative_text"])
        for idx, chunk in enumerate(chunks):
            embedding = model.encode(chunk).tolist()
            rows.append(
                (
                    f"{doc['id']}:{idx}",   # id
                    doc["id"],              # document_id
                    idx,                    # chunk_index
                    chunk,                  # chunk_text
                    _vec_to_str(embedding), # embedding (cast to ::vector below)
                    MODEL_NAME,             # model_name
                )
            )

    print(f"Chunks to write: {len(rows)}")

    written = 0
    with lakebase.get_connection() as conn:
        with conn.cursor() as cur:
            for start in range(0, len(rows), BATCH_ROWS):
                batch = rows[start : start + BATCH_ROWS]
                execute_values(
                    cur,
                    f"""
                    INSERT INTO {EMB_TABLE}
                        (id, document_id, chunk_index, chunk_text, embedding, model_name, created_at)
                    VALUES %s
                    ON CONFLICT (id) DO UPDATE SET
                        chunk_text = EXCLUDED.chunk_text,
                        embedding  = EXCLUDED.embedding,
                        model_name = EXCLUDED.model_name,
                        created_at = now()
                    """,
                    batch,
                    template="(%s, %s, %s, %s, %s::vector, %s, now())",
                )
                written += len(batch)
        conn.commit()

    print(f"Wrote {written} embedding rows into {EMB_TABLE}.")


if __name__ == "__main__":
    main()
