# Postgres Articles Embedding Utility (`pg_articles_util.py`)

Succinct guide for importing article data, generating embeddings with Ollama, and running similarity search using PostgreSQL + pgvector.

## Prerequisites

- PostgreSQL 16+ running locally (default host 127.0.0.1, port 5432)
- `pgvector` extension installed and available (`CREATE EXTENSION vector;`)
- Python packages: `psycopg2-binary`, `ollama`, `python-dotenv` (optional)
- Ollama running locally with model `all-minilm:l12-v2` (auto-pulled if missing)

## Environment Variables (optional overrides)

```
PG_HOST=127.0.0.1
PG_PORT=5432
PG_DB=precise_articles
PG_USER=postgres
PG_PASSWORD= (blank by default)
EMBED_MODEL=all-minilm:l12-v2
ARTICLES_CSV=data/InputData_IndustryEconomics_Jul24-Jun25.csv  # used in no-arg mode
AUTO_EMBED_LIMIT=500  # optional cap for one-shot embedding phase
```

You can place these in a `.env` file.

## Table Schema (created automatically)

`articles` with columns:

- ArticleID (PK)
- ArtDate, Month, Year, CompetName, Issue, Industry, Comments, SpokespersonName
- Impact metrics (KPMG/Deloitte/EY/PwC)
- Comments_embedding vector(384)
- Timestamps

## Core Workflow

### One-Shot (no arguments)

Run everything (ensure DB + extension + table, import CSV, generate embeddings for all or limited rows, show status):

```
python pg_articles_util.py
```

Configure with `ARTICLES_CSV` and optionally `AUTO_EMBED_LIMIT`.

### Manual Steps

1. Create database + extension + table:

```
python pg_articles_util.py create-db
```

2. Import CSV (raw text only first):

```
python pg_articles_util.py import data/InputData_IndustryEconomics_Jul24-Jun25.csv
```

3. Generate embeddings (batching):

```
python pg_articles_util.py embed --batch-size 50 --limit 500   # limit optional
```

4. Check status:

```
python pg_articles_util.py status
```

5. Similarity search:

```
python pg_articles_util.py similar "KPMG AI transformation strategy" --k 5
```

## Command Summary

| Command         | Purpose                                   |
| --------------- | ----------------------------------------- |
| create-db       | Ensure DB, pgvector extension, and table  |
| create-table    | Ensure extension + table only             |
| import <csv>    | Upsert article rows from CSV              |
| embed           | Generate embeddings for rows missing them |
| status          | Show total / embedded / pending counts    |
| similar <query> | Embed query + return nearest articles     |

## Notes

- Re-running `import` will upsert (existing rows updated).
- Embeddings only generated for rows with non-empty `Comments` and null `Comments_embedding`.
- Distance metric uses `<->` (lower = closer).
- If dimensions differ, vectors are truncated/padded to 384.

## Minimal Dependency Install

```
pip install psycopg2-binary ollama python-dotenv
```

(Ensure Ollama service is running.)

## Optional: Performance Index (after enough data)

To speed large similarity queries you can add (HNSW example):

```sql
CREATE INDEX IF NOT EXISTS idx_articles_embedding_hnsw
ON articles USING hnsw (Comments_embedding vector_cosine_ops) WITH (m=16, ef_construction=64);
```

(Then raise `SET enable_seqscan = OFF;` during tuning tests.)

## Troubleshooting

| Symptom                                       | Fix                                                     |
| --------------------------------------------- | ------------------------------------------------------- |
| `ERROR:  extension "vector" is not installed` | Install/build pgvector, then `CREATE EXTENSION vector;` |
| Empty similarity results                      | Ensure embeddings generated (`embed` + `status`)        |
| Model pull fails                              | Manually run `ollama pull all-minilm:l12-v2`            |
| Connection refused                            | Confirm PostgreSQL running and env vars correct         |

---

Concise reference complete. For deeper customization, inspect the script source.
