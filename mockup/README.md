# Summary Pipeline (Isolated)

This folder contains an isolated LangGraph-based summarisation workflow that does **not** modify existing project files.

## Components

- `pg_ingest_node.py`: Ingest CSV into a specified test table (default `articles_test`).
- `pg_aggregate_node.py`: Runs aggregation metrics on the chosen table.
- `pg_summary_agent.py`: Few-shot LLM summariser.
- `pg_graph.py`: Orchestrates ingest -> aggregate -> summarise.
- `data/synthetic_articles.csv`: Synthetic sample dataset.

## Table

If `--table articles_test` (default), a table clone schema is created automatically.
To use the production table, pass `--table articles` (will reuse existing schema logic).

## Run (Synthetic Data)

```bash
python summary_pipeline/pg_graph.py --csv summary_pipeline/data/synthetic_articles.csv --table articles_test
```

## Output

Prints JSON import stats, aggregation metrics, and generated summary.

## Environment Variables

Reuse the same Postgres env vars:

- PG_HOST, PG_PORT, PG_DB, PG_USER, PG_PASSWORD
- LLM_MODEL (default `llama3.1`)

## Extend Metrics

Edit `BASE_QUERIES` in `pg_aggregate_node.py` to add more derived stats.
