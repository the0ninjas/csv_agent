# CSV / Postgres AI Agent

Minimal guide to ingest article CSVs, create embeddings, and query / summarise with an LLM.

## 1. Environment

Install deps (example):

```
pip install psycopg2-binary ollama python-dotenv langchain langgraph langchain-ollama pgvector sqlalchemy
```

Run Ollama locally and pull required models (examples):

```
ollama pull all-minilm:l12-v2
ollama pull llama3.1
```

Optional `.env` variables:

```
PG_HOST=127.0.0.1
PG_PORT=5432
PG_DB=precise_articles
PG_USER=postgres
PG_PASSWORD=
LLM_MODEL=llama3.1
EMBED_MODEL=all-minilm:l12-v2
ARTICLES_CSV=data/InputData_IndustryEconomics_Jul24-Jun25.csv
AUTO_EMBED_LIMIT=500
```

## 2. Ingest + Embed Workflow

One-shot (creates DB objects, imports CSV, embeds):

```
python pg_articles_util.py
```

Manual steps:

```
python pg_articles_util.py create-db
python pg_articles_util.py import data/InputData_IndustryEconomics_Jul24-Jun25.csv
python pg_articles_util.py embed --batch-size 50 --limit 300
python pg_articles_util.py status
python pg_articles_util.py similar "KPMG housing affordability outlook" --k 5
```

## 3. SQL Question Answering Agent

Ask natural language questions over `articles` table:

```
python examples/pg_sql_agent.py "How many articles per year?"
```

(Uses ReAct agent + ChatOllama.)

## 4. CSV -> JSON Serialiser

Convert a simplified CSV to JSON or NDJSON:

```
python data/serialiser.py data/simplified/KPMG_July2024.csv -o data/simplified/KPMG_July2024.json
python data/serialiser.py data/simplified/KPMG_July2024.csv -o kpmg.ndjson --ndjson
```

Outputs only columns present in CSV.

## 5. Summariser (Few-Shot on Raw Articles)

Provide a list of article dicts:

```
from pg_summary_agent import Summariser
from prompt_example import EXAMPLES
summ = Summariser()
print(summ.run(EXAMPLES[0]["articles"]))
```

## 6. Troubleshooting

| Issue                    | Fix                                         |
| ------------------------ | ------------------------------------------- |
| pgvector extension error | Ensure `CREATE EXTENSION vector;` in DB     |
| Connection refused       | Verify PostgreSQL running, env vars correct |
| No embeddings returned   | Run `embed` step, check `status`            |
| Model pull failure       | `ollama pull <model>` manually              |

## 7. Next Ideas

- Add retrieval-based dynamic few-shot selection
- Map-reduce summarisation for large periods
- HNSW index for `Comments_embedding`

---

Concise reference. See `PG_ARTICLES_USAGE.md` for deeper notes.
