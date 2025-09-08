"""PostgreSQL article ingestion & embedding utility (pgvector).

Features (parity-inspired from mysql `sql_util.py`):
- Ensure database & pgvector extension
- Create `articles` table with vector(384) column for comment embeddings
- Import CSV rows (two-step: raw text then embeddings)
- Generate embeddings via Ollama (model: all-minilm:l12-v2) and store
- Similarity search by query text
- Embedding status summary
- Simple CLI entry points

Environment variables (override defaults):
  PG_HOST (default 127.0.0.1)
  PG_PORT (default 5432)
  PG_DB   (default precise_articles)
  PG_USER (default postgres)
  PG_PASSWORD (default empty)
  EMBED_MODEL (default all-minilm:l12-v2)

Usage examples:
  python pg_articles_util.py create-db
  python pg_articles_util.py import data/InputData_IndustryEconomics_Jul24-Jun25.csv
  python pg_articles_util.py embed --batch-size 25 --limit 200
  python pg_articles_util.py status
  python pg_articles_util.py similar "KPMG AI transformation strategy"
"""
from __future__ import annotations
import os
import sys
import csv
import argparse
from datetime import datetime
from typing import List, Optional, Tuple

import psycopg2
import psycopg2.extras

# Optional: load .env if present
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# Configuration
PG_HOST = os.getenv("PG_HOST", "127.0.0.1")
PG_PORT = int(os.getenv("PG_PORT", "5432"))
PG_DB = os.getenv("PG_DB", "precise_articles")
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASSWORD = os.getenv("PG_PASSWORD", "")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-minilm:l12-v2")
DIM = 384
DEFAULT_CSV = os.getenv("ARTICLES_CSV", "data/InputData_IndustryEconomics_Jul24-Jun25.csv")
_auto_limit_env = os.getenv("AUTO_EMBED_LIMIT")
try:
    AUTO_EMBED_LIMIT = int(_auto_limit_env) if _auto_limit_env else None
except ValueError:
    AUTO_EMBED_LIMIT = None


class PostgresArticles:
    def __init__(self,
                 host: str = PG_HOST,
                 port: int = PG_PORT,
                 db: str = PG_DB,
                 user: str = PG_USER,
                 password: str = PG_PASSWORD):
        self.host = host
        self.port = port
        self.db = db
        self.user = user
        self.password = password

    # --- Connection helpers ---
    def _conn(self, target_db: Optional[str] = None):
        return psycopg2.connect(
            host=self.host,
            port=self.port,
            dbname=target_db or self.db,
            user=self.user,
            password=self.password,
            connect_timeout=5,
        )

    def ensure_database(self):
        """Create database if it does not exist (ignore if lacking permission)."""
        try:
            conn = self._conn("postgres")
            conn.autocommit = True
            cur = conn.cursor()
            cur.execute("SELECT 1 FROM pg_database WHERE datname=%s", (self.db,))
            if not cur.fetchone():
                cur.execute(f"CREATE DATABASE {self.db}")
                print(f"✅ Created database {self.db}")
            else:
                print(f"ℹ️ Database {self.db} exists")
            cur.close(); conn.close()
        except Exception as e:
            print(f"⚠️ Skipping database ensure (permissions?): {e}")

    def ensure_extension(self):
        try:
            conn = self._conn()
            cur = conn.cursor()
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.commit()
            cur.close(); conn.close()
            print("✅ pgvector extension ensured")
        except Exception as e:
            print(f"❌ Failed ensuring pgvector extension: {e}")
            raise

    # --- Schema ---
    def create_table(self):
        sql = f"""
        CREATE TABLE IF NOT EXISTS articles (
            ArticleID VARCHAR(255) PRIMARY KEY,
            ArtDate DATE,
            Month VARCHAR(20),
            Year INT,
            CompetName VARCHAR(255),
            KPMGTotalImpact DOUBLE PRECISION,
            DeloitteTotalImpact DOUBLE PRECISION,
            EYTotalImpact DOUBLE PRECISION,
            PwCTotalImpact DOUBLE PRECISION,
            Issue TEXT,
            Industry VARCHAR(255),
            Comments TEXT,
            Comments_embedding vector({DIM}),
            SpokespersonName VARCHAR(255),
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW()
        )"""
        try:
            conn = self._conn(); cur = conn.cursor()
            cur.execute(sql)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_articles_year ON articles(Year)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_articles_industry ON articles(Industry)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_articles_compet ON articles(CompetName)")
            conn.commit(); cur.close(); conn.close()
            print("✅ Table 'articles' ensured")
        except Exception as e:
            print(f"❌ Failed creating table: {e}")
            raise

    # --- CSV Import (Step 1) ---
    def import_csv(self, path: str) -> int:
        if not os.path.isfile(path):
            print(f"❌ CSV not found: {path}")
            return 0
        inserted = 0
        conn = self._conn(); cur = conn.cursor()
        upsert = """
        INSERT INTO articles (
            ArticleID, ArtDate, Month, Year, CompetName,
            KPMGTotalImpact, DeloitteTotalImpact, EYTotalImpact, PwCTotalImpact,
            Issue, Industry, Comments, SpokespersonName
        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT (ArticleID) DO UPDATE SET
            ArtDate=EXCLUDED.ArtDate,
            Month=EXCLUDED.Month,
            Year=EXCLUDED.Year,
            CompetName=EXCLUDED.CompetName,
            KPMGTotalImpact=EXCLUDED.KPMGTotalImpact,
            DeloitteTotalImpact=EXCLUDED.DeloitteTotalImpact,
            EYTotalImpact=EXCLUDED.EYTotalImpact,
            PwCTotalImpact=EXCLUDED.PwCTotalImpact,
            Issue=EXCLUDED.Issue,
            Industry=EXCLUDED.Industry,
            Comments=EXCLUDED.Comments,
            SpokespersonName=EXCLUDED.SpokespersonName,
            updated_at=NOW();
        """
        with open(path, newline='', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    art_date = None
                    raw_date = row.get('ArtDate')
                    if raw_date and raw_date.strip():
                        for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y"):
                            try:
                                art_date = datetime.strptime(raw_date.strip(), fmt).date()
                                break
                            except Exception:
                                continue
                    def parse_num(val):
                        if val is None or str(val).strip() == '':
                            return None
                        try:
                            return float(val)
                        except Exception:
                            return None
                    data = (
                        row.get('ArticleID') or f"row_{inserted+1}",
                        art_date,
                        row.get('Month') or None,
                        int(row['Year']) if row.get('Year') and row['Year'].isdigit() else None,
                        row.get('CompetName') or None,
                        parse_num(row.get('KPMGTotalImpact')),
                        parse_num(row.get('DeloitteTotalImpact')),
                        parse_num(row.get('EYTotalImpact')),
                        parse_num(row.get('PwCTotalImpact')),
                        row.get('Issue') or None,
                        row.get('Industry') or None,
                        row.get('Comments') or None,
                        row.get('SpokespersonName') or None,
                    )
                    cur.execute(upsert, data)
                    inserted += 1
                    if inserted % 200 == 0:
                        conn.commit(); print(f"📥 Imported {inserted} rows...")
                except Exception as e:
                    if inserted < 5:
                        print(f"⚠️ Row skip: {e}")
                    continue
        conn.commit(); cur.close(); conn.close()
        print(f"✅ Imported {inserted} rows from {path}")
        return inserted

    # --- Embedding Generation (Step 2) ---
    def generate_embeddings(self, batch_size: int = 50, limit: Optional[int] = None) -> int:
        try:
            import ollama  # type: ignore
        except ImportError:
            print("❌ Install ollama: pip install ollama")
            return 0

        conn = self._conn(); cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM articles WHERE Comments IS NOT NULL AND Comments != '' AND Comments_embedding IS NULL")
        total = cur.fetchone()[0]
        if total == 0:
            print("ℹ️ No pending embeddings.")
            cur.close(); conn.close(); return 0
        remaining = total if limit is None else min(total, limit)
        print(f"🔄 Generating embeddings for up to {remaining} rows (model={EMBED_MODEL}).")

        # Validate model availability
        try:
            ollama.show(EMBED_MODEL)
        except Exception:
            print("⬇️ Pulling model...")
            try:
                ollama.pull(EMBED_MODEL)
            except Exception as e:
                print(f"❌ Model pull failed: {e}")
                cur.close(); conn.close(); return 0

        select_sql = "SELECT ArticleID, Comments FROM articles WHERE Comments IS NOT NULL AND Comments != '' AND Comments_embedding IS NULL LIMIT %s"
        updated = 0
        while remaining > 0:
            cur.execute(select_sql, (batch_size,))
            rows = cur.fetchall()
            if not rows:
                break
            for article_id, comment in rows:
                if not comment or len(comment.strip()) < 5:
                    continue
                try:
                    emb_res = ollama.embed(model=EMBED_MODEL, input=comment.strip())
                    vec = emb_res.get('embedding') or emb_res.get('embeddings')
                    if isinstance(vec, list) and vec and isinstance(vec[0], list):
                        vec = vec[0]
                    if not vec:
                        continue
                    if len(vec) != DIM:
                        if len(vec) > DIM:
                            vec = vec[:DIM]
                        else:
                            vec = vec + [0.0]*(DIM-len(vec))
                    # Use parameter casting for vector
                    cur.execute("UPDATE articles SET Comments_embedding=%s::vector, updated_at=NOW() WHERE ArticleID=%s", (vec, article_id))
                    updated += 1
                    remaining -= 1
                    if updated % 25 == 0:
                        print(f"🧠 {updated} embeddings stored...")
                    if remaining == 0:
                        break
                except Exception as e:
                    print(f"⚠️ Embed fail for {article_id}: {e}")
            conn.commit()
            if len(rows) < batch_size:
                break
        cur.close(); conn.close()
        print(f"✅ Stored {updated} embeddings.")
        return updated

    # --- Similarity Search ---
    def similar(self, query: str, k: int = 5):
        try:
            import ollama  # type: ignore
        except ImportError:
            print("❌ Install ollama: pip install ollama"); return []
        emb_res = ollama.embed(model=EMBED_MODEL, input=query.strip())
        q = emb_res.get('embedding') or emb_res.get('embeddings')
        if isinstance(q, list) and q and isinstance(q[0], list):
            q = q[0]
        if not q:
            print("⚠️ Empty query embedding"); return []
        if len(q) != DIM:
            if len(q) > DIM:
                q = q[:DIM]
            else:
                q = q + [0.0]*(DIM-len(q))
        conn = self._conn(); cur = conn.cursor()
        cur.execute("SELECT ArticleID, Comments, Comments_embedding <-> %s::vector AS distance FROM articles WHERE Comments_embedding IS NOT NULL ORDER BY Comments_embedding <-> %s::vector LIMIT %s", (q, q, k))
        rows = cur.fetchall(); cur.close(); conn.close()
        return rows

    # --- Status ---
    def status(self):
        conn = self._conn(); cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM articles")
        total = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM articles WHERE Comments_embedding IS NOT NULL")
        embedded = cur.fetchone()[0]
        cur.close(); conn.close()
        print(f"📊 Articles: {total} | Embedded: {embedded} | Pending: {total - embedded}")


# --- CLI ---

def build_arg_parser():
    p = argparse.ArgumentParser(description="Postgres articles + embeddings utility")
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("create-db")
    sub.add_parser("create-table")

    imp = sub.add_parser("import")
    imp.add_argument("path")

    emb = sub.add_parser("embed")
    emb.add_argument("--batch-size", type=int, default=50)
    emb.add_argument("--limit", type=int, default=None)

    sub.add_parser("status")

    sim = sub.add_parser("similar")
    sim.add_argument("query")
    sim.add_argument("--k", type=int, default=5)

    return p


def main(argv=None):
    argv = argv or sys.argv[1:]
    parser = build_arg_parser()
    # No args => run full pipeline automatically
    if not argv:
        print("➡️  No arguments supplied: running pipeline: ensure DB+extension+table -> import CSV -> embeddings -> status")
        pg = PostgresArticles()
        pg.ensure_database(); pg.ensure_extension(); pg.create_table()
        if not os.path.isfile(DEFAULT_CSV):
            print(f"❌ Default CSV not found: {DEFAULT_CSV}. Set ARTICLES_CSV env var.")
            return
        pg.import_csv(DEFAULT_CSV)
        pg.generate_embeddings(limit=AUTO_EMBED_LIMIT)
        pg.status()
        return

    args = parser.parse_args(argv)
    pg = PostgresArticles()

    if args.cmd == "create-db":
        pg.ensure_database(); pg.ensure_extension(); pg.create_table()
    elif args.cmd == "create-table":
        pg.ensure_extension(); pg.create_table()
    elif args.cmd == "import":
        pg.ensure_extension(); pg.create_table(); pg.import_csv(args.path)
    elif args.cmd == "embed":
        pg.generate_embeddings(batch_size=args.batch_size, limit=args.limit)
    elif args.cmd == "status":
        pg.status()
    elif args.cmd == "similar":
        rows = pg.similar(args.query, k=args.k)
        for rid, comment, dist in rows:
            snippet = (comment or '')[:70].replace('\n', ' ')
            print(f"{rid}\t{dist:.6f}\t{snippet}...")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
