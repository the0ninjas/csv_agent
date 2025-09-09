"""Ingestion node (isolated) for summary_pipeline.
Supports target table override (default: articles_test).
"""
from __future__ import annotations
from typing import TypedDict, Optional, Dict, Any, List
from pg_articles_util import PostgresArticles, DIM
import os, csv
from datetime import datetime

class GraphState(TypedDict, total=False):
    csv_path: Optional[str]
    table: Optional[str]
    import_results: Optional[Dict[str, Any]]
    errors: List[str]


def _ensure_table(pg: PostgresArticles, table: str):
    if table == "articles":
        pg.create_table(); return
    conn = pg._conn(); cur = conn.cursor()
    cur.execute(f"""
    CREATE TABLE IF NOT EXISTS {table} (
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
    )""")
    conn.commit(); cur.close(); conn.close()


def ingest_csv(state: GraphState) -> GraphState:
    path = state.get("csv_path")
    if not path:
        return state
    table = state.get("table") or os.getenv("TEST_TABLE") or "articles_test"
    pg = PostgresArticles()
    try:
        pg.ensure_extension(); _ensure_table(pg, table)
        if table == "articles":  # reuse util logic
            count = pg.import_csv(path)
        else:
            conn = pg._conn(); cur = conn.cursor()
            upsert = f"""
            INSERT INTO {table} (
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
            inserted = 0
            with open(path, newline='', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                for row in reader:
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
                        (row.get('SpokespersonName') or None if (row.get('SpokespersonName') and row.get('SpokespersonName') not in ('--','')) else None),
                    )
                    cur.execute(upsert, data)
                    inserted += 1
            conn.commit(); cur.close(); conn.close()
            count = inserted
        state["import_results"] = {"rows_imported": count, "path": path, "table": table}
    except Exception as e:
        state.setdefault("errors", []).append(f"ingest: {e}")
    return state

__all__ = ["ingest_csv"]
