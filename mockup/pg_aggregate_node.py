"""Aggregation node (isolated) with dynamic table.
"""
from __future__ import annotations
from typing import TypedDict, Optional, Dict, Any, List
import os, re

try:
    import psycopg2  # type: ignore
except ImportError:
    psycopg2 = None  # type: ignore

PG_HOST = os.getenv("PG_HOST", "127.0.0.1")
PG_PORT = int(os.getenv("PG_PORT", 5432))
PG_DB = os.getenv("PG_DB", "precise_articles")
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASSWORD = os.getenv("PG_PASSWORD", "")

class GraphState(TypedDict, total=False):
    table: Optional[str]
    aggregates: Optional[Dict[str, Any]]
    errors: List[str]

BASE_QUERIES = {
    "total_articles": "SELECT COUNT(*) FROM {table}",
    "avg_kpmg_impact": "SELECT AVG(kpmgtotalimpact) FROM {table} WHERE kpmgtotalimpact IS NOT NULL",
    "max_impact_row": "SELECT articleid, kpmgtotalimpact, issue, spokespersonname FROM {table} ORDER BY kpmgtotalimpact DESC NULLS LAST LIMIT 1",
    "top_spokesperson": "SELECT spokespersonname, COUNT(*) AS c FROM {table} WHERE spokespersonname IS NOT NULL AND spokespersonname <> '' GROUP BY spokespersonname ORDER BY c DESC LIMIT 1",
    "impact_buckets": """
        SELECT bucket, COUNT(*) FROM (
          SELECT CASE 
            WHEN kpmgtotalimpact IS NULL THEN 'null'
            WHEN kpmgtotalimpact < 1 THEN '0_1'
            WHEN kpmgtotalimpact < 3 THEN '1_3'
            WHEN kpmgtotalimpact < 6 THEN '3_6'
            ELSE '6_10' END AS bucket
          FROM {table}) t GROUP BY bucket""",
    "monthly_counts": "SELECT month, year, COUNT(*) FROM {table} GROUP BY month, year ORDER BY year, MIN(CASE month WHEN 'Jan' THEN 1 WHEN 'Feb' THEN 2 WHEN 'Mar' THEN 3 WHEN 'Apr' THEN 4 WHEN 'May' THEN 5 WHEN 'Jun' THEN 6 WHEN 'Jul' THEN 7 WHEN 'Aug' THEN 8 WHEN 'Sep' THEN 9 WHEN 'Oct' THEN 10 WHEN 'Nov' THEN 11 WHEN 'Dec' THEN 12 ELSE 13 END)",
}

def aggregate(state: GraphState) -> GraphState:
    if psycopg2 is None:
        state.setdefault("errors", []).append("psycopg2 not installed")
        return state
    table = state.get("table") or os.getenv("TEST_TABLE") or "articles_test"
    if not re.fullmatch(r"[A-Za-z0-9_]+", table):
        state.setdefault("errors", []).append("invalid table name")
        return state
    conn = psycopg2.connect(host=PG_HOST, port=PG_PORT, dbname=PG_DB, user=PG_USER, password=PG_PASSWORD)
    cur = conn.cursor()
    queries = {k: v.format(table=table) for k, v in BASE_QUERIES.items()}
    results: Dict[str, Any] = {}
    for key, q in queries.items():
        try:
            cur.execute(q)
            rows = cur.fetchall()
            if key == "total_articles":
                results[key] = rows[0][0]
            elif key == "avg_kpmg_impact":
                results[key] = float(rows[0][0]) if rows[0][0] is not None else None
            elif key == "max_impact_row":
                if rows:
                    aid, impact, issue, sp = rows[0]
                    results["max_impact"] = {"articleid": aid, "kpmgtotalimpact": float(impact) if impact is not None else None, "issue": issue, "spokespersonname": sp}
            elif key == "top_spokesperson":
                if rows:
                    sp, c = rows[0]
                    results[key] = {"spokespersonname": sp, "article_count": int(c)}
            elif key == "impact_buckets":
                results["impact_distribution"] = {r[0]: int(r[1]) for r in rows}
            elif key == "monthly_counts":
                results[key] = [{"month": r[0], "year": r[1], "count": r[2]} for r in rows]
        except Exception as e:
            state.setdefault("errors", []).append(f"aggregate {key}: {e}")
    cur.close(); conn.close()
    state["aggregates"] = results
    return state

__all__ = ["aggregate"]
