"""Aggregation node module.
Runs predefined metrics queries and stores structured results in state['aggregates'].
Exports: aggregate(state)
"""
from __future__ import annotations
from typing import TypedDict, Optional, Dict, Any, List
import os

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
    aggregates: Optional[Dict[str, Any]]
    errors: List[str]

# Generic (impact-agnostic) queries
GENERIC_QUERIES = {
    "total_articles": "SELECT COUNT(*) FROM articles",
    "top_spokesperson": """SELECT spokespersonname, COUNT(*) AS c
                           FROM articles
                           WHERE spokespersonname IS NOT NULL AND spokespersonname <> ''
                           GROUP BY spokespersonname
                           ORDER BY c DESC
                           LIMIT 1""",
    "monthly_counts": """SELECT month, year, COUNT(*) FROM articles GROUP BY month, year ORDER BY year,
                           MIN(CASE month WHEN 'Jan' THEN 1 WHEN 'Feb' THEN 2 WHEN 'Mar' THEN 3 WHEN 'Apr' THEN 4 WHEN 'May' THEN 5 WHEN 'Jun' THEN 6 WHEN 'Jul' THEN 7 WHEN 'Aug' THEN 8 WHEN 'Sep' THEN 9 WHEN 'Oct' THEN 10 WHEN 'Nov' THEN 11 WHEN 'Dec' THEN 12 ELSE 13 END)""",
}

IMPACT_COLUMNS_PRIORITY = [
    "kpmgtotalimpact",
    "pwctotalimpact",
    "deloittetotalimpact",
    "eytotalimpact",
]

CANONICAL_IMPACT_NAMES = {
    "kpmgtotalimpact": "KPMGTotalImpact",
    "pwctotalimpact": "PwCTotalImpact",
    "deloittetotalimpact": "DeloitteTotalImpact",
    "eytotalimpact": "EYTotalImpact",
}

def _detect_impact_column(cur) -> Optional[str]:
    cur.execute("""
        SELECT lower(column_name) FROM information_schema.columns
        WHERE table_name = 'articles'
    """)
    cols = {r[0] for r in cur.fetchall()}
    candidates = [c for c in IMPACT_COLUMNS_PRIORITY if c in cols]
    for c in cols:
        if c.endswith("totalimpact") and c not in candidates:
            candidates.append(c)
    if not candidates:
        return None
    counts = []
    for col in candidates:
        try:
            cur.execute(f"SELECT COUNT(*) FROM articles WHERE {col} IS NOT NULL AND TRIM({col}::text) <> ''")
            cnt = cur.fetchone()[0]
        except Exception:
            cnt = 0
        counts.append((col, cnt))
    non_zero = [c for c in counts if c[1] > 0] or counts
    non_zero.sort(key=lambda x: (-x[1], IMPACT_COLUMNS_PRIORITY.index(x[0]) if x[0] in IMPACT_COLUMNS_PRIORITY else 999))
    return non_zero[0][0] if non_zero else None

def aggregate(state: GraphState) -> GraphState:
    if psycopg2 is None:
        state.setdefault("errors", []).append("psycopg2 not installed")
        return state
    try:
        conn = psycopg2.connect(host=PG_HOST, port=PG_PORT, dbname=PG_DB, user=PG_USER, password=PG_PASSWORD)
    except Exception as e:
        state.setdefault("errors", []).append(f"aggregate connect: {e}")
        return state
    cur = conn.cursor()
    results: Dict[str, Any] = {}

    # Generic queries
    for key, q in GENERIC_QUERIES.items():
        try:
            cur.execute(q)
            rows = cur.fetchall()
            if key == "total_articles":
                results[key] = rows[0][0]
            elif key == "top_spokesperson" and rows:
                sp, c = rows[0]
                results[key] = {"spokespersonname": sp, "article_count": int(c)}
            elif key == "monthly_counts":
                results[key] = [{"month": r[0], "year": r[1], "count": r[2]} for r in rows]
        except Exception as e:
            state.setdefault("errors", []).append(f"aggregate {key}: {e}")

    # Dynamic impact column
    impact_col = None
    try:
        impact_col = _detect_impact_column(cur)
    except Exception as e:
        state.setdefault("errors", []).append(f"detect_impact: {e}")

    if impact_col:
        pretty = CANONICAL_IMPACT_NAMES.get(impact_col, impact_col)
        results["company_impact_field"] = pretty
        # Average
        try:
            cur.execute(f"SELECT AVG({impact_col}) FROM articles WHERE {impact_col} IS NOT NULL")
            avg_val = cur.fetchone()[0]
            results["avg_impact"] = float(avg_val) if avg_val is not None else None
            if impact_col == "kpmgtotalimpact":
                results["avg_kpmg_impact"] = results["avg_impact"]
        except Exception as e:
            state.setdefault("errors", []).append(f"aggregate avg_impact: {e}")
        # Max impact row
        try:
            cur.execute(f"SELECT articleid, {impact_col}, issue, spokespersonname FROM articles WHERE {impact_col} IS NOT NULL ORDER BY {impact_col} DESC NULLS LAST LIMIT 1")
            row = cur.fetchone()
            if row:
                aid, impact, issue, sp = row
                results["max_impact"] = {"articleid": aid, "impact": float(impact) if impact is not None else None, "issue": issue, "spokespersonname": sp}
        except Exception as e:
            state.setdefault("errors", []).append(f"aggregate max_impact: {e}")
        # Distribution
        try:
            cur.execute(f"""
                SELECT bucket, COUNT(*) FROM (
                  SELECT CASE
                    WHEN {impact_col} IS NULL THEN 'null'
                    WHEN {impact_col} < 1 THEN '0_1'
                    WHEN {impact_col} < 3 THEN '1_3'
                    WHEN {impact_col} < 6 THEN '3_6'
                    ELSE '6_10' END AS bucket
                  FROM articles) t GROUP BY bucket
            """)
            rows = cur.fetchall()
            results["impact_distribution"] = {b: int(c) for b, c in rows if b != 'null'}
        except Exception as e:
            state.setdefault("errors", []).append(f"aggregate impact_distribution: {e}")
    else:
        results["company_impact_field"] = None
        results["avg_impact"] = None
        state.setdefault("errors", []).append("no impact column detected")

    cur.close(); conn.close()
    state["aggregates"] = results
    return state

__all__ = ["aggregate"]
