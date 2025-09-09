"""Ingestion node module.
Ensures extension/table then optionally imports CSV into `articles`.
Exports: ingest_csv(state)
"""
from __future__ import annotations
from typing import TypedDict, Optional, Dict, Any, List
from pg_articles_util import PostgresArticles

class GraphState(TypedDict, total=False):
    csv_path: Optional[str]
    import_results: Optional[Dict[str, Any]]
    errors: List[str]


def ingest_csv(state: GraphState) -> GraphState:
    path = state.get("csv_path")
    if not path:
        return state
    pg = PostgresArticles()
    try:
        pg.ensure_extension(); pg.create_table()
        count = pg.import_csv(path)
        state["import_results"] = {"rows_imported": count, "path": path}
    except Exception as e:
        state.setdefault("errors", []).append(f"ingest: {e}")
    return state

__all__ = ["ingest_csv"]
