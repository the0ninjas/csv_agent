"""LangGraph workflow tying together ingestion, aggregation, and summarisation.

Nodes:
- ingest_csv: optional CSV import using PostgresArticles utility
- aggregate: run SQL metrics queries
- summarise: produce executive summary text

CLI examples:
  python pg_graph.py --csv data/InputData_IndustryEconomics_Jul24-Jun25.csv --period 2025-07
  python pg_graph.py --period 2025-Q3
  python pg_graph.py  (auto uses env ARTICLES_CSV if present, else skips ingest)
"""
from __future__ import annotations
import os
import sys
import argparse
import json
from typing import TypedDict, Optional, Dict, Any, List
from datetime import datetime, date

from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver

from pg_summary_agent import Summariser
from pg_ingest_node import ingest_csv
from pg_aggregate_node import aggregate

# ---------------- State Definition -----------------
class GraphState(TypedDict, total=False):
    csv_path: Optional[str]
    period: Optional[str]
    period_start: Optional[str]
    period_end: Optional[str]
    company: Optional[str]
    import_results: Optional[Dict[str, Any]]
    aggregates: Optional[Dict[str, Any]]
    summary: Optional[str]
    errors: List[str]

# --------------- Helpers ---------------------------

def parse_period(period: Optional[str]) -> tuple[Optional[date], Optional[date]]:
    """Parse period string like '2025-07' (month) or '2025-Q3'."""
    if not period:
        return None, None
    try:
        if "-Q" in period.upper():
            year, q = period.upper().split("-Q")
            y = int(year); qi = int(q)
            start_month = (qi - 1) * 3 + 1
            start = date(y, start_month, 1)
            end_month = start_month + 2
            # end = last day of end_month (approx by next month -1 day)
            if end_month == 12:
                end = date(y, 12, 31)
            else:
                end = date(y, end_month + 1, 1) - timedelta(days=1)  # type: ignore
            return start, end
        # monthly
        dt = datetime.strptime(period, "%Y-%m")
        start = date(dt.year, dt.month, 1)
        if dt.month == 12:
            end = date(dt.year, 12, 31)
        else:
            end = date(dt.year, dt.month + 1, 1) - timedelta(days=1)  # type: ignore
        return start, end
    except Exception:
        return None, None

# --------------- Nodes -----------------------------

from datetime import timedelta

summariser = Summariser()

def summarise(state: GraphState) -> GraphState:
    aggs = state.get("aggregates") or {}
    try:
        text = summariser.run(aggs)
        state["summary"] = text
    except Exception as e:
        state.setdefault("errors", []).append(f"summarise: {e}")
    return state

# -------------- Build Graph ------------------------

def build_graph():
    g = StateGraph(GraphState)
    g.add_node("ingest_csv", ingest_csv)
    g.add_node("aggregate", aggregate)
    g.add_node("summarise", summarise)

    # Linear for now
    g.set_entry_point("ingest_csv")
    g.add_edge("ingest_csv", "aggregate")
    g.add_edge("aggregate", "summarise")
    g.set_finish_point("summarise")
    return g.compile(checkpointer=MemorySaver())

# --------------- CLI ------------------------------

def run_pipeline(csv: Optional[str], period: Optional[str], thread_id: Optional[str] = None):
    """Run the LangGraph workflow.

    thread_id: required by checkpointer; allows resuming runs. If not supplied, a
    deterministic id based on period (when provided) or a timestamp will be used.
    """
    workflow = build_graph()
    init: GraphState = {"csv_path": csv, "period": period}
    # Provide the required configurable key for the MemorySaver checkpointer.
    if not thread_id:
        # Prefer period to allow natural grouping/resume, else timestamp.
        if period:
            thread_id = f"period-{period}".replace(" ", "_")
        else:
            thread_id = datetime.utcnow().strftime("run-%Y%m%d%H%M%S")
    final = workflow.invoke(init, config={"configurable": {"thread_id": thread_id}})
    print(f"[debug] thread_id={thread_id}")
    print("=== Aggregates ===")
    print(json.dumps(final.get("aggregates"), indent=2))
    print("\n=== Summary ===")
    print(final.get("summary"))
    if final.get("errors"):
        print("\nErrors:", final.get("errors"))


def cli(argv=None):
    p = argparse.ArgumentParser(description="Article summary LangGraph pipeline")
    p.add_argument("--csv", help="CSV file to ingest", default=os.getenv("ARTICLES_CSV"))
    p.add_argument("--period", help="Period spec e.g. 2025-07 or 2025-Q3", default=None)
    p.add_argument("--thread", help="Thread id for checkpointing (resume/inspect). If omitted uses period or timestamp.", default=None)
    args = p.parse_args(argv)

    # Only pass csv if file exists
    csv_path = args.csv if args.csv and os.path.isfile(args.csv) else None
    run_pipeline(csv_path, args.period, args.thread)

if __name__ == "__main__":
    cli()
