"""Isolated summary pipeline graph (no changes to root files).
Usage:
  python summary_pipeline/pg_graph.py --csv summary_pipeline/data/synthetic_articles.csv --table articles_test
"""
from __future__ import annotations
import os, sys, argparse, json
from typing import TypedDict, Optional, Dict, Any, List
from langgraph.graph import StateGraph

from pg_articles_util import PG_DB, PG_HOST, PG_PORT, PG_USER, PG_PASSWORD
from mockup.pg_summary_agent import Summariser
from mockup.pg_ingest_node import ingest_csv
from mockup.pg_aggregate_node import aggregate

class GraphState(TypedDict, total=False):
    csv_path: Optional[str]
    table: Optional[str]
    aggregates: Optional[Dict[str, Any]]
    import_results: Optional[Dict[str, Any]]
    summary: Optional[str]
    errors: List[str]

summariser = Summariser()

def summarise(state: GraphState) -> GraphState:
    aggs = state.get("aggregates") or {}
    try:
        state["summary"] = summariser.run(aggs)
    except Exception as e:
        state.setdefault("errors", []).append(f"summarise: {e}")
    return state

def build_graph():
    g = StateGraph(GraphState)
    g.add_node("ingest", ingest_csv)
    g.add_node("aggregate", aggregate)
    g.add_node("summarise", summarise)
    g.set_entry_point("ingest")
    g.add_edge("ingest", "aggregate")
    g.add_edge("aggregate", "summarise")
    g.set_finish_point("summarise")
    return g.compile()

def run_pipeline(csv: Optional[str], table: Optional[str]):
    wf = build_graph()
    init: GraphState = {"csv_path": csv, "table": table}
    final = wf.invoke(init)
    print("=== Import ===")
    print(json.dumps(final.get("import_results"), indent=2))
    print("\n=== Aggregates ===")
    print(json.dumps(final.get("aggregates"), indent=2))
    print("\n=== Summary ===")
    print(final.get("summary"))
    if final.get("errors"):
        print("\nErrors:", final.get("errors"))

def cli(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=None, help="CSV path")
    p.add_argument("--table", default="articles_test", help="Target table name")
    args = p.parse_args(argv)
    csv = args.csv if args.csv and os.path.isfile(args.csv) else None
    run_pipeline(csv, args.table)

if __name__ == "__main__":
    cli()
