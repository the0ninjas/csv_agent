"""Summariser module for article aggregation results.

Takes structured aggregation metrics (dict) and produces a concise period summary.
Uses FewShotPromptTemplate with synthetic examples to steer style.
"""
from __future__ import annotations
import os
from typing import Dict, Any, List

from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_ollama import ChatOllama
from prompt_example import EXAMPLES

LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1")

example_prompt = PromptTemplate(
    input_variables=["aggregates", "summary"],
    template="Aggregates:\n{aggregates}\nSummary:\n{summary}\n---\n",
)

FEW_SHOT_TEMPLATE = FewShotPromptTemplate(
    examples=[{"aggregates": e["aggregates"], "summary": e["summary"]} for e in EXAMPLES],
    example_prompt=example_prompt,
    prefix=(
        "You are an analyst assistant. Given aggregate media metrics produce an executive summary. "
        "Include: total articles, average impact (2dp) for detected company impact field (company_impact_field), distribution buckets, highest impact article (id if available) with impact, issue and spokesperson, and top spokesperson overall. "
        "State 'not reported' where data missing. If aggregates empty return None. Keep concise (<6 sentences unless articles > 1200)."
    ),
    suffix="Aggregates JSON (keys may be subset):\n{aggregates}\n\nSummary:",
    input_variables=["aggregates"],
)

class Summariser:
    def __init__(self, model: str | None = None):
        self.model_name = model or LLM_MODEL
        self.llm = ChatOllama(model=self.model_name, temperature=0)
        self.chain = FEW_SHOT_TEMPLATE | self.llm

    def run(self, aggregates: Dict[str, Any]) -> str:
        if not aggregates:
            return "None"
        try:
            return self.chain.invoke({"aggregates": aggregates}).content.strip()
        except Exception:
            return self._fallback(aggregates)

    def _fallback(self, aggs: Dict[str, Any]) -> str:
        total = aggs.get("total_articles")
        avg = aggs.get("avg_impact") or aggs.get("avg_kpmg_impact")
        dist = aggs.get("impact_distribution") or {}
        max_row = aggs.get("max_impact") or {}
        top_sp = aggs.get("top_spokesperson") or {}
        parts = []
        if total is not None: parts.append(f"Total articles: {total}.")
        if avg is not None: parts.append(f"Average impact: {avg:.2f}.")
        if dist:
            parts.append("Dist " + ", ".join(f"{k}:{v}" for k,v in sorted(dist.items())))
        if max_row:
            parts.append(f"Top {max_row.get('articleid')} {max_row.get('impact')} {max_row.get('issue')} {max_row.get('spokespersonname')}")
        if top_sp:
            parts.append(f"Top spokesperson {top_sp.get('spokespersonname')} ({top_sp.get('article_count')})")
        return " ".join(parts) if parts else "None"

__all__ = ["Summariser"]
