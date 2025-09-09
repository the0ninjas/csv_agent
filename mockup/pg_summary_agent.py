"""Isolated summariser for summary_pipeline.
"""
from __future__ import annotations
import os
from typing import Dict, Any, List
import json
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_ollama import ChatOllama

LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1")

_example_json = json.dumps({
    "total_articles": 10,
    "avg_kpmg_impact": 4.3,
    "max_impact": {"articleid": "SYN-010", "kpmgtotalimpact": 9.4, "issue": "Sustainability", "spokespersonname": "Jane Doe"},
    "top_spokesperson": {"spokespersonname": "Jane Doe", "article_count": 5},
    "impact_distribution": {"0_1": 1, "1_3": 2, "3_6": 3, "6_10": 4},
    "monthly_counts": [{"month": "Jul", "year": 2025, "count": 10}],
}, indent=2)
_example_json_escaped = _example_json.replace('{', '{{').replace('}', '}}')
EXAMPLES: List[Dict[str, Any]] = [
    {
        "aggregates_text": _example_json_escaped,
        "summary": (
            "July 2025 produced 10 articles with an elevated average KPMG impact of 4.3 and four high-impact outliers. "
            "SYN-010 on Sustainability led (impact 9.4). Jane Doe drove visibility (5 mentions). Impact spread leaned upper-mid to high."),
    }
]

example_prompt = PromptTemplate(
    input_variables=["aggregates_text", "summary"],
    template="Aggregates JSON:\n{aggregates_text}\nSummary:\n{summary}\n---\n",
)

FEW_SHOT_TEMPLATE = FewShotPromptTemplate(
    examples=EXAMPLES,
    example_prompt=example_prompt,
    prefix=(
        "You are an analyst assistant. Summarise period article metrics succinctly. "
        "Input provides a JSON object of aggregate metrics. "
        "Mention: total volume, average impact, distribution, top-impact article (id, theme, impact), most-cited spokesperson. "
        "3-5 sentences, no bullet formatting, no invented data."),
    suffix="Aggregates JSON to summarise:\n{aggregates_text}\n\nSummary:",
    input_variables=["aggregates_text"],
)

class Summariser:
    def __init__(self, model: str | None = None):
        self.model_name = model or LLM_MODEL
        self.llm = ChatOllama(model=self.model_name, temperature=0)
        self.chain = FEW_SHOT_TEMPLATE | self.llm

    def run(self, aggregates: Dict[str, Any]) -> str:
        safe = dict(aggregates)
        safe.setdefault("total_articles", 0)
        safe.setdefault("avg_kpmg_impact", None)
        safe.setdefault("max_impact", {})
        safe.setdefault("top_spokesperson", {})
        safe.setdefault("impact_distribution", {})
        safe.setdefault("monthly_counts", [])
        agg_json = json.dumps(safe, indent=2, sort_keys=True)
        agg_json_escaped = agg_json.replace('{', '{{').replace('}', '}}')
        resp = self.chain.invoke({"aggregates_text": agg_json_escaped})
        return getattr(resp, 'content', str(resp)).strip()

__all__ = ["Summariser"]
