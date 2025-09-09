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
    template="Aggregates:\n{{ aggregates }}\nSummary:\n{{ summary }}\n---\n",
    template_format="jinja2",
)

FEW_SHOT_TEMPLATE = FewShotPromptTemplate(
    examples=[{"aggregates": e["aggregates"], "summary": e["summary"]} for e in EXAMPLES],
    example_prompt=example_prompt,
    prefix=(
        "You are an analyst assistant. You will see example pairs of Aggregates and Summary to learn the tone. "
        "The examples are context only — do not write summaries for them. After the examples, you will receive NEW aggregates and must write ONE monthly executive summary for those aggregates only. "
        "Write a single paragraph of 4–8 sentences in the same narrative style as the examples, prioritising the gist of coverage (what was talked about, by whom, and in what context). "
        "OPENING PATTERN: 'Impact led by {top_spokesperson_name} ({Volume}; {Share}% of {Company} Impact)', followed by semicolon-separated themes they spoke on, drawn from content_samples/issues. Replace placeholders with aggregates: Volume = their article_count; Share = impact_share_pct; Company = company_name (or 'the company' if missing). "
        "STRICT FORMATTING: Put all numbers, statistics, sources, or clarifications in parentheses immediately after the noun. Separate items inside parentheses with semicolons. Do NOT use bullets, lists, colon-led lists, or code blocks. "
        "MANDATORY CONTENT: Include at least two concrete paraphrased content points drawn from content_samples (e.g., 'on US tariffs and recession risk', 'on federal budget “fiscal headroom” and avoiding a downturn', 'on trade war impacts on Australia’s growth'). Use spokesperson names with these points when available. "
        "Then, add 'Additional Impact from ...' listing 1–2 other spokespeople (name in parentheses with a short descriptor drawn from content_samples). If max_impact exists, optionally mention 'highest-impact article {id} ({impact}; {issue}; {spokesperson})'. "
        "Use the following if present: total_articles; avg_impact (to 2dp) using company_impact_field; impact_distribution buckets; max_impact (articleid, impact, issue, spokespersonname); top_spokesperson and/or top_spokesperson_by_impact (and their impact_share_pct if available); top_issues; and content_samples (issue, spokespersonname, excerpt). "
        "Weave 1–3 of the most salient issues and spokespersons into the narrative using content_samples as evidence. Prefer concise language similar to the examples. State 'not reported' where data is missing. Output only the paragraph (no bullets, lists, headings, or code). If aggregates are empty, output 'None'."
    ),
    suffix="Final Aggregates JSON (keys may be subset):\n{{ aggregates }}\n\nWrite ONLY one paragraph in the described style:",
    input_variables=["aggregates"],
    template_format="jinja2",
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
        except Exception as e:
            print(f"Error: {e}")

__all__ = ["Summariser"]
