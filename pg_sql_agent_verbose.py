"""Simple LangGraph SQL agent for Postgres `articles` database.

Uses:
- langchain_community.utilities.SQLDatabase for connection
- SQLDatabaseToolkit tools (list tables, schema, query, query checker)
- langgraph.prebuilt.create_react_agent for quick agent graph

Environment:
  PG_HOST, PG_PORT, PG_DB, PG_USER, PG_PASSWORD
  LLM_MODEL (chat model id usable with ChatOllama or other provider)
  AGENT_DEBUG=1 to stream intermediate reasoning/tool calls

Example:
  python pg_sql_agent.py "How many articles per year?"
"""
import os
import sys
import re
import argparse
from dotenv import load_dotenv

load_dotenv()

from pgvector.sqlalchemy import Vector  # noqa: F401 (import kept for potential future vector usage)
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama

PG_HOST = os.getenv("PG_HOST", "127.0.0.1")
PG_PORT = int(os.getenv("PG_PORT", 5432))
PG_DB = os.getenv("PG_DB", "precise_articles")
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASSWORD = os.getenv("PG_PASSWORD", "")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1")
DEBUG_ENV = os.getenv("AGENT_DEBUG", "0") == "1"

# Build PostgreSQL URI (no password if blank)
if PG_PASSWORD:
    URI = f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}"
else:
    URI = f"postgresql+psycopg2://{PG_USER}@{PG_HOST}:{PG_PORT}/{PG_DB}"

# --- Helpers -----------------------------------------------------------------

def looks_like_sql(text: str) -> bool:
    if not text:
        return False
    candidate = text.strip().strip("`\n")
    # Take first line (model might prepend commentary)
    first_line = candidate.splitlines()[0].lower()
    return first_line.startswith("select ") or first_line.startswith("with ")


def extract_first_sql_statement(text: str) -> str | None:
    if not text:
        return None
    # crude split by semicolon; keep first SELECT ... ; if present
    pattern = re.compile(r"(select.*?;)(?=\s|$)", re.IGNORECASE | re.DOTALL)
    m = pattern.search(text)
    if m:
        return m.group(1)
    # fallback: if whole content seems SQL
    if looks_like_sql(text):
        return text.strip()
    return None

# --- Agent Build --------------------------------------------------------------

def build_agent():
    db = SQLDatabase.from_uri(URI, include_tables=["articles"], sample_rows_in_table_info=3)
    llm = ChatOllama(model=LLM_MODEL, temperature=0)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()

    # Few-shot pattern to strongly encourage tool usage
    react_fewshot = (
        "Example\nQuestion: Count all articles per competitor.\nThought: I should inspect tables first.\nAction: sql_db_list_tables\nAction Input: ''\nObservation: articles\nThought: I should view the schema to know column names.\nAction: sql_db_schema\nAction Input: articles\nObservation: (schema text)\nThought: I can aggregate counts grouped by competname.\nAction: sql_db_query\nAction Input: SELECT competname, COUNT(articleid) AS article_count FROM articles GROUP BY competname;\nObservation: [('Deloitte', 120), ('KPMG', 110)]\nThought: I have the counts so I can answer.\nFinal Answer: Deloitte has 120 articles; KPMG has 110.\n---\n"
    )

    system_prompt = (
        "You are an agent designed to interact with a PostgreSQL database containing an `articles` table of industry news.\n"
        "Follow a strict ReAct loop with these EXACT markers: Thought, Action, Action Input, Observation, Final Answer.\n\n"
        "Rules:\n"
        "- ALWAYS begin by listing tables (sql_db_list_tables) unless you've already done so in the conversation.\n"
        "- Use sql_db_schema before forming non-trivial queries.\n"
        "- Use sql_db_query_checker for complex queries before execution when uncertainty is high.\n"
        "- Never perform DML (INSERT, UPDATE, DELETE, DROP).\n"
        "- Only select necessary columns; avoid SELECT *.\n"
        "- If user requests aggregated counts by month and company, use GROUP BY month, competname.\n"
        "- Provide a concise natural language Final Answer after observations.\n\n"
        "Schema (all lowercase identifiers):\n"
        "- articleid\n"
        "- artdate\n"
        "- month\n"
        "- year\n"
        "- competname\n"
        "- kpmgtotalimpact\n"
        "- deloitttotalimpact\n"
        "- eytotalimpact\n"
        "- pwctotalimpact\n"
        "- issue\n"
        "- industry\n"
        "- comments\n"
        "- comments_embedding\n"
        "- spokespersonname\n"
        "- created_at\n"
        "- updated_at\n\n"
        f"{react_fewshot}Respond to the next user question now."
    )
    agent = create_react_agent(llm, tools, prompt=system_prompt)
    return agent

# --- Streaming & Invocation ---------------------------------------------------

def stream_question(agent, question: str):
    print(f"Question: {question}\n--- Streaming steps ---")
    for update in agent.stream({"messages": [{"role": "user", "content": question}]}, stream_mode="updates"):
        for node, value in update.items():
            print(f"[Node: {node}]")
            msgs = value.get("messages")
            if msgs:
                last = msgs[-1]
                try:
                    last.pretty_print()
                except Exception:
                    print(last)


def invoke_and_maybe_execute(agent, question: str):
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    final_msg = result["messages"][-1]
    try:
        final_msg.pretty_print()
    except Exception:
        print(final_msg)

    content = getattr(final_msg, "content", "") or ""

    if not content.lower().startswith("final answer"):
        # Maybe the model stopped early; attempt SQL extraction
        sql_candidate = extract_first_sql_statement(content)
        if sql_candidate:
            print("\n[Auto Execution] Detected SQL without final answer; executing:")
            print(sql_candidate)
            try:
                db = SQLDatabase.from_uri(URI, include_tables=["articles"])
                rows = db.run(sql_candidate)
                print("Result:")
                print(rows)
                # Provide synthesized answer
                if isinstance(rows, list) and rows and isinstance(rows[0], tuple):
                    print("Synthesized Answer:")
                    print(rows)
            except Exception as e:
                print("Execution failed:", e)
        elif looks_like_sql(content):
            print("\n[Notice] Content looks like SQL but couldn't parse a full statement.")
    return result

# --- Main ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Run SQL agent against articles DB")
    parser.add_argument("question", type=str, nargs="?", help="Natural language question")
    parser.add_argument("--debug", action="store_true", help="Stream intermediate steps")
    return parser.parse_args()

def main():
    args = parse_args()
    if not args.question:
        print("Provide a natural language question, e.g.: python pg_sql_agent.py 'How many articles mention Deloitte?'")
        return 1

    agent = build_agent()

    if args.debug or DEBUG_ENV:
        stream_question(agent, args.question)
        print("\n--- Invocation (final) ---")

    invoke_and_maybe_execute(agent, args.question)
    return 0

if __name__ == "__main__":
    sys.exit(main())
