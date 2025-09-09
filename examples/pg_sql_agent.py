"""Simple LangGraph SQL agent for Postgres `articles` database.

Uses:
- langchain_community.utilities.SQLDatabase for connection
- SQLDatabaseToolkit tools (list tables, schema, query, query checker)
- langgraph.prebuilt.create_react_agent for quick agent graph

Environment:
  PG_HOST, PG_PORT, PG_DB, PG_USER, PG_PASSWORD
  LLM_MODEL (chat model id usable with ChatOllama or other provider)

Example:
  python pg_sql_agent.py "How many articles per year?"
"""
import os
import sys
from dotenv import load_dotenv

load_dotenv()

from pgvector.sqlalchemy import Vector
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

# Build PostgreSQL URI (no password if blank)
if PG_PASSWORD:
    URI = f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}"
else:
    URI = f"postgresql+psycopg2://{PG_USER}@{PG_HOST}:{PG_PORT}/{PG_DB}"

def build_agent():
    db = SQLDatabase.from_uri(URI, include_tables=["articles"], sample_rows_in_table_info=3)
    llm = ChatOllama(model=LLM_MODEL, temperature=0)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()

    system_prompt = """
You are an agent designed to interact with a PostgreSQL database containing an `articles` table of industry news. 
Given an input question, create a syntactically correct {dialect} query to run,
then look at the results of the query and return the answer. 

You can order the results by a relevant column to return the most interesting 
examples in the database. Never query for all the columns from a specific table, 
only ask for the relevant columns given the question. 

You MUST double check your query before executing it. If you get an error while
executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
database.

To start you should ALWAYS look at the tables in the database to see what you
can query. Do NOT skip this step.

You have a PostgreSQL database with a single table: articles.
Schema of articles table (all identifiers are unquoted, so they are queried in lowercase):
- articleid: Unique identifier for the article record.
- artdate: Publication date (may be missing).
- month: Month label (string form, e.g., "Jul").
- year: Publication year (integer; primary for time grouping).
- competname: Competitor or firm the article is about (e.g., Deloitte, KPMG).
-kpmgtotalimpact: Numeric impact score attributed to KPMG in the article.
- deloitttotalimpact: Impact score attributed to Deloitte.
- eytotalimpact: Impact score attributed to EY.
- pwctotalimpact: Impact score attributed to PwC.
- issue: Topical issue/theme classification.
- industry: Industry segment referenced.
- comments: Freetext commentary or summary.
- comments_embedding: Vector(384) embedding of comments (for semantic similarity).
- spokespersonname: Named spokesperson cited (if any).
- created_at: Row creation timestamp.
- updated_at: Last modification timestamp.

Guidelines:
- Only select needed columns.
- For similarity or embeddings you do NOT generate new embeddings—only query existing columns.
""".format(
    dialect=db.dialect,
)
    agent = create_react_agent(llm, tools, prompt=system_prompt)
    return agent

def stream_question(agent, question: str):
    print(f"Question: {question}\n--- Streaming steps ---")
    # Use updates to see each tool node
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
                    
def main():
    if len(sys.argv) < 2:
        print("Provide a natural language question, e.g.: python pg_sql_agent.py 'How many articles mention Deloitte?'")
        return 1
    question = sys.argv[1]
    agent = build_agent()


    # Prefer full streaming with tool observations
    stream_question(agent, question)

    # If the final message looks like raw SQL (no natural language), auto-execute it:
    print("\n--- Ensuring execution / final answer ---")
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    final_content = result["messages"][-1].content

    # print(f"Question: {question}\n--- Streaming steps ---")
    # for step in agent.stream({"messages": [{"role": "user", "content": question}]}, stream_mode="values"):
    #     msg = step["messages"][-1]
    #     try:
    #         msg.pretty_print()
    #     except Exception:
    #         print(msg)
    return 0

if __name__ == "__main__":
    sys.exit(main())
