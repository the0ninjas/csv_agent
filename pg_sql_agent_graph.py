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
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama
from typing_extensions import TypedDict, Annotated
from langgraph.graph import START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

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

class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str

db = SQLDatabase.from_uri(URI, include_tables=["articles"], sample_rows_in_table_info=3)
llm = ChatOllama(model=LLM_MODEL, temperature=0)

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

Only use the following tables:
{table_info}

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
- Pay attention to use only the column names that you can see in the schema
description. Be careful to not query for columns that do not exist. Also,
pay attention to which column is in which table.
- For similarity or embeddings you do NOT generate new embeddings—only query existing columns.
"""

user_prompt = "Question: {input}"

query_prompt_template = ChatPromptTemplate(
    [("system", system_prompt), ("user", user_prompt)]
)

# for message in query_prompt_template.messages:
#     message.pretty_print()

class QueryOutput(TypedDict):
    """Generated SQL query."""

    query: Annotated[str, ..., "Syntactically valid SQL query."]


def write_query(state: State):
    """Generate SQL query to fetch information."""
    prompt = query_prompt_template.invoke(
        {
            "dialect": db.dialect,
            "table_info": db.get_table_info(),
            "input": state["question"],
        }
    )
    structured_llm = llm.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt)
    return {"query": result["query"]}

def execute_query(state: State):
    """Execute SQL query."""
    execute_query_tool = QuerySQLDatabaseTool(db=db)
    return {"result": execute_query_tool.invoke(state["query"])}

def generate_answer(state: State):
    """Answer question using retrieved information as context."""
    prompt = (
        "Given the following user question, corresponding SQL query, "
        "and SQL result, answer the user question.\n\n"
        f"Question: {state['question']}\n"
        f"SQL Query: {state['query']}\n"
        f"SQL Result: {state['result']}"
    )
    response = llm.invoke(prompt)
    return {"answer": response.content}

memory = MemorySaver()
graph_builder = StateGraph(State).add_sequence(
    [write_query, execute_query, generate_answer]
)
graph_builder.add_edge(START, "write_query")
graph = graph_builder.compile(checkpointer=memory, interrupt_before=["execute_query"])

# Now that we're using persistence, we need to specify a thread ID
# so that we can continue the run after review.
config = {"configurable": {"thread_id": "1"}}

for step in graph.stream(
    {"question": "How many articles in year 2024?"},
    config,
    stream_mode="updates",
):
    print(step)

try:
    user_approval = input("Do you want to go to execute query? (yes/no): ")
except Exception:
    user_approval = "no"

if user_approval.lower() == "yes":
    # If approved, continue the graph execution
    for step in graph.stream(None, config, stream_mode="updates"):
        print(step)
else:
    print("Operation cancelled by user.")

# follow the last part to build complete tool and grpah https://langchain-ai.github.io/langgraph/tutorials/sql/sql-agent/?h=sql#3-customizing-the-agent