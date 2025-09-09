from dotenv import load_dotenv
import os
import pandas as pd
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from operator import add as add_messages
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_experimental.agents import create_pandas_dataframe_agent
import psycopg2

load_dotenv()

pg_config = {
    'host': os.getenv('PG_HOST', '127.0.0.1'),
    'port': int(os.getenv('PG_PORT', 5432)),
    'dbname': os.getenv('PG_DB', 'precise_articles'),
    'user': os.getenv('PG_USER', 'postgres'),
    'password': os.getenv('PG_PASSWORD', '')
}

# llm = ChatOllama(
#     model="llama3.1",
#     temperature=0,
# )


embed = OllamaEmbeddings(
    model="nomic-embed-text"
)

# agent = create_pandas_dataframe_agent(
#     llm,
#     df,
#     agent_type="tool-calling",
#     verbose=False,
#     return_intermediate_steps=True,
#     allow_dangerous_code=True,
# )

# response = agent.invoke("how many rows of data are in this file?")
# print(response['output'])
# print(response['intermediate_steps'][-1][0].tool_input.replace('; ', '\n'))