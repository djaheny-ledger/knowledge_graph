import os
import json
import time
import re
import logging
import sys

from dotenv import load_dotenv
import pinecone
import openai
import nltk
from nostril import nonsense
import tiktoken
from langsmith.run_helpers import traceable

from fastapi import FastAPI, Request, HTTPException, status, Depends, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.security import APIKeyHeader

from pydantic import BaseModel, parse_obj_as

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from starlette.status import HTTP_429_TOO_MANY_REQUESTS

from llama_index import (
    SimpleDirectoryReader,
    ServiceContext,
    KnowledgeGraphIndex,
    download_loader,
)
from llama_index.graph_stores import SimpleGraphStore
from llama_index.llms import OpenAI
from llama_index.storage.storage_context import StorageContext


# Initialize OpenAI credentials
load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']

# Set up logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# Initialize the language model
llm = OpenAI(temperature=0, model="gpt-3.5-turbo")
service_context = ServiceContext.from_defaults(llm=llm, chunk_size=512)

# Load documents from a directory
data_path = os.path.join("data")
data = SimpleDirectoryReader(data_path).load_data()
print('Documents loaded!')

# Load the contents of the JSON file
with open(os.path.join(data_path, "output.json"), 'r') as json_file:
    json_data = json.load(json_file)
    # Assuming json_data is a list of dictionaries, extract 'text' values
    text_data = [item['text'] for item in json_data if 'text' in item]

# Save the 'text' values back into a new JSON file
new_json_path = os.path.join(data_path, "processed_output.json")
with open(new_json_path, 'w') as new_json_file:
    json.dump(text_data, new_json_file)

# Count the number of characters in processed_output.json
with open(new_json_path, 'r') as new_json_file:
    content = new_json_file.read()
    char_count = len(content)
    print(f"The processed_output.json file contains {char_count} characters.")

# JSON data loader: 'download_loader' retrieves some loader class.
JSONReader = download_loader("JSONReader")
loader = JSONReader()
documents = loader.load_data(new_json_path)

graph_store = SimpleGraphStore()
storage_context = StorageContext.from_defaults(graph_store=graph_store)
print('Json file loaded!')

### BUILD AND USE BASIC KEYWORD QUERY

# # Build a knowledge graph index from the documents (may be time-consuming)
# index = KnowledgeGraphIndex.from_documents(
#     documents,
#     max_triplets_per_chunk=2,
#     storage_context=storage_context,
#     service_context=service_context,
# )
# print('Knowledge graph built!')

# # Query the knowledge graph
# query_engine = index.as_query_engine(include_text=False, response_mode="tree_summarize")
# response = query_engine.query("Tell me more about Polkadot")
# print(response)

## BUILD GRAPH AND QUERY WITH EMBEDDINGS ##

index = KnowledgeGraphIndex.from_documents(
    documents,
    max_triplets_per_chunk=2,
    service_context=service_context,
    include_embeddings=True,
)

query_engine = index.as_query_engine(
    include_text=True,
    response_mode="tree_summarize",
    embedding_mode="hybrid",
    similarity_top_k=5,
)
response = query_engine.query(
    "Tell me more about Polkadot staking",
)
print(response)