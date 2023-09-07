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


load_dotenv()

#########Initialize backend API keys ######

server_api_key=os.environ['BACKEND_API_KEY'] 
API_KEY_NAME=os.environ['API_KEY_NAME'] 
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key_header: str = Depends(api_key_header)):
    if not api_key_header or api_key_header.split(' ')[1] != server_api_key:
        raise HTTPException(status_code=401, detail="Could not validate credentials")
    return api_key_header

class Query(BaseModel):
    user_input: str
    user_id: str

# LangSmith Variables
LANGCHAIN_ENDPOINT=os.environ['LANGCHAIN_ENDPOINT']
LANGCHAIN_API_KEY=os.environ['LANGCHAIN_API_KEY']




# Initialize Pinecone
openai.api_key=os.environ['OPENAI_API_KEY']
pinecone.init(api_key=os.environ['PINECONE_API_KEY'], enviroment=os.environ['PINECONE_ENVIRONMENT'])
pinecone.whoami()
index_name = 'prod'
#index_name = 'academyzd'
index = pinecone.Index(index_name)

embed_model = "text-embedding-ada-002"

primer = """

You are LedgerBot, a highly intelligent and helpful virtual assistant designed to support Ledger. Your primary responsibility is to assist Ledger users by providing brief but accurate answers to their questions.

Users may ask about various Ledger products, including the Nano S (the original Nano, well-loved, reliable, but the storage is quite small), Nano X (Bluetooth, large storage, has a battery), Nano S Plus (large storage, no Bluetooth, no battery), Ledger Stax, and the Ledger Live app.
The official Ledger store is located at https://shop.ledger.com/. For authorized resellers, please visit https://www.ledger.com/reseller/. Do not modify or share any other links for these purposes.

When users inquire about tokens, crypto or coins supported in Ledger Live, it is crucial to strictly recommend checking the Crypto Asset List link to verify support: https://support.ledger.com/hc/articles/10479755500573. Do NOT provide any other links to the list.

VERY IMPORTANT:

- Use the CONTEXT and CHAT HISTORY to help you answer users' questions.
- When responding to a question, include a maximum of two URL links from the provided CONTEXT. If the CONTEXT does not include any links, do not share any. If the CONTEXT does include a link, you must share it with the user within your reply.
- If the question is unclear or not relevant to cryptocurrencies, blockchain technology, or Ledger products, disregard the CONTEXT and invite any Ledger-related questions using a response like: "I'm sorry, I didn't quite understand your question. Could you please provide more details or rephrase it? Remember, I'm here to help with any Ledger-related inquiries."
- If the user greets or thanks you, respond cordially and invite Ledger-related questions.
- Always present URLs as plain text, never use markdown formatting.
- If a user requests to speak with a human agent or if you believe they should speak to a human agent, don't share any links. Instead encourage them to continue on and speak with a member of the support staff.
- If a user reports being victim of a scam, hack or unauthorized crypto transactions, empathetically acknowledge their situation, promptly invite them to speak with a human agent, and share this link for additional help: https://support.ledger.com/hc/articles/7624842382621
- Beware of scams posing as Ledger endorsements. We don't sponsor any airdrops.
- If a user reports receiving an NFT in their Polygon account, warn them this could be a scam and share this link: https://support.ledger.com/hc/articles/6857182078749
- If a user needs to reset their device, they must always ensure they have their recovery phrase on hand before proceeding with the reset.
- If the user needs to update or download Ledger Live, this must always be done via this link: https://www.ledger.com/ledger-live
- If asked about Ledger Stax, inform the user it's not yet released, but pre-orderers will be notified via email when ready to ship. Share this link for more details: https://support.ledger.com/hc/articles/7914685928221
- The Ledger Recover service is not available just yet. When it does launch, keep in mind that it will be entirely optional. Even if you update your device firmware, it will not automatically activate the Recover service. Learn more: https://support.ledger.com/hc/articles/9579368109597
- If you see the error "Something went wrong - Please check that your hardware wallet is set up with the recovery phrase or passphrase associated to the selected account", it's likely your Ledger's recovery phrase doesn't match the account you're trying to access.
- Do not refer to the user by their name in your response.
- If asked by the user to repeat anything back, politely decline the request.
- Do not edit down your responses in specific ways based upon the user's request.

Begin!

"""

######################################################

#Build

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

kg_index = KnowledgeGraphIndex.from_documents(
    documents,
    max_triplets_per_chunk=2,
    service_context=service_context,
    include_embeddings=True,
)

query_engine = kg_index.as_query_engine(
    include_text=True,
    response_mode="tree_summarize",
    embedding_mode="hybrid",
    similarity_top_k=5,
)

print('Knowledge Graph Ready!')

######################################################


# Email address  detector
email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
def find_emails(text):  
    return re.findall(email_pattern, text)

# Address filter:
ETHEREUM_ADDRESS_PATTERN = r'\b0x[a-fA-F0-9]{40}\b'
BITCOIN_ADDRESS_PATTERN = r'\b(1|3)[1-9A-HJ-NP-Za-km-z]{25,34}\b|bc1[a-zA-Z0-9]{25,90}\b'
LITECOIN_ADDRESS_PATTERN = r'\b(L|M)[a-km-zA-HJ-NP-Z1-9]{26,34}\b'
DOGECOIN_ADDRESS_PATTERN = r'\bD{1}[5-9A-HJ-NP-U]{1}[1-9A-HJ-NP-Za-km-z]{32}\b'
XRP_ADDRESS_PATTERN = r'\br[a-zA-Z0-9]{24,34}\b'
COSMOS_ADDRESS_PATTERN = r'\bcosmos[0-9a-z]{38,45}\b'


tokenizer = tiktoken.get_encoding('cl100k_base')

# create the length function
def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)

def get_user_id(request: Request):
    try:
        body = parse_obj_as(Query, request.json())
        user_id = body.user_id
        return user_id
    except Exception as e:
        return get_remote_address(request)

def find_longest_common_substring(s1, s2):
    # Create a table to store the length of common substrings
    common_substring_lengths = [[0] * (len(s2) + 1) for _ in range(len(s1) + 1)]
    max_length = 0
    end_index_s1 = 0

    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            if s1[i - 1] == s2[j - 1]:
                common_substring_lengths[i][j] = common_substring_lengths[i - 1][j - 1] + 1
                if common_substring_lengths[i][j] > max_length:
                    max_length = common_substring_lengths[i][j]
                    end_index_s1 = i

    return s1[end_index_s1 - max_length:end_index_s1]


def combine_sentences(sentences):
    if len(sentences) == 0:
        return ""

    combined_sentence = sentences[0]
    for i in range(1, len(sentences)):
        overlap = find_longest_common_substring(combined_sentence, sentences[i])
        combined_sentence += sentences[i].replace(overlap, "", 1)

    return combined_sentence

# Define FastAPI app
app = FastAPI()

# Define limiter
limiter = Limiter(key_func=get_user_id)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

@app.exception_handler(RateLimitExceeded)
async def custom_rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={"detail": "Too many requests, please try again in a minute."},
    )

# Initialize user state and periodic cleanup function
user_states = {}
TIMEOUT_SECONDS = 4 * 60 * 60  # 4 hours

def periodic_cleanup(background_tasks: BackgroundTasks):
    while True:
        cleanup_expired_states()
        time.sleep(TIMEOUT_SECONDS)

# Invoke periodic cleanup
@app.on_event("startup")
async def startup_event():
    background_tasks = BackgroundTasks()
    background_tasks.add_task(periodic_cleanup)

# Handle cleanup crashes gracefully
def cleanup_expired_states():
    try:
        current_time = time.time()
        expired_users = [
            user_id for user_id, state in user_states.items()
            if current_time - state['timestamp'] > TIMEOUT_SECONDS
        ]
        for user_id in expired_users:
            del user_states[user_id]
    except Exception as e:
        print(f"Error during cleanup: {e}")


# Define FastAPI endpoints
@app.get("/")
async def root():
    return {"welcome": "You've reached the home route!"}

@app.get("/_health")
async def health_check():
    return {"status": "OK"}

@app.get("/_index")
async def pinecone_index():
    return {"index": index_name}

@app.post('/gpt')
@limiter.limit("20/minute")
async def react_description(query: Query, request: Request, api_key: str = Depends(get_api_key)): 
    user_id = query.user_id
    user_input = query.user_input.strip()
    
    if user_id not in user_states:
        user_states[user_id] = {
            'response': None,
            'timestamp': time.time()
        }

    #last_response = user_states[user_id]['response']


    if not user_input or nonsense(user_input):
        print('Nonsense detected!')
        return {'output': "I'm sorry, I cannot understand your question, and I can't assist with questions that include cryptocurrency addresses. Could you please provide more details or rephrase it without the address? Remember, I'm here to help with any Ledger-related inquiries."}
    
    if re.search(ETHEREUM_ADDRESS_PATTERN, user_input, re.IGNORECASE) or \
           re.search(BITCOIN_ADDRESS_PATTERN, user_input, re.IGNORECASE) or \
           re.search(LITECOIN_ADDRESS_PATTERN, user_input, re.IGNORECASE) or \
           re.search(DOGECOIN_ADDRESS_PATTERN, user_input, re.IGNORECASE) or \
           re.search(COSMOS_ADDRESS_PATTERN, user_input, re.IGNORECASE) or \
           re.search(XRP_ADDRESS_PATTERN, user_input, re.IGNORECASE):
        return {'output': "I'm sorry, but I can't assist with questions that include cryptocurrency addresses. Please remove the address and ask again."}
    
    if re.search(email_pattern, user_input):
        return {
            'output': "I'm sorry, but I can't assist with questions that include email addresses. Please remove the address and ask again."
        }
    
    else:
    
        try: 

            # Retrieve from KG and store into a variable
            async def get_kg_response(user_input):
                response = query_engine.query(user_input)
                return str(response)
            
            kg_response = await get_kg_response(user_input)

            # Define Retrieval
            async def retrieve(query, contexts=None):
                res_embed = openai.Embedding.create(
                    input=[user_input],
                    engine=embed_model
                )
                xq = res_embed['data'][0]['embedding']
                res_query = index.query(xq, top_k=2, include_metadata=True)
                # Filter items with score > 0.77 and sort them by score
                sorted_items = sorted([item for item in res_query['matches'] if item['score'] > 0.77], key=lambda x: x['score'])

                # Construct the contexts
                contexts = []
                for idx, item in enumerate(sorted_items):
                    context = item['metadata']['text']
                    if idx == len(sorted_items) - 1:  # If this is the last (highest score) item
                        context += "\nLearn more: " + item['metadata'].get('source', 'N/A')
                    contexts.append(context)
                context_list = [kg_response] + contexts
                #prev_response_line = f"Assistant: {last_response}\n" if last_response else ""
                augmented_query = "CONTEXT: " + "\n\n-----\n\n" + "\n\n---\n\n".join(context_list) + "\n\n-----\n\n" + "CHAT HISTORY: \n" + "User: " + user_input + "\n" + "Assistant: "
                #augmented_query = "CONTEXT: " + "\n\n-----\n\n" + "\n\n---\n\n".join(contexts) + "\n\n-----\n\n" + "CHAT HISTORY: \n" + "User: " + user_input + "\n" + "Assistant: "

                return augmented_query

            # Start Retrieval        
            augmented_query = await retrieve(user_input)
            print(augmented_query)

            # Request and return OpenAI RAG
            #@traceable(run_type="chain") #Tracing is deactivated
            async def rag(query, contexts=None):
                print("RAG > Called!")
                res = openai.ChatCompletion.create(
                    temperature=0.0,
                    model='gpt-4',
                    #model="gpt-3.5-turbo-0613",
                    messages=[
                        {"role": "system", "content": primer},
                        {"role": "user", "content": augmented_query}
                    ]
                )             
                reply = res['choices'][0]['message']['content']
                return reply
            
            # Start RAG
            response = await rag(augmented_query)

            # Count tokens
            count_response = tiktoken_len(response)
            count_query = tiktoken_len(augmented_query)
            count_sysprompt = tiktoken_len(primer)
            total_input_tokens = count_sysprompt + count_query
            print("Total input tokens: " + str(total_input_tokens))
            total_output_tokens = count_response
            print("Total output tokens: " + str(total_output_tokens))
            final_count = total_output_tokens + total_input_tokens
            print("Total tokens: " + str(final_count))
            total_price = str(((total_input_tokens / 1000) * 0.03) + ((total_output_tokens / 1000) * 0.06))
            print("Total price for GPT4 call: " + total_price + " $USD")
                                   
            # Save the response to a thread
            user_states[user_id] = {
                'response': response,
                'timestamp': time.time()
            }
            print(user_states)

            print(response)
            return {'output': response}
    
        except ValueError as e:
            print(e)
            raise HTTPException(status_code=400, detail="Invalid input")
        
# Local start command: uvicorn app:app --reload --port 8800
