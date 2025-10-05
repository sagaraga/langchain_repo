import json
import requests
from typing import Dict, List, Optional
#from langchain.document_loaders import JSONLoader
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings,HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from transformers import T5Tokenizer  # Add this import

SWAGGER_URL = "http://127.0.0.1:5000/swagger.json"


def download_json():
    result = requests.get(SWAGGER_URL)
    if result.status_code == 200:
        swagger_json = result.json()
        with open("files/swagger.json","w") as of:
            json.dump(swagger_json,of,indent=4)
        of.close()
        return "files/swagger.json"
    else:
        print(f"Failed to fetch swagger json file")


# 1. Load and process the swagger.json file
def load_swagger_file(file_path: str) -> List:
    # Load JSON file
    with open(file_path, 'r') as file:
        swagger_data = json.load(file)

    # Convert JSON to string for processing
    swagger_str = json.dumps(swagger_data)
    loader = JSONLoader(file_path=file_path,jq_schema=".paths",text_content=False)
    documents = loader.load()

    # Split documents for better processing
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return text_splitter.split_documents(documents)


# 2. Create vector store with Hugging Face embeddings
def create_vector_store(documents):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # In v0.3, FAISS.from_documents is still valid
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store


# 3. Extract API parameters from user input with explicit context
def extract_api_parameters(user_input: str, vector_store) -> Dict:
    # Initialize Hugging Face model pipeline
    llm = HuggingFacePipeline.from_model_id(
        model_id="google/flan-t5-base",
        task="text2text-generation",
        pipeline_kwargs={"max_length": 512}
    )

    # Initialize tokenizer for flan-t5-base
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")

    # Define prompt template
    prompt_template = """
    Given the following user input and Swagger API documentation context,
    extract the relevant API endpoint and parameters:

    User Input: {user_input}

    Context: {context}

    Return the result in this format:
    {
        "endpoint": "extracted_endpoint",
        "method": "HTTP_method",
        "parameters": {
            "param1": "value1",
            "param2": "value2"
        }
    }
    """
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["user_input", "context"]
    )

    # Retrieve context from vector store
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    relevant_docs = retriever.invoke(user_input)
    context = "\n".join([doc.page_content for doc in relevant_docs])

    # Calculate token lengths
    max_tokens = 512
    user_input_tokens = len(tokenizer.encode(user_input))
    template_tokens = len(tokenizer.encode(prompt_template.format(user_input="", context="")))
    available_context_tokens = max_tokens - (user_input_tokens + template_tokens) - 10  # Buffer of 10 tokens

    # Truncate context if necessary
    if available_context_tokens < 0:
        available_context_tokens = 50  # Minimum context length
    context_tokens = tokenizer.encode(context)
    if len(context_tokens) > available_context_tokens:
        truncated_context = tokenizer.decode(context_tokens[:available_context_tokens], skip_special_tokens=True)
    else:
        truncated_context = context

    # Debug output
    print("User Input Tokens:", user_input_tokens)
    print("Template Tokens:", template_tokens)
    print("Context Tokens (original):", len(context_tokens))
    print("Context Tokens (truncated):", len(tokenizer.encode(truncated_context)))
    print("Total Tokens:", user_input_tokens + template_tokens + len(tokenizer.encode(truncated_context)))

    # Format the prompt with truncated context
    formatted_prompt = PROMPT.format(user_input=user_input, context=truncated_context)

    # Invoke the LLM
    result = llm.invoke(formatted_prompt)
    return json.loads(result)
def extract_api_parameters1(user_input: str, vector_store) -> Dict:
    # Initialize Hugging Face model pipeline
    llm = HuggingFacePipeline.from_model_id(
        model_id="google/flan-t5-base",
        task="text2text-generation",
        pipeline_kwargs={"max_length": 512}
    )

    # Define prompt template (cleaned up, no extra whitespace or hidden characters)
    prompt_template = """
    Given the following user input and Swagger API documentation context,
    extract the relevant API endpoint and parameters:

    User Input: {user_input}

    Context: {context}

    Return the result in this format:
    {
        "endpoint": "extracted_endpoint",
        "method": "HTTP_method",
        "parameters": {
            "param1": "value1",
            "param2": "value2"
        }
    }
    """
    # Verify the template
    print("Prompt Template:", prompt_template)  # Debug output

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["user_input", "context"]
    )

    # Explicitly retrieve context from vector store
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    relevant_docs = retriever.invoke(user_input)
    context = "\n".join([doc.page_content for doc in relevant_docs])

    # Debug output
    print("User Input:", user_input)
    print("Context:", context)

    # Format the prompt with user_input and context
    try:
        formatted_prompt = PROMPT.invoke({'user_input':user_input, 'context':context})#PROMPT.format(user_input=user_input, context=context)
        print("Formatted Prompt:", formatted_prompt)  # Debug output
    except KeyError as e:
        print(f"KeyError during formatting: {e}")
        raise

    # Invoke the LLM directly with the formatted prompt
    result = llm.invoke(formatted_prompt)

    # Debug output
    print("LLM Result:", result)

    # Parse the result as JSON
    return json.loads(result)


swagger_json_file_path = download_json()
split_documents = load_swagger_file(swagger_json_file_path)
list_of_vectors = create_vector_store(split_documents)
while True:
    user_input = input("Enter your request (or 'quit' to exit): ")
    if user_input.lower() == 'quit':
        break

    # Extract API details
    api_details = extract_api_parameters(user_input, list_of_vectors)
    print("Extracted API Details:", json.dumps(api_details, indent=2))
