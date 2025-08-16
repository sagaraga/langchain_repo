import json
import os.path
import requests
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
import re
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.output_parsers import JsonOutputParser
import streamlit as st


SWAGGER_URL = "http://127.0.0.1:5000/swagger.json"

schema = [
    ResponseSchema(name='URL', description= "rest api url"),
    ResponseSchema(name='Method', description="get the method type of rest api"),
    ResponseSchema(name='user_id', description="unique ID of the user of integer type"),
    ResponseSchema(name='name', description="name of the user"),
    ResponseSchema(name='email', description="email id of the user")

]
parser = StructuredOutputParser.from_response_schemas(schema)


# download the json file from given URL
def download_json():
    result = requests.get(SWAGGER_URL)
    if result.status_code == 200:
        swagger_json = result.json()
        return swagger_json
    else:
        print(f"Failed to fetch swagger json file")


# Parse the json received from URL and extract the details ( like endpoint, method, request body )
# for each api
def extract_api_details(swagger_data):
    endpoints = []
    base_url = swagger_data.get("servers", [{"url": ""}])[0]["url"]

    for path, methods in swagger_data["paths"].items():
        for method, details in methods.items():
            api_info = {
                "endpoint": path,
                "method": method.upper(),
                "summary": details.get("summary", ""),
                "description": details.get("description", ""),
                "full_url": base_url + path,
            }
            # Extract request body (if present)
            if "requestBody" in details:
                request_body = details["requestBody"].get("content", {}).get("application/json", {}).get("schema", {})
                api_info["requestBody"] = {
                    "type": request_body.get("type", "unknown"),
                    "required": request_body.get("required", []),
                    "properties": request_body.get("properties", {})
                }
            endpoints.append(api_info)
    return endpoints


# Append all the details of an api into a string.
# The whole string for each api will be considered as one Document and used for training
def format_api_for_embedding(api_info):
    text = f"URL: {api_info['full_url']}\n"
    text += f"Method: {api_info['method']}\n"
    text += f"Summary: {api_info['summary']}\n"

    if api_info.get("requestBody",{}):
        text += f"Request Body (Type: {api_info['requestBody']['type']}):\n"
        for param_name, param_details in api_info["requestBody"]["properties"].items():
            required_flag = "(Required)" if param_name in api_info["requestBody"]["required"] else ""
            example = param_details.get("example", "N/A")
            text += f" - {param_name} {required_flag}: {param_details.get('type', 'unknown')} (Example: {example})\n"
    return text


def create_vector_store(embed_model):
    if os.path.exists("faiss_index"):
        print("embeddings already exist")
    else:
        api_embeddings = embed_model.embed_documents(api_texts)
        text_embeddings = list(zip(api_texts, api_embeddings))
        # Store embeddings in FAISS
        vector_store = FAISS.from_embeddings(text_embeddings, embed_model)
        # Save FAISS index
        vector_store.save_local("faiss_index")


def get_rest_api(embed_model,query):
    retrieved_vector_store = FAISS.load_local("faiss_index", embed_model, allow_dangerous_deserialization=True)
    #query = "get the update the user details"
    retrieved_docs = retrieved_vector_store.similarity_search(query, k=1)
    return retrieved_docs[0].page_content


def extract_api_url_method(api_text,request_body):
    url = request_body.get('URL')
    method = re.search(r"Method:\s*(.*)", api_text).group(1)
    return url.strip(), method.strip()

def combine_swagger_data_with_user_query(api_info,user_query):
    llm = HuggingFaceEndpoint(
        repo_id='google/gemma-2-2b-it',
        task='text-generation'
    )
    model = ChatHuggingFace(llm=llm)

    # LLM Prompt to extract structured data
    prompt = PromptTemplate(
        input_variables=["query","rest_api_info"],
        partial_variables={"format_instructions":parser.get_format_instructions()},
        template="""
        Extract structured data from the user query below.
        User Query: "{query}"
        
        The rest api information is given as {rest_api_info}. Also extract the URL from the given information.
        If the  URL field in {rest_api_info} contains any dynamic variables, then replace any variables in the URL from the user provided information.
        Ex: If the URL is http://127.0.0.1:5000/users/{{user_id}}, replace {{user_id}} with the information provided in user query.
        If the URL doesn't contains any variable, don't replace anything in the URL
        Ex: If the URL is http://127.0.0.1:5000/users/, get the same URL
        Return output in JSON format with keys matching API request body fields in given below format
        {format_instructions}   
        """
    )

    final_prompt = prompt.invoke({'query':user_query,'rest_api_info':api_info})
    structured_data = model.invoke(final_prompt)

    # Convert extracted text to JSON
    json_parser = JsonOutputParser()
    request_body = json_parser.parse(structured_data.content)  # Automatically extracts JSON

    api_url, api_method = extract_api_url_method(api_info,request_body)

    # Make API Request dynamically
    print(f"Calling API: {api_method} {api_url}")
    print(f"Request Body: {json.dumps(request_body, indent=2)}")

    response_from_api = None
    if api_method == "GET":
        response_from_api = requests.get(api_url, params=request_body)
    elif api_method == "POST":
        response_from_api = requests.post(api_url, json=request_body)
    elif api_method == "PUT":
        response_from_api = requests.put(api_url, json=request_body)
    elif api_method == "DELETE":
        response_from_api = requests.delete(api_url, json=request_body)

    return response_from_api


if __name__ == '__main__':
    swagger_json  = download_json()
    print(swagger_json)
    api_endpoints = extract_api_details(swagger_json)
    print(api_endpoints)
    api_texts = [format_api_for_embedding(api) for api in api_endpoints]
    print(api_texts)
    # api_texts is a list of strings, where each string represents formatted API information

    # Load embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    create_vector_store(embedding_model)

    # Initialize session state for storing previous queries
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""

    st.title("Rest API Agent")

    user_input = st.text_input("Enter your query:", value=st.session_state.user_input)

    # Submit button
    if st.button("Submit"):
        if user_input.strip():
            # Store last input in session state
            st.session_state.user_input = user_input

            # retrieve the relevant api from vector store
            rest_api_end_point = get_rest_api(embedding_model, user_input)
            print(rest_api_end_point)
            response = combine_swagger_data_with_user_query(rest_api_end_point,user_input)

            if response:
                formatted_json = json.dumps(response.json(), indent=4)
                st.code(formatted_json, language="json")
            else:
                st.write("### Response:")
                st.success(f"No Response Received")

            # Clear input field for next query
            st.session_state.user_input = ""




