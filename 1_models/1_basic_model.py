from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI(
    model="gpt-3.5-turbo-instruct",
    temperature=0,
    max_retries=2,
    # api_key="...",
    # base_url="...",
    # organization="...",
    # other params...
)
input_text = "tell me something about India Independence"
result = llm.invoke(input_text)

print(result)

# from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
# from dotenv import load_dotenv
# import os

# load_dotenv()
# HF_TOKEN = os.getenv("HF_TOKEN")


# llm = HuggingFaceEndpoint(
#     #repo_id = "HuggingFaceH4/zephyr-7b-beta",
#     repo_id = "tiiuae/falcon-7b-instruct",
#     #repo_id = "openai/gpt-oss-20b",
#     task= "text-generation",
#     huggingfacehub_api_token=HF_TOKEN
# )


# model = ChatHuggingFace(llm=llm)

# result = llm.invoke("Generate response in English. Hi, I am practising building llm based applications. Started just now. Please wish me")

# print(result.content)

# from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace

# llm = HuggingFaceEndpoint(
#     repo_id="EleutherAI/gpt-neo-2.7B",
#     #provider="novita",
#     max_new_tokens=100,
#     do_sample=False,
#     huggingfacehub_api_token=""
# )
# print(llm.invoke("what is the importance of meditation"))

# from langchain_huggingface import HuggingFaceEndpoint
# from dotenv import load_dotenv
# import os

# # Load your HuggingFace API token from .env
# load_dotenv()
# HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# # Initialize the endpoint with a supported model
# llm = HuggingFaceEndpoint(
#     repo_id="EleutherAI/gpt-neo-2.7B",  # You can change to another supported model
#     task="text-generation",
#     huggingfacehub_api_token="",
#     max_new_tokens=100,
#     do_sample=True
# )

# # Invoke the model
# response = llm.invoke("Generate response in English. Hi, I am practising building LLM-based applications. Please wish me.")
# print(response)