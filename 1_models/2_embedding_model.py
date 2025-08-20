from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

# using openai embeddings

from langchain_openai import OpenAIEmbeddings

embed = OpenAIEmbeddings(
    model="text-embedding-3-large"
    # With the `text-embedding-3` class
    # of models, you can specify the size
    # of the embeddings you want returned.
    # dimensions=1024
)

documents = [
    "The meaning of life is 42.",
    "The capital of France is Paris.",
    "The largest ocean on Earth is the Pacific Ocean."
]

embedded_docs = embed.embed_documents(documents)
query = "What is the capital of France?"

embedded_query = embed.embed_query(query)

result = cosine_similarity([embedded_query],embedded_docs)[0]

print(f"Query   : {query}. \nRespose : {documents[sorted(enumerate(result), key=lambda x: x[1], reverse=True)[0][0]]}")

# exit(0)
# print("exited")


# # using huggingface embeddings
# from langchain_huggingface import HuggingFaceEmbeddings


# model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# documents = [
#     "Virat kohli has best cover drive",
#     "Rohit Sharma has best pull shot",
#     "Dhoni has best helicopter shot"
# ]

# doc_embeddings = model.embed_documents(documents)

# query = "tell me about Dhoni"

# query_embedding = model.embed_query(query)

# result = cosine_similarity([query_embedding],doc_embeddings)[0]
# print(documents[sorted(list(enumerate(result)), key=lambda x:x[1])[-1][0]])