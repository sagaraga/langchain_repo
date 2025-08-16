from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

documents = [
    "Virat kohli has best cover drive",
    "Rohit Sharma has best pull shot",
    "Dhoni has best helicopter shot"
]

doc_embeddings = model.embed_documents(documents)

query = "tell me about Dhoni"

query_embedding = model.embed_query(query)

result = cosine_similarity([query_embedding],doc_embeddings)[0]
print(documents[sorted(list(enumerate(result)), key=lambda x:x[1])[-1][0]])