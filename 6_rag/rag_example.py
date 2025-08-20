from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_core.prompts import PromptTemplate

load_dotenv()


# RAG - Four Steps
# 1. Indexing - Data Source -> Chunking -> Embedding -> Vector Store
# 2. Retrieval - Querying the Vector Store -> Similarity Search -> Document Retrieval
# 3. Augmentation - Augment the prompt with retrieved documents
# 4. Generation - Generate final output using augmented prompt

# 1. Indexing - Data Source -> Chunking -> Embedding -> Vector Store

## load the pdf
loader = PyPDFLoader("./Sivasagar_Marriage_Profile.pdf")
documents = loader.load()
# print(f"Loaded documents: {documents}")
# print(f"Number of documents: {len(documents)}")
# print(f"First document content: {documents[0].page_content}")
# print(f"First document metadata: {documents[0].metadata}")

## Text Splitting - chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=10
)

chunks = splitter.split_documents(documents)
# print(f"Number of chunks: {len(chunks)}")
# print(f"First chunk content: {chunks[0].page_content}")
# print(f"First chunk metadata: {chunks[0].metadata}")

## Embedding - Convert text chunks to embeddings
# embeddings = OpenAIEmbeddings()
# chunk_embeddings = embeddings.embed_documents(chunks)
# print(f"Number of chunk embeddings: {len(chunk_embeddings)}")

## Vector Store - Store embeddings in a vector database
if not os.path.exists('my_chroma_db'):
    vector_store = Chroma(
        embedding_function=OpenAIEmbeddings(),
        persist_directory='my_chroma_db',
        collection_name='sample'
    )
    vector_store.add_documents(chunks)
else:
    print("Loading existing vector store...")
    vector_store = Chroma(
        embedding_function=OpenAIEmbeddings(),
        persist_directory='my_chroma_db',
        collection_name='sample'
    )

# add documents
# print(vector_store.get(include=['embeddings','documents', 'metadatas']))
# print()
# print()
# print(vector_store.get(['7bbb214e-48f7-4f09-9909-7a91ab0e3450']))


## 2. Retrieval - Querying the Vector Store -> Similarity Search -> Document Retrieval
base_retriever = vector_store.as_retriever(search_kwargs={"k": 6}, search_type="similarity")

# Set up the compressor using an LLM
llm = ChatOpenAI(model="gpt-3.5-turbo")
compressor = LLMChainExtractor.from_llm(llm)

# Create the contextual compression retriever
compression_retriever = ContextualCompressionRetriever(
    base_retriever=base_retriever,
    base_compressor=compressor
)


# get user query
user_query = input("Enter your query: ")
# using compressed retriever
compressed_results = compression_retriever.invoke(user_query)
context_text = ""

for i, doc in enumerate(compressed_results):
    # print(f"\n--- Result {i+1} ---")
    # print(doc.page_content)
    context_text += doc.page_content + "\n"

# print(f"\n--- Compressed Results ---\n{context_text}")

# # using normal retriever
# normal_results = base_retriever.invoke(user_query)

# for i, doc in enumerate(normal_results):
#     print(f"\n--- Result {i+1} ---")
#     print(doc.page_content)


## 3. Augmentation - Augment the prompt with retrieved documents

prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)

final_prompt = prompt.invoke({"context": context_text, "question": user_query})

print(f"\n--- Final Prompt ---\n{final_prompt}")

response = llm.invoke(final_prompt)

print(f"\n\n\n --- Query ---\n{user_query}\n--- Response ---\n{response.content}")
