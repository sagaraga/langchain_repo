from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv

load_dotenv()


llm = HuggingFaceEndpoint(
    #repo_id = "HuggingFaceH4/zephyr-7b-beta",
    repo_id = "tiiuae/falcon-7b-instruct",
    #repo_id = "openai/gpt-oss-20b",
    task= "text-generation"
)


model = ChatHuggingFace(llm=llm)

result = model.invoke("Generate response in English. Hi, I am practising building llm based applications. Started just now. Please wish me")

print(result.content)