from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

template = """
You are a {subject} teacher. Please provide the roadmap for this {topic}
"""
prompt = PromptTemplate(
    template = template,
    input_variables = ["subject","topic"]
)

llm = HuggingFaceEndpoint(
    repo_id = "tiiuae/falcon-7b-instruct",
    task = "text-generation"
)
model = ChatHuggingFace(llm=llm)

# Approach  - 1 ( without chains )
#prompt = prompt.invoke({'subject':"AI",'topic':"GenAI"})
# response = model.invoke(prompt)
# print(response.content)

# Approach - 2 ( with chains )
parser = StrOutputParser()
chain = prompt | model | parser
print(chain.invoke({'subject':"AI",'topic':"GenAI"}))