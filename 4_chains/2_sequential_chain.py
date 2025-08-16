from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

template1 = """
You are good {subject} teacher. Provide deep analysis on the {topic} in 25 lines.
"""
template2 = """
On the given {text}, extract the summary and important concept in 3 points
"""

prompt1 = PromptTemplate(
    template = template1,
    input_variables = ["subject","topic"]
)

prompt2 = PromptTemplate(
    template = template2,
    input_variables = ["text"]
)

llm = HuggingFaceEndpoint(
    repo_id = "tiiuae/falcon-7b-instruct",
    task = "text-generation"
)

model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser
#print(chain.invoke({"subject":"Mathematics","topic":"linear algebra"}))

chain.get_graph().print_ascii()