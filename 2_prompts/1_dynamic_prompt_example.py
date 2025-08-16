from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

template = """
You are a good {topic} teacher. 
Please explain the {concept} in five steps summary
"""

final_template = PromptTemplate(
    template=template,
    input_variables=['topic','concept']
)

llm = HuggingFaceEndpoint(
    repo_id = "tiiuae/falcon-7b-instruct",
    task = "text-generation"
)
model = ChatHuggingFace(llm=llm)

# approach - 1
#
# prompt = final_template.invoke({'topic':'sports','concept':'cricket'})
# result = model.invoke(prompt)
# print(result.content)


# approach - 2 . using chains
parser = StrOutputParser()
chain = final_template | model | parser
print(chain.invoke({'topic':'sports','concept':'cricket'}))