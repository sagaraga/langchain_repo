from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

llm = HuggingFaceEndpoint(
    repo_id='google/gemma-2-2b-it',
    task='text-generation'
)
model = ChatHuggingFace(llm=llm)

class Person(BaseModel):
    name : str = Field(description="provide name of the person")
    city : str = Field(description="provide the city of person")
    age : int = Field(description="provide the age of person")

parser = PydanticOutputParser(pydantic_object=Person)

template1 = """
Generate name, city and age of a fictional {place} person.
{format_instructions}
"""
prompt1 = PromptTemplate(
    template = template1,
    input_variables = ['place'],
    partial_variables = {'format_instructions': parser.get_format_instructions()}
)

chain = prompt1 | model | parser
print(chain.invoke({'place':'india'}))


