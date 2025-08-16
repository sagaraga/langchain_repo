from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

parser = JsonOutputParser()
llm = HuggingFaceEndpoint(
    repo_id='google/gemma-2-2b-it',
    task='text-generation'
)
model = ChatHuggingFace(llm=llm)

template1 = """
Given the topic name {topic}. Please generate notes of 100 lines on {topic} in the given format.
{format_instructions}
"""
prompt1 = PromptTemplate(
    template = template1,
    input_variables = ['topic'],
    partial_variables = {'format_instructions': parser.get_format_instructions()}
)

chain = prompt1 | model | parser
print(chain.invoke({'topic':'cricket'}))


