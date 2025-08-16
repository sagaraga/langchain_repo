from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()
llm = HuggingFaceEndpoint(
    repo_id='google/gemma-2-2b-it',
    task='text-generation'
)
model = ChatHuggingFace(llm=llm)

template1 = """
Given the topic name {topic}. Please generate notes of 100 lines on {topic} in the given format.

"""
prompt1 = PromptTemplate(
    template = template1,
    input_variables = ['topic']
    #partial_variables = {'format_instructions': parser.get_format_instructions()}
)

template2="""
Given the text {notes}. Please generate summary in 3 points on {notes} in the simple list with numbers like point-1, point-2 and point-3.
in the given format.

"""
prompt2=PromptTemplate(
    template=template2,
    input_variables=['notes']
    #partial_variables = {'format_instructions': parser.get_format_instructions()}
)


chain = prompt1 | model | parser | prompt2 | model | parser
result = chain.invoke({'topic':'science'})
print(result)
