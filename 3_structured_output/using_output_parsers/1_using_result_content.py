from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate

from first_script import final_prompt

llm = HuggingFaceEndpoint(
    repo_id='tiiuae/falcon-7b-instruct',
    task='text-generation'
)
model = ChatHuggingFace(llm=llm)

template1 = """
Given the topic name {topic}. Please generate notes of 100 lines on {topic}
"""
prompt1 = PromptTemplate(
    template=template1,
    input_variables=['topic']
)
final_prompt1 = prompt1.invoke({'topic':'sports'})
result1 = model.invoke(final_prompt1)
print(result1.content)


template2="""
Given the text {notes}. Please generate summary in 3 points on {notes} in the simple list with numbers like point-1, point-2 and point-3
"""
prompt2=PromptTemplate(
    template=template2,
    input_variables=['notes']
)
final_prompt2 = prompt2.invoke({'notes':result1.content})
result2 = model.invoke(final_prompt2)
print(result2)