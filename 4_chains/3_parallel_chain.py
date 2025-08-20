# using OpenAI APIs

from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableParallel
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    model="gpt-3.5-turbo"
)


template1 = """
Give the topic name {topic}. Please generate a details notes about 20 sentences.
"""

template2 = """
Given the topic name {topic}, Please generate 5 questions for quiz.
"""

template3 = """
Merge the {notes} and {quiz} into a single document
"""

prompt1 = PromptTemplate(
    template=template1,
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template=template2,
    input_variables=["topic"]
)

prompt3 = PromptTemplate(
    template=template3,
    input_variables=["notes", "quiz"]
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    "notes": prompt1 | llm | parser,
    "quiz": prompt1 | llm | parser
})

merge_chain = prompt3 | llm | parser

final_chain = parallel_chain | merge_chain

result = final_chain.invoke({"topic": "Meditation"})

print(result)


# using Hugginface APIs
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence,RunnableParallel
from dotenv import load_dotenv
load_dotenv()

from langchain.globals import set_llm_cache
from langchain_community.cache import InMemoryCache
set_llm_cache(InMemoryCache())

template1 = """
You are subject matter expert on Health Science. Generate 20 lines of text on the {topic}
"""
prompt1 = PromptTemplate(
    template = template1,
    input_variables = ["topic"]
)

template2 = """
Extract 2 important points from the {text}
"""
prompt2 = PromptTemplate(
    template = template2,
    input_variables = ["text"]
)

template3 = """
Generate the 2 questions quiz from the {text}
"""
prompt3 = PromptTemplate(
    template = template3,
    input_variables = ["text"]
)

template4 = """
Combine the {notes} and {quiz} into a single document
"""
prompt4 = PromptTemplate(
    template = template4,
    input_variables = ["notes","quiz"]
)

llm = HuggingFaceEndpoint(
    repo_id = "tiiuae/falcon-7b-instruct",
    task = "text-generation"
)
#model = ChatHuggingFace(llm=llm)
model = ChatOpenAI(
    model="gpt-3.5-turbo"
)
parser = StrOutputParser()

text_gen_chain = prompt1 | model | parser

parallel_chain = RunnableParallel( {
    "notes" : prompt2 | model | parser,
    "quiz" : prompt3 | model | parser
})

merge_chain = prompt4 | model | parser

final_chain = text_gen_chain | parallel_chain # | merge_chain

print(final_chain.invoke({"topic":"Meditation"}))