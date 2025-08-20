# huggingface APIs - Second time
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

class Review(BaseModel):
    keypoints : list[str] = Field(description="give the keypoints discussed in the review")
    reviewer : str = Field(description= "provide the name of the reviewer")     
    sentiment : Literal['positive', 'negative'] = Field(description="provide the sentiment of the review")
    date : datetime = Field(description="provide the date of the review")


review = """I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful
                                 
Review by Nitish Singh
"""

template = """Review : {review}
{format_instructions}
"""

parser = PydanticOutputParser(pydantic_object=Review)

prompt_template = PromptTemplate(
    template=template,
    input_variables=["review"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

llm = HuggingFaceEndpoint(
    repo_id='google/gemma-2-2b-it',
    task='text-generation'
)
model = ChatHuggingFace(llm=llm)

result = model.invoke(prompt_template.format(review=review))
print(result.content)

exit(0)


# use huggingface APIs
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


