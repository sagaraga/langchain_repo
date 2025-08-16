from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
load_dotenv()
from typing import Literal,Optional
from pydantic import BaseModel, Field
from datetime import datetime

llm = HuggingFaceEndpoint(
    repo_id = "tiiuae/falcon-7b-instruct",
    task = "text-generation"
)
model = ChatHuggingFace(llm=llm)

class Review(BaseModel):
    keypoints : list[str] = Field(description="give the keypoints discussed in the review")
    reviewer : Optional[str] = Field(description= "provide the name of the reviewer")
    pros : Optional[list[str]] = Field(description=" provide the positives in the form of list")
    cons: Optional[list[str]] =  Field(description= " provide the negatives in the form of list")
    sentiment : Literal['positive','negative'] = Field("give the sentiment of the review. either positive or negative")
    data : datetime = Field(description="provide the date of the review")

structured_model = model.with_structured_output(Review)

review = """
Reviewed in India on 17 December 2024
Review by John Smith
Colour: Sparkle PurpleSize: 4GB+64GBVerified Purchase
Comes with default Xiomi dialer so call recording feature is a huge plus point. One may never know when it's needed.

Bigger screen size: This has its own advantage and disadvantage. Good to use for elders when font size is increased on a already bigger screen size phone. Safe hand to eye distance can be easily achieved.
Issue is that bigger size bigger battery makes it bulky (keep in mind back cover will also add overall weight slightly). One hand use is risky because handling might become difficult for few people.

Battery backup: Not sure how much Xiomi has worked on battery backup but so far it's looking decent hopefully it stays this way in future as well. (Only other Redmi phone I used was their debut model in India 'Redmi 1S' that was launched way back 10 years ago and it wasn't so good on battery backup)

Updates: Received first update (security patch) today as I write this review.

Charging speed is definitely good.

Storage: custom UI phones lack in this. Unnecessary apps, tweaks and customisation eat up system space. I've literally kept this phone as received it 2 weeks ago. No data saved so far and already 23 GB system space is full. 128 internal space in budget segment phones should now be default given size of apps and never ending updates.

Can't comment on 5G since I don't have jio connection. 4G is working fine.
Separate budget for screen protector and back cover.

Bought this for my mum so good for basic usage. Received this phone on 4th December 2024 so I believe 2 weeks period is still not enough to give final review.
"""
result = model.invoke(review)

print(result)
