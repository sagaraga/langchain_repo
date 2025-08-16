from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate

# defining prompts - dynamic
prompt_template = PromptTemplate.from_template(
                    "You are a good {profession}. Tell me about {topic}"
)

profession = "teacher"
topic = "solar system"

final_prompt = prompt_template.invoke({"profession":profession, "topic":topic})


# defining models - opensource models
llm = HuggingFaceEndpoint(
       #repo_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
       #repo_id = "upstage/SOLAR-10.7B-Instruct-v1.0",
       repo_id = "HuggingFaceH4/zephyr-7b-beta",
       task="text-generation" )
model = ChatHuggingFace(llm=llm)

result = model.invoke(final_prompt)
print(result)
