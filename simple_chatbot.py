from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

llm = HuggingFaceEndpoint(
       repo_id = "HuggingFaceH4/zephyr-7b-beta",
       task="text-generation" )
model = ChatHuggingFace(llm=llm)


chat_history = [SystemMessage(content="You are a good cricket player")]

while True:
    user_input = input("You : ")
    if user_input == "exit":
        break
    chat_history.append(HumanMessage(content=user_input))
    result = model.invoke(chat_history)
    print("AI  :", result.content)
    chat_history.append(AIMessage(content=result.content))

