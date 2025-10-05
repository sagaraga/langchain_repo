from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage
from dotenv import load_dotenv
load_dotenv()

@tool
def multiply_nums(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


llm = ChatOpenAI()

llm_with_tools = llm.bind_tools([multiply_nums])

# response1 = llm_with_tools.invoke("general news today")
# print(response1)
# print(f"content : {response1.content} and tool calls : {response1.tool_calls}")

#esponse2 = llm_with_tools.invoke("multiply 2 with 3")
# print(f"content : {response2.content} and tool calls : {response2.tool_calls}")
# print(f"args : {response2.tool_calls[0]['args']}")

# response3 = llm.invoke("give me the five important news today")
# print(f"content : {response3.content} and tool calls : {response3.tool_calls}")
# print(response3)

## Approach 1 : Manually selecting and calling with multiply_nums tool : Hardcoding
# final_result = multiply_nums.invoke(response2.tool_calls[0]['args'])
# print(f"multiplication result : {final_result}")


## Approach 2 : Automatically selecting and calling with multiply_nums tool : Dynamic
# user_msg = HumanMessage(content="multiply 2 with 3 and provide the result")
# final_result = llm_with_tools.invoke([user_msg])
# print(f"multiplication result : {final_result}")

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Step 1. Define tools
@tool
def add_numbers(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

@tool
def greet(name: str) -> str:
    """Say hello to a person."""
    return f"Hello, {name}!"

# Step 2. Create LLM and bind tools
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools([add_numbers, greet])

# Step 3. Give an instruction requiring both tools
user_msg = HumanMessage(content="Add 40 and 60, then greet me with the result.")

# Step 4. Invoke LLM
response = llm_with_tools.invoke([user_msg])

print(multiply_nums.invoke(response.tool_calls[0]))