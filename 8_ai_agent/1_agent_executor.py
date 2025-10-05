from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from dotenv import load_dotenv

load_dotenv()

@tool
def multiply_nums(a: str) -> int:
    """Multiply given string with 3."""
    return int(a) * 3

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools([multiply_nums])



agent_executor = initialize_agent(
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    llm=llm_with_tools,
    tools=[multiply_nums],
    verbose=True,
    handle_parsing_errors=True
)

result = agent_executor.invoke("multiply number 6")
