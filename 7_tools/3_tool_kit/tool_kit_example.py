from langchain_core.tools import tool

@tool
def multiply_nums(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

@tool
def add_nums(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

class MathToolKit:
    def get_tools(self):
        return [multiply_nums, add_nums]
    
math_tool_kit = MathToolKit()

for tool in math_tool_kit.get_tools():
    print(f"Tool name: {tool.name} ==> Tool description: {tool.description}")