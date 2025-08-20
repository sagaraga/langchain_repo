from langchain_community.tools import DuckDuckGoSearchRun

search_tools = DuckDuckGoSearchRun()

response = search_tools.invoke("india team for Asia cup 2025")

#print(response)

print(f"tool name : {search_tools.name}")
print(f"tool description : {search_tools.description}")
print(f"tool args schema : {search_tools.args_schema.model_json_schema}")
print(f"tool args : {search_tools.args}")