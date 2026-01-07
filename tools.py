from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from datetime import datetime


# DuckDuckGoSearchRun is already a BaseTool, so we can use it directly
search_tool = DuckDuckGoSearchRun()
# Optionally customize the name and description
search_tool.name = "search"
search_tool.description = "Search the web for information"