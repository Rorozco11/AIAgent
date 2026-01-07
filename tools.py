from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import StructuredTool
from datetime import datetime


search = DuckDuckGoSearchRun()
search_tool = StructuredTool.from_function(
    func=search.run,
    name="search",
    description="Search the web for information"
)