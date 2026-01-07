from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents.agent import AgentExecutor
from langchain.agents.tool_calling_agent.base import create_tool_calling_agent
from tools import search_tool
from pprint import pprint
import json


load_dotenv()

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

def get_agent_executor():
    """Initialize and return the agent executor."""
    llmOpen = ChatOpenAI(model="gpt-4o-mini")
    
    parser = PydanticOutputParser(pydantic_object=ResearchResponse)
    
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a research assistant that will help generate a research paper.
                Answer the user query and use necessary tools.
                Wrap the output in this format and provide no other text\n{format_instructions}
                """,
            ),
            ("placeholder", "{chat_history}"),
            ("human", "{query}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    ).partial(format_instructions=parser.get_format_instructions())
    
    tools = [search_tool]
    
    agent = create_tool_calling_agent(
        llmOpen,
        tools,
        prompt
    )
    
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return agent_executor, parser

def execute_research_query(query: str):
    """Execute a research query and return the parsed response."""
    agent_executor, parser = get_agent_executor()
    raw_response = agent_executor.invoke({"query": query})
    
    try:
        structured_response = parser.parse(raw_response["output"])
        return structured_response, None
    except Exception as e:
        return None, {"error": str(e), "raw_response": raw_response}

def print_research(response: ResearchResponse) -> None:
    """Print research response in formatted text (for CLI use)."""
    print("\n" + "=" * 60)
    print(f"TOPIC: {response.topic}")
    print("=" * 60 + "\n")

    print("SUMMARY:\n")
    print(response.summary)
    print("\n")

    print("SOURCES:")
    for src in response.sources:
        print(f"  • {src}")
    print("\n")

    print("TOOLS USED:")
    for tool in response.tools_used:
        print(f"  • {tool}")
    print("\n" + "=" * 60 + "\n")

# CLI execution (only runs if this file is executed directly)
if __name__ == "__main__":
    agent_executor, parser = get_agent_executor()
    query = input("What can I help you research? ")
    raw_response = agent_executor.invoke({"query": query})
    
    try:
        structured_response = parser.parse(raw_response["output"])
        print_research(structured_response)
    except Exception as e:
        print("Error parsing response", e, "Raw Response -", raw_response)


