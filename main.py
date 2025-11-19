from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent , AgentExecutor
from tools import search_tool
from pprint import pprint
import json


load_dotenv()

llmOpen = ChatOpenAI(model="gpt-4o-mini")
llmAnthropic = ChatAnthropic(model="claude-sonnet-4-5-20250929")

# llmAnthropic = ChatAnthropic(model="claude-sonnet-4-5-20250929")
# response = llmAnthropic.invoke("How well do you think I will do on my AI interview?")

# openAIResponse = llmOpen.invoke("Hello")

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

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
    llm=llmOpen,
    prompt=prompt,
    tools=tools
)




agent_executor = AgentExecutor(agent=agent, tools=[], verbose=True)
query = input("What can I help you research? ")
raw_response = agent_executor.invoke({"query": query})


def print_research(response: ResearchResponse) -> None:
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

try:
    structured_response = parser.parse(raw_response["output"])
    print_research(structured_response)

except Exception as e:
    print("Error parsing response", e, "Raw Response -" , raw_response)


