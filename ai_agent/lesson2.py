import operator
from typing import Annotated, TypedDict, cast
from dotenv import load_dotenv
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages import AnyMessage, ToolCall, ToolMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langgraph.constants import END
from langgraph.graph import StateGraph
from rich import print

load_dotenv()

tool = TavilySearchResults(max_results=4)
print(tool.name)
print(tool.description)

class AgentState(TypedDict):
  messages: Annotated[list[AnyMessage], operator.add]

class Agent:
  def __init__(self, model, tools, system="") -> None:
    self.system = system
    graph = StateGraph(AgentState)
    graph.add_node("llm", self.call_openai)
    graph.add_node("action", self.take_action)
    graph.add_conditional_edges("llm", self.exists_action, {True: "action", False: END})
    graph.add_edge("action", "llm")
    graph.set_entry_point("llm")
    self.graph = graph.compile()
    self.tools = {t.name: t for t in tools}
    self.model = model.bind_tools(tools)

  def exists_action(self, state: AgentState):
    result = state['messages'][-1]
    if isinstance(result, AIMessage):
      return len(result.tool_calls) > 0
    return False

  def call_openai(self, state: AgentState):
    messages = state['messages']
    if self.system:
      messages = [SystemMessage(content = self.system)] + messages
    message = self.model.invoke(messages)
    return {'messages': [message]}

  def take_action(self, state: AgentState):
    if not isinstance(state['messages'][-1], AIMessage):
      return {'messages': [SystemMessage(content="No tool calls found")]}

    ai_message: AIMessage = state['messages'][-1] # type: ignore
    tool_calls: list[ToolCall] = ai_message.tool_calls
    results = []
    for t in tool_calls:
      print(f"Calling: {t}")
      if t['name'] not in self.tools:
        print("\n .....bad tool name.....")
        result = "bad tool name, retry"
      else:
        result = self.tools[t['name']].invoke(t['args'])
      results.append(ToolMessage(tool_call_id=cast(str, t['id']), name=t['name'], content=str(result)))
    print("Back to the model!")
    return {'messages': results}

system_prompt = """You are a smart research assistant. Use the search engine to look up information. \
You are allowed to make multiple calls (either together or in sequence). \
Only look up information when you are sure of what you want. \
If you need to look up some information before asking a follow up question, you are allowed to do that!
"""

# model_name = "gpt-3.5-turbo"
model_name = "gpt-4o"

model = ChatOpenAI(model=model_name)
abot = Agent(model, [tool], system=system_prompt)

# abot.graph.get_graph().draw_png("lesson2-graph.png")

messages = [HumanMessage(content="What is the weather in sf?")]
result = abot.graph.invoke({"messages": messages})

print("💡", result['messages'][-1].content)

print('🐬' * 80)

query = "Who won the super bowl in 2024? In what state is the winning team headquarters located? \
What is the GDP of that state? Answer each question."
messages = [HumanMessage(content=query)]

model = ChatOpenAI(model=model_name)
abot = Agent(model, [tool], system=system_prompt)
result = abot.graph.invoke({"messages": messages})

print("💡", result['messages'][-1].content)

#### Run outputs

#### gpt-3.5-turbo
# ❯ python ai_agent/lesson2.py
# tavily_search_results_json
# A search engine optimized for comprehensive, accurate, and trusted results. Useful for when you need to answer questions about current events. Input should be a search query.
# Calling: {'name': 'tavily_search_results_json', 'args': {'query': 'weather in San Francisco'}, 'id': 'call_nLj1TBWiGWW8qqTHmdGyyUsx'}
# Back to the model!
# 💡 The current weather in San Francisco is as follows:
# - Temperature: 53.0°F (11.7°C)
# - Condition: Clear
# - Wind: 15.7 mph (25.2 kph) from WNW
# - Humidity: 87%
# - Cloud Cover: 5%
# - Feels like: 48.6°F (9.2°C)
# - Pressure: 30.02 in
#
# If you need more detailed information or historical weather data, please let me know!
# 🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬
# Calling: {'name': 'tavily_search_results_json', 'args': {'query': 'Super Bowl 2024 winner'}, 'id': 'call_xIHJ9vsLn9INtcC0AdBV81HV'}
# Calling: {'name': 'tavily_search_results_json', 'args': {'query': 'headquarters of Super Bowl 2024 winner'}, 'id': 'call_VPJahG6bxS8CJCEypf9EsfKC'}
# Back to the model!
# Calling: {'name': 'tavily_search_results_json', 'args': {'query': 'GDP of Missouri'}, 'id': 'call_0EXLxlwDt9yI0tqxXsiTNl7T'}
# Back to the model!
# 💡 The GDP of Missouri was $336.63 billion in 2022.

#### gpt-4o
# ❯ python ai_agent/lesson2.py
# tavily_search_results_json
# A search engine optimized for comprehensive, accurate, and trusted results. Useful for when you need to answer questions about current events. Input should be a search query.
# Calling: {'name': 'tavily_search_results_json', 'args': {'query': 'current weather in San Francisco'}, 'id': 'call_3t0mYsisy92SXwi0iDrhRQUt'}
# Back to the model!
# 💡 The current weather in San Francisco is as follows:
#
# - **Temperature:** 11.7°C (53.0°F)
# - **Condition:** Clear
# - **Wind:** 15.7 mph (25.2 kph) from the WNW
# - **Humidity:** 87%
# - **Visibility:** 10 km (6 miles)
# - **Pressure:** 1016.0 mb (30.02 in)
# - **Feels Like:** 9.2°C (48.6°F)
# - **Dew Point:** 9.4°C (48.9°F)
# - **UV Index:** 1.0
#
# It's currently nighttime in San Francisco. The weather data was last updated at midnight local time.
# 🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬
# Calling: {'name': 'tavily_search_results_json', 'args': {'query': 'Who won the Super Bowl 2024'}, 'id': 'call_l7SunreYRiRcCCrAK2cCn2pB'}
# Back to the model!
# Calling: {'name': 'tavily_search_results_json', 'args': {'query': 'Kansas City Chiefs headquarters location'}, 'id': 'call_CVlTksy9aQ2UM2smIvZObdWZ'}
# Back to the model!
# Calling: {'name': 'tavily_search_results_json', 'args': {'query': 'GDP of Missouri 2024'}, 'id': 'call_xNtblB8ljRNfwJ729xeJBiwH'}
# Back to the model!
# 💡 1. **Who won the Super Bowl in 2024?**
#    - The Kansas City Chiefs won the Super Bowl in 2024, defeating the San Francisco 49ers 25-22 in overtime.
#
# 2. **In what state is the winning team's headquarters located?**
#    - The Kansas City Chiefs' headquarters are located in Kansas City, Missouri.
#
# 3. **What is the GDP of that state?**
#    - The GDP of Missouri in 2022 was approximately $336.63 billion. The latest specific figure for 2024 is not readily available, but it is reasonable to assume it would be in a similar range with potential adjustments for economic growth or inflation.
