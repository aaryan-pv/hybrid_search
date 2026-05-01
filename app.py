from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, ToolMessage
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated

from tools import ingest_pdf_tool, search_tool, keyword_search_tool

llm = ChatOpenAI(
    base_url="http://103.172.92.233:8006/v1",
    api_key="EMPTY",
    model="cyankiwi/Qwen3.5-4B-AWQ-4bit"
)

tools = [ingest_pdf_tool, search_tool, keyword_search_tool]
tool_map = {t.name: t for t in tools}

llm_with_tools = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


SYSTEM_PROMPT = SystemMessage(content="""
You are a document intelligence agent.

TOOLS:
- ingest_pdf_tool → load PDFs
- search_tool → BEST hybrid retrieval (use this first)
- keyword_search_tool → exact keyword match

RULES:
- ALWAYS use search_tool for document questions
- Use keyword_search_tool only if search_tool fails
- Never guess answers
""")


def call_model(state: AgentState):
    messages = state["messages"]

    response = llm_with_tools.invoke(
        [SYSTEM_PROMPT] + messages
    )

    return {"messages": [response]}

def call_tools(state: AgentState):
    messages = state["messages"]
    last_msg = messages[-1]

    tool_outputs = []

    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:

        for call in last_msg.tool_calls:
            tool = tool_map[call["name"]]

            result = tool.invoke(call["args"])

            tool_outputs.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=call["id"]
                )
            )

    return {"messages": tool_outputs}


def should_continue(state: AgentState):
    last = state["messages"][-1]

    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"

    return END

graph = StateGraph(AgentState)

# nodes
graph.add_node("agent", call_model)
graph.add_node("tools", call_tools)

# entry
graph.set_entry_point("agent")

# routing logic
graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        END: END
    }
)

graph.add_edge("tools", "agent")

agent = graph.compile()

chat_history = []

print("\n📚 Document Agent Ready (type 'exit' to quit)\n")

while True:
    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit"]:
        break

    chat_history.append(HumanMessage(content=user_input))

    result = agent.invoke({
        "messages": chat_history
    })

    response = result["messages"][-1]

    print(f"\n🤖 AI: {response.content}\n")

    chat_history.append(response)