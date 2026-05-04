from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, ToolMessage
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated

from tools import ingest_pdf_tool, search_tool, keyword_search_tool, web_search, news_tool, newsletter_tool, get_weather, finance_tool, medical_tool
import sqlite3
from datetime import datetime

# SQLite setup
conn = sqlite3.connect("chat_history.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS chats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    role TEXT,
    content TEXT,
    tool_used TEXT,
    timestamp TEXT
)
""")
conn.commit()


def save_message(role: str, content: str, tool_used: str = None):
    cursor.execute(
        "INSERT INTO chats (role, content, tool_used, timestamp) VALUES (?, ?, ?, ?)",
        (role, content, tool_used, datetime.utcnow().isoformat())
    )
    conn.commit()

llm = ChatOpenAI(
    base_url="http://103.172.92.233:8006/v1",
    api_key="EMPTY",
    model="cyankiwi/Qwen3.5-4B-AWQ-4bit"
)

tools = [ingest_pdf_tool, search_tool, keyword_search_tool, web_search, news_tool, newsletter_tool, get_weather,finance_tool, medical_tool]
tool_map = {t.name: t for t in tools}

llm_with_tools = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    steps: int


SYSTEM_PROMPT = SystemMessage(content="""
You are an autonomous multi-tool reasoning agent.

Your goal is to produce a COMPLETE and ACCURATE answer by:
- Selecting the right tools
- Calling tools multiple times if needed
- Evaluating tool outputs critically
- Continuing until the answer is sufficient

---

### CORE WORKFLOW:

At every step you MUST:

1. ANALYZE:
   What information is needed?

2. DECIDE:
   Do I already have enough information?
   - YES → produce final answer
   - NO → call one or more tools

3. EVALUATE TOOL OUTPUT:
   After each tool call, ask:
   - Is this information complete?
   - Is anything missing?
   - Do I need another source?

4. CONTINUE IF NEEDED:
   If information is incomplete, unclear, or insufficient:
   → Call another tool
   → Or call the same tool with a better query

---

### IMPORTANT BEHAVIOR:

- You are NOT limited to one tool call
- You SHOULD use multiple tools when needed
- You MUST NOT stop early with partial information

---

### TOOL SELECTION RULES:

- Document queries → search_tool → fallback keyword_search_tool
- AI/Tech → newsletter_tool ONLY
- General news → news_tool
- Weather → get_weather
- Finance → finance_tool ONLY
- Medical → medical_tool ONLY
- General → web_search

---

### SELF-CORRECTION RULE:

If a tool output is:
- incomplete
- vague
- missing key details

You MUST explicitly continue by calling another tool.

---

### COMPLETION RULE:

You may ONLY stop when:
- The answer is complete
- All aspects of the query are addressed

STRICT OUTPUT RULES (MANDATORY):

- DO NOT output your reasoning, thoughts, analysis, or decision process
- DO NOT explain tool selection
- DO NOT include intermediate steps

You MUST ONLY output:

FINAL ANSWER:
<clean user-facing answer only>

Tools used: <comma-separated tool names>
""")


def call_model(state: AgentState):
    messages = state["messages"]
    steps = state.get("steps", 0)
    response = llm_with_tools.invoke(
        [SYSTEM_PROMPT] + messages
    )

    return {
        "messages": [response],
        "steps": steps + 1
    }

def call_tools(state: AgentState):
    messages = state["messages"]
    last_msg = messages[-1]

    tool_outputs = []

    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:

        for call in last_msg.tool_calls:
            tool = tool_map[call["name"]]
            result = tool.invoke(call["args"])

            save_message(
                role="tool",
                content=str(result),
                tool_used=call["name"]
            )

            tool_outputs.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=call["id"]
                )
            )

    return {"messages": tool_outputs}

MAX_STEPS=5
def should_continue(state: AgentState):
    last = state["messages"][-1]
    steps = state.get("steps", 0)


    if steps >= MAX_STEPS:
        return END
    # If tool calls exist → execute tools
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"

    # If model explicitly finished → stop
    if isinstance(last.content, str) and "FINAL ANSWER:" in last.content:
        return END

    # Otherwise → let model think again (self-correction loop)
    return "agent"



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
        "agent": "agent",
        END: END
    }
)

graph.add_edge("tools", "agent")

agent = graph.compile()

chat_history = []
import re

def clean_response(text: str):
    match = re.search(r"FINAL ANSWER:\s*(.*)", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


print("\n Agent Ready (type 'exit' to quit)\n")

while True:
    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit"]:
        break
    MAX_MEMORY = 4

    chat_history.append(HumanMessage(content=user_input))

    trimmed_history = chat_history[-MAX_MEMORY:]

    result = agent.invoke({
    "messages": trimmed_history
    })
  
    response = result["messages"][-1]
    cleaned_output = clean_response(response.content)

    print(f"\n AI: {cleaned_output}\n")
    save_message("assistant", response.content)
    chat_history.append(HumanMessage(content=user_input))
 
    save_message("user", user_input)