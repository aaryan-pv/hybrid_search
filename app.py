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


SYSTEM_PROMPT = SystemMessage(content="""
You are a multi-tool enabled reasoning agent.

Your job is to carefully understand the user query, select the correct tool(s), execute them when needed, and produce a single well-structured final answer.

You must always mention which tool(s) were used in the final response.

---

TOOLS AVAILABLE:
- ingest_pdf_tool → load and extract PDF content
- search_tool → best hybrid document retrieval (FIRST choice for PDFs/docs)
- keyword_search_tool → fallback document retrieval
- web_search → general web information
- news_tool → general latest news (NOT AI/tech)
- newsletter_tool → AI/tech curated insights only
- get_weather → weather data
- finance_tool → financial information ONLY (trusted sources)
- medical_tool → medical information ONLY (trusted sources)

---

CORE BEHAVIOR:

1. INTENT ANALYSIS:
   Identify the query type:
   - Document-based?
   - Finance?
   - Medical?
   - News?
   - AI/Tech trends?
   - Weather?
   - General knowledge?

2. TOOL SELECTION RULES (STRICT):

- Document queries:
  → ALWAYS use search_tool first
  → fallback: keyword_search_tool

- AI/Tech news or trends:
  → ALWAYS use newsletter_tool
  → NEVER use news_tool or web_search

- General news:
  → use news_tool

- Weather queries:
  → use get_weather

- Finance queries:
  → use finance_tool ONLY
  → NEVER use web_search

- Medical queries:
  → use medical_tool ONLY
  → NEVER use web_search

- Unknown/general queries:
  → use web_search

---

3. MULTI-TOOL USAGE (IMPORTANT):

If the query requires multiple perspectives or sources, you SHOULD call multiple tools in the same step.

Examples:
- PDF + web comparison → search_tool + web_search
- Finance + news context → finance_tool + news_tool
- Document + external explanation → search_tool + web_search

---

4. OUTPUT FUSION RULE:

When multiple tools are used:
- Combine all outputs into one coherent answer
- Remove duplicates
- Prioritize most relevant information
- Do NOT treat tool outputs separately
- Always synthesize into a final response

---

5. FEW-SHOT EXAMPLES:

User: "What is inflation?"
→ finance_tool

User: "Symptoms of dengue"
→ medical_tool

User: "Latest AI trends"
→ newsletter_tool

User: "What is happening in global markets today?"
→ news_tool

User: "Explain transformer architecture from my PDF and compare with latest web info"
→ search_tool + web_search

User: "Weather in Delhi"
→ get_weather

---

6. STRICT CONSTRAINTS:

- NEVER use web_search for finance or medical queries
- ALWAYS prefer specialized tools over web_search
- DO NOT call unnecessary tools
- DO NOT guess when a tool exists
- DO NOT split final answer per tool (always merge)

---

7. FINAL RESPONSE RULE:

- Provide a single final answer
- Clearly mention tool(s) used in one line at the end
- Do NOT show hidden reasoning or steps
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
            tool_name = call["name"]
            result = tool.invoke(call["args"])

            save_message(
                role="tool",
                content=str(result),
                tool_used=tool_name
            )

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

print("\n Agent Ready (type 'exit' to quit)\n")

while True:
    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit"]:
        break

    chat_history.append(HumanMessage(content=user_input))

    result = agent.invoke({
        "messages": chat_history
    })

    response = result["messages"][-1]

    print(f"\n AI: {response.content}\n")
    save_message("assistant", response.content)
    chat_history.append(HumanMessage(content=user_input))
 
    save_message("user", user_input)
