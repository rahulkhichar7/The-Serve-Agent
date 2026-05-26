from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

# ----- Logging Setup ----
import logging
logging.basicConfig(
    filename = 'atg_agent_logs.log',
    filemode = "a", #append
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level = logging.INFO
)

logger = logging.getLogger("ATG workflow")

# ----- STATE -----
class State(TypedDict):
    user_query: str
    intent: str
    recommendation: str
    feedback: str
    requirement_clear: bool
    single_source: bool


# ----- NODES -----
def chat_node(state: State) -> State:
    logger.info("[CHAT NODE] User Query: %s", state["user_query"])

    state["intent"] = "diagnose_issue"
    
    if "simple" in state["user_query"]:
        state["data_ready"] = True
    else:
        state["data_ready"] = False
    return state


def data_decider(state: State) -> State:
    logger.info("[DATA DECIDER] Deciding what data is required...")
    
    # Decide if single or multi source needed
    if "single" in state["user_query"]:
        state["single_source"] = True
    else:
        state["single_source"] = False

    return state


def data_fetcher(state: State) -> State:
    logger.info("[DATA FETCHER] Fetching mock data...")
    return state


def timeline_builder(state: State) -> State:
    logger.info("[TIMELINE BUILDER] Building correlation timeline...")
    return state


def diagnosis_agent(state: State) -> State:
    logger.info("[DIAGNOSIS AGENT] Diagnosing issue...")
    return state


def recommendation_node(state: State) -> State:
    state["recommendation"] = "Restart the service"
    logger.info("[RECOMMENDATION] Recommendation: %s", state["recommendation"])
    return state

# ----- ROUTERS -----
def route_from_chat(state: State) -> State:
    if state["requirement_clear"]:
        return "data_fetcher"
    return "data_decider"

def route_from_fetcher(state: State) -> State:
    if state["single_source"]:
        return "chat_node"
    return "timeline_builder"


# ----- WORKFLOW -----
workflow = StateGraph(State)

# Add Nodes
workflow.add_node("chat_node", chat_node)
workflow.add_node("data_decider", data_decider)
workflow.add_node("data_fetcher", data_fetcher)
workflow.add_node("timeline_builder", timeline_builder)
workflow.add_node("diagnosis_agent", diagnosis_agent)
workflow.add_node("recommendation", recommendation_node)

# Define edges
workflow.add_edge(START, "chat_node")
workflow.add_conditional_edges("chat_node", route_from_chat)
workflow.add_edge("data_decider", "data_fetcher")
workflow.add_conditional_edges("data_fetcher", route_from_fetcher)
workflow.add_edge("timeline_builder", "diagnosis_agent")
workflow.add_edge("diagnosis_agent", "recommendation")
workflow.add_edge("recommendation", END)


# Compile graph
atg_agent = workflow.compile()

"""
# Test
result = atg_agent.invoke({
    "user_query": "System crashed due to memory spike",
    "intent": "",
    "recommendation": "",
    "feedback": "",

    "requirement_clear": False,
    "single_source": False

})

print("\nFINAL STATE:\n", result)
"""



"""
print(atg_agent.get_graph().draw_mermaid())

from langchain_core.runnables.graph import MermaidDrawMethod

png = atg_agent.get_graph().draw_mermaid_png(
    draw_method=MermaidDrawMethod.PYPPETEER
)

with open("graph.png", "wb") as f:
    f.write(png)
    """
