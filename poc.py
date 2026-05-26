import tools
log_data = tools.collect_logs(path = r"data/monitoring_logs.txt")
collection = tools.create_or_get_cromadb(logs = log_data)

from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")

import os, json
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

class State(TypedDict):
    history: list  
    query: str           
    answer: str        
    use_tool: bool      


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="openai/gpt-oss-20b",
    temperature=0.7
)


class Tools:
    def __init__(self, collection):
        self.collection = collection

    def search_query(self, query: str, n_results: int = 3):
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return results
    
tools = Tools(collection)

class Decision(BaseModel):
    use_tool: bool = Field(description="Whether to call the tool")
    answer: str = Field(description="Direct answer if applicable")

parser = JsonOutputParser(pydantic_object=Decision)

prompt = PromptTemplate(
    template="""
    Conversation history: {history}
    User query: {query}
    
    Decide: Should I answer directly or use the tool?
    Use tool when we don't know about query, user may ask about system or logs, tool will give logs of the system.
    use Tool when user reports any system related issue.
    Respond strictly in JSON with keys: no other formate, exact below one.
    {{
      "use_tool": true/false,
      "answer": "direct answer if applicable"
    }}
    """,
    input_variables=["history", "query"],
    output_parser=parser,
)

chain = prompt | llm | parser

def chat_node(state: State) -> State:
    print("\n[CHAT NODE]")
    history_text = "\n".join([str(msg) for msg in state["history"]])
    result: Decision = chain.invoke({"history": history_text, "query": state["query"]})
    print(f"[result]: {result}")

    state["use_tool"] = result["use_tool"]
    state["answer"] = result["answer"]
    state["history"].append(AIMessage(content=state["answer"]))

    return state



def tool_node(state: State) -> State:
    print("\n[TOOL NODE]")
    results = tools.search_query(query=state["query"], n_results=3)
    print(f"[Tool Result]: {json.dumps(results, indent = 2)}")
    tool_msg = ToolMessage(content=str(results), tool_call_id="search_query")
    state["history"].append(tool_msg)

    prompt = PromptTemplate(
        template="""
        User query: {query}
        Tool results: {results}

        Write a helpful natural language answer. don't write much from your side, minimal answers, answer based on chathistory or 
        """,
        input_variables=["query", "results"]
    )

    chain_tool = prompt | llm
    response = chain_tool.invoke({"query": state["query"], "results": results})
    state["answer"] = response.content
    state["history"].append(AIMessage(content=state["answer"]))
    return state

workflow = StateGraph(State)
workflow.add_node("chat_node", chat_node)
workflow.add_node("tool_node", tool_node)

workflow.add_edge(START, "chat_node")
workflow.add_conditional_edges("chat_node", lambda s: "tool_node" if s["use_tool"] else END)
workflow.add_edge("tool_node", END)

agent = workflow.compile()


def run():
    state: State = {"history": [], "query": "", "answer": "", "use_tool": False}
    print("Starting")
    while True:
        query = input("\nYou: ")
        print(f"\n\nYou: {query}")
        if query.lower() in ["quit", "exit"]:
            print(f"[State]: {json.dumps(state, indent = 2)}")
            break

        # Save HumanMessage
        state["query"] = query
        state["history"].append(HumanMessage(content=query))

        result = agent.invoke(state)

        print("Assistant:", result["answer"])
        state.update(result)


if __name__ == "__main__":
    run()