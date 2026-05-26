from dotenv import load_dotenv
load_dotenv(dotenv_path = ".env")

import os
from typing_extensions import TypedDict, Annotated
from langchain.messages import SystemMessage, AIMessage, HumanMessage, ToolMessage
from typing import Literal, Any, Sequence, List 
from langgraph.graph import StateGraph, START, END

from langchain_groq import ChatGroq

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENROUTER_API = os.getenv("OPENROUTER_API")
LANGSMITH_API = os.getenv("LANGSMITH_API")

# """
if GROQ_API_KEY and OPENROUTER_API and LANGSMITH_API:
    print("Successful import")
    # """

llm = ChatGroq(
    api_key= GROQ_API_KEY,
    model_name = "llama-3.3-70b-versatile",
    temperature=0.7
)
print(llm.invoke("Hi"))


#Graph State
class State(TypedDict):
    user_query:str
    intent: str
    recommendation: str
    feedback:str

#Node 1
def router_agent():
    pass


