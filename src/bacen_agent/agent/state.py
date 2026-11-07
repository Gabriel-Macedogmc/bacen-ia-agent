import operator
from typing import Annotated, TypedDict

from langchain.messages import AnyMessage
from langgraph.graph import add_messages

from bacen_agent.rag.pipeline import RagResponse
from bacen_agent.rag.schemas import Answer
from bacen_agent.retrieval.schemas import SearchResult


class AgentState(TypedDict, total=False):
    question: str
    chunks: list[SearchResult]
    answer: Answer | None
    response: RagResponse
    query: str
    queries: Annotated[list[str], operator.add]
    attempts: int
    input_tokens: Annotated[int, operator.add]
    output_tokens: Annotated[int, operator.add]
    
    
class ToolAgentState(TypedDict, total=False):
    question: str
    messages: Annotated[list[AnyMessage], add_messages]
    response: RagResponse