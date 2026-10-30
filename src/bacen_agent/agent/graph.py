from typing import Protocol

from langfuse import get_client
from langgraph.graph import END, START, StateGraph

from bacen_agent.agent.nodes import RagAgent
from bacen_agent.agent.state import AgentState
from bacen_agent.config import Settings, get_settings
from bacen_agent.embeddings.embedder import EmbedderModel
from bacen_agent.llm.anthropic import AnthropicProvider
from bacen_agent.observability import make_langfuse
from bacen_agent.opensearch import make_client
from bacen_agent.rag.pipeline import RagResponse
from bacen_agent.retrieval.reranker import Reranker
from bacen_agent.retrieval.retriever import Retriever, search_methods

class Answerer(Protocol):
    k: int
    retrieval_method: str
    runner: str
    def answer(self, question: str) -> RagResponse: ...

def build_graph(agent: RagAgent, settings: Settings):
    graph = StateGraph(AgentState)
    graph.add_node("retrieve", agent.retrieve)
    graph.add_node("generate", agent.generate)
    graph.add_node("validate", agent.validate)
    graph.add_node("rewrite", agent.rewrite)
    
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "validate")
    
    def route_after_validate(state: AgentState) -> str:
        if state["response"].found or state.get("attempts", 0) >= settings.max_rewrites:
            return END
        return "rewrite"
    
    graph.add_conditional_edges("validate", route_after_validate, ["rewrite", END])
    graph.add_edge("rewrite", "retrieve")
    
    return graph.compile()

class RagGraph:
    def __init__(self, agent: RagAgent, settings: Settings):
        self.app = build_graph(agent, settings)
        self.k = agent.k
        self.retrieval_method = agent.retrieval_method
        self.runner = "graph_v1"
        
    def answer(self, question: str) -> RagResponse:
        return self.app.invoke({"question": question, "query": question, "queries": [question]})["response"]


if __name__ == "__main__":
    settings = get_settings()
    make_langfuse(settings)
        
    retriever = Retriever(make_client(settings), EmbedderModel(settings), settings.index_name, Reranker(settings))
    provider = AnthropicProvider(settings)
            
    methods = search_methods(retriever, settings)
    
    agent = RagAgent(methods[settings.retrieval_method], settings.retrieval_method, provider)
    
    ragGraph = RagGraph(agent, settings)
    
    response = ragGraph.answer(question="O banco pode me cobrar para fazer um Pix?")
    
    ragGraph.app.get_graph().draw_mermaid_png(output_file_path="docs/agent_graph.png")
    
    get_client().flush()
    