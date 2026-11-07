
from langchain.messages import HumanMessage
from langfuse import get_client
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from bacen_agent.agent.nodes import ToolAgent
from bacen_agent.agent.state import ToolAgentState
from bacen_agent.config import get_settings
from bacen_agent.embeddings.embedder import EmbedderModel
from bacen_agent.llm.chat_anthropic import ChatAnthropicProvider
from bacen_agent.observability import make_langfuse
from bacen_agent.opensearch import make_client
from bacen_agent.rag.pipeline import RagResponse
from bacen_agent.retrieval.reranker import Reranker
from bacen_agent.retrieval.retriever import Retriever, search_methods

def build_tool_graph(agent: ToolAgent):
    graph = StateGraph(ToolAgentState)
    graph.add_node("agent", agent.agent)
    graph.add_node("tools", ToolNode(agent.tools))
    graph.add_node("finalize", agent.finalize)
    
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", agent.route, ["tools", "finalize"])
    graph.add_edge("tools", "agent")
    graph.add_edge("finalize", END)
    
    return graph.compile()
    
class AgentGraph:
    def __init__(self, agent: ToolAgent):
        self.app = build_tool_graph(agent)
        self.k = agent.k
        self.retrieval_method = agent.retrieval_method
        self.runner = "agent_v1_tools"
        
    def answer(self, question: str) -> RagResponse:
        out = self.app.invoke({"question": question, "messages": [HumanMessage(question)]})
        return out["response"]
    
if __name__ == "__main__":
    settings = get_settings()
    make_langfuse(settings)
    
    retriever = Retriever(make_client(settings), EmbedderModel(settings), settings.index_name, Reranker(settings))
    search = search_methods(retriever, settings)[settings.retrieval_method]
    llm = ChatAnthropicProvider(settings)
    
    agent = ToolAgent(llm.model, search, settings.retrieval_method)
    graph = AgentGraph(agent)
    
    for q in ["Caí num golpe no Pix, consigo o dinheiro de volta?",
              "Qual a cotação do dólar hoje?", 
              "Recebi um Pix e o banco bloqueou o dinheiro por suspeita de fraude. Por quanto tempo ele pode ficar bloqueado e o que acontece depois?",
              ]:
        out = graph.answer(q)
        print(out)

    get_client().flush()
