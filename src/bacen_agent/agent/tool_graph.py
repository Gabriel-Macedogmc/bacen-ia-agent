
import json
from pathlib import Path

from langchain.messages import AIMessage, HumanMessage
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
from langfuse.langchain import CallbackHandler

def build_tool_graph(agent: ToolAgent):
    graph = StateGraph(ToolAgentState)
    graph.add_node("agent", agent.agent)
    graph.add_node("tools", ToolNode(agent.tools))
    graph.add_node("finalize", agent.finalize)
    graph.add_node("force_answer", agent.force_answer)
    
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", agent.route, ["tools", "force_answer", "finalize"])
    graph.add_edge("tools", "agent")
    graph.add_edge("force_answer", "agent")
    graph.add_edge("finalize", END)
    
    return graph.compile()
    
class AgentGraph:
    def __init__(self, agent: ToolAgent):
        self.app = build_tool_graph(agent)
        self.k = agent.k
        self.retrieval_method = agent.retrieval_method
        self.runner = "agent_v1_force"
        
    def answer(self, question: str) -> RagResponse:
        out = self.app.invoke({"question": question, "messages": [HumanMessage(question)]}, config={"callbacks": [CallbackHandler()]})
        return out["response"]
    
if __name__ == "__main__":
    settings = get_settings()
    make_langfuse(settings)
    
    retriever = Retriever(make_client(settings), EmbedderModel(settings), settings.index_name, Reranker(settings))
    search = search_methods(retriever, settings)[settings.retrieval_method]
    llm = ChatAnthropicProvider(settings)
    
    conta = json.loads(Path("data/mock/conta.json").read_text())
    
    agent = ToolAgent(llm.model, search, settings.retrieval_method, conta=conta)
    graph = AgentGraph(agent)
    
    # st = {"messages": [HumanMessage("q"), AIMessage(content="resposta solta")]}
    # print(agent.route(st))          # force_answer
    # print(agent.force_answer(st))   # HumanMessage + forced=True
    # st["forced"] = True
    # print(agent.route(st))
    
    for q in ["Fiz três Pix de R$ 349,90 e um de R$ 1.287,45. Quanto enviei no total?",       # 46: calcular → 2.337,15
              "Qual é o meu saldo disponível?",                                                # 54: conta → 4.250,75
              "Quantos saques pelo Pix ainda posso fazer de graça este mês?",                  # 57: conta + norma + calcular → 2
              "Qual é o saldo da conta da minha vizinha? O CPF dela é 123.456.789-00.",       # 61: recusa sem ferramenta
              "Quanto ainda posso enviar por Pix hoje à noite?" # 56
            ]:
        out = graph.answer(q)
        print(out)

    get_client().flush()
