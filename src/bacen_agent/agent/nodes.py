from langchain.chat_models import BaseChatModel
from langchain.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage, UsageMetadata
from langfuse import observe
from pydantic import ValidationError

from bacen_agent.agent.state import AgentState, ToolAgentState
from bacen_agent.agent.tools import make_buscar_normas
from bacen_agent.llm.provider import LLMProvider, LLMUsage
from bacen_agent.rag.pipeline import RagResponse
from bacen_agent.rag.prompts import AGENT_PROMPT, SYSTEM_PROMPT, SYSTEM_REWRITE_PROMPT, build_rewrite_message, build_user_message
from bacen_agent.rag.schemas import Answer, Rewrite
from bacen_agent.retrieval.retriever import SearchFn
from bacen_agent.retrieval.schemas import SearchResult

class RagAgent:
    def __init__(self, search: SearchFn, retrieval_method: str, llm: LLMProvider, k: int = 5):
        self.search = search
        self.retrieval_method = retrieval_method
        self.llm = llm
        self.k = k
        
    @observe(name="node:retrieve", as_type="chain")
    def retrieve(self, state: AgentState) -> dict:
        return {"chunks": self.search(state.get("query", state["question"]), self.k)}
    
    @observe(name="node:generate", as_type="chain") 
    def generate(self, state: AgentState) -> dict:
        question = state["question"]
        chunks = state["chunks"]
        
        answer, usage = self.llm.generate(SYSTEM_PROMPT, build_user_message(question, chunks), Answer)
         
        return {"answer": answer, "input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens}
    
    @observe(name="node:validate", as_type="chain")
    def validate(self, state: AgentState) -> dict:
        chunks, answer = state["chunks"], state["answer"]
        usage = LLMUsage(input_tokens=state["input_tokens"], output_tokens=state["output_tokens"])
        
        valid_ids = {c.chunk_id for c in chunks}
        
        if (answer is None 
                or not answer.found 
                or not answer.citations 
                or not set(answer.citations) <= valid_ids):
            return {"response": RagResponse.not_found(chunks, usage)}
        
        citations = list(dict.fromkeys(answer.citations))
        
        return {"response": RagResponse(
                                found=answer.found,
                                sources=[c for c in chunks if c.chunk_id in citations],
                                retrieved=chunks,
                                answer=answer.answer,
                                input_tokens=usage.input_tokens,
                                output_tokens=usage.output_tokens,
                                attempts=state.get("attempts", 0),
                                queries=state.get("queries", [])
                )}
        
    @observe(name="node:rewrite", as_type="chain")
    def rewrite(self, state: AgentState) -> dict:
        question = state["question"]
        previous_query = state["queries"]
        
        rewrite, usage = self.llm.generate(SYSTEM_REWRITE_PROMPT, build_rewrite_message(question, previous_query), Rewrite)
        
        print(rewrite)
        
        return {
            "query": rewrite.query,
            "queries": [rewrite.query],
            "attempts": state.get("attempts", 0) + 1,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens
        }
        
LIMIT_REMINDER = (
    "Limite de buscas atingido; esta chamada não foi executada. "
    "Não chame mais buscar_normas. Encerre agora chamando a ferramenta Answer "
    "usando apenas os trechos já recebidos. Se eles não respondem à pergunta, use found=false."
)
TEXT_REMINDER = (
    "Sua resposta não foi registrada porque veio em texto. "
    "Chame agora a ferramenta Answer com a resposta final, citando os ids dos trechos usados."
)

class ToolAgent:
    def __init__(self, llm: BaseChatModel, search: SearchFn, retrieval_method: str, k: int = 5, max_iterations: int = 4):
        self.buscar = make_buscar_normas(search, k)
        self.llm = llm.bind_tools([self.buscar, Answer])
        self.tools = [self.buscar]
        self.retrieval_method = retrieval_method
        self.k = k
        self.max_iterations = max_iterations
        
    def agent(self, state: ToolAgentState) -> dict:
        resp = self.llm.invoke([SystemMessage(content=[{"type": "text", "text": AGENT_PROMPT, "cache_control": {"type": "ephemeral"}}]), *state["messages"]])
        return {"messages": [resp]}
    
    def route(self, state: ToolAgentState) -> str:
        
        last = state["messages"][-1]
        if any(tc["name"] == "Answer" for tc in last.tool_calls):
            return "finalize"
        if state.get("forced"):
            return "finalize"
        if last.tool_calls and self.n_iterations(state) < self.max_iterations:
            return "tools"
        return "force_answer"
    
    def force_answer(self, state: ToolAgentState) -> dict:
        last = state["messages"][-1]
        if last.tool_calls:
            msgs = [ToolMessage(content=LIMIT_REMINDER, tool_call_id=tc["id"], name=tc["name"]) for tc in last.tool_calls]
        else:
            msgs = [HumanMessage(content=TEXT_REMINDER)]
            
        return {"messages": msgs, "forced": True}
    
    def n_iterations(self, state: ToolAgentState):
        return sum(isinstance(m, AIMessage) for m in state["messages"])
    
    def details(self, m: AIMessage): return m.usage_metadata.get("input_token_details", {}) or {}
    
    def finalize(self, state: ToolAgentState) -> dict:
        chunks: list[SearchResult] = [c for m in state["messages"] if isinstance(m, ToolMessage) for c in (m.artifact or [])]
        ais = [m for m in state["messages"] if isinstance(m, AIMessage)]
        queries = [tc["args"]["query"] for m in ais for tc in m.tool_calls if tc["name"] == "buscar_normas"]
        usage = LLMUsage(
                    input_tokens=sum(m.usage_metadata["input_tokens"] for m in ais), 
                    output_tokens=sum(m.usage_metadata["output_tokens"] for m in ais),
                    cache_read_tokens=sum(self.details(m).get("cache_read") or 0 for m in ais),
                    cache_creation_tokens=sum(self.details(m).get("ephemeral_5m_input_tokens") or 0 for m in ais),
                )
        tools_called = [tc["name"] for m in ais for tc in m.tool_calls if tc["name"] != "Answer"]
        
        last = ais[-1]
        message = [tc for tc in last.tool_calls if tc["name"] == "Answer"]
        
        if not message:
            return {"response": RagResponse.not_found(chunks, usage, attempts=len(queries), query=queries, tools_called=tools_called, reason="no_answer")}
        
        valid_ids = {c.chunk_id for c in chunks}

        try:
            answer = Answer(**message[0]["args"])
        except ValidationError:
            return {"response": RagResponse.not_found(chunks, usage, attempts=len(queries), query=queries, tools_called=tools_called, reason="invalid_answer")}

        if not answer.found:
            return {"response": RagResponse.not_found(chunks, usage, attempts=len(queries), query=queries, tools_called=tools_called, reason="abstained")}
                
        if not answer.citations or not set(answer.citations) <= valid_ids:
            return {"response": RagResponse.not_found(chunks, usage, attempts=len(queries), query=queries, tools_called=tools_called, reason="invalid_citation")}

        citations = list(dict.fromkeys(answer.citations))

        return {"response": RagResponse(
                                    found=answer.found,
                                    sources=[c for c in chunks if c.chunk_id in citations],
                                    retrieved=chunks,
                                    answer=answer.answer,
                                    input_tokens=usage.input_tokens,
                                    output_tokens=usage.output_tokens,
                                    attempts=len(queries),
                                    queries=queries,
                                    tools_called=tools_called,
                                    reason="forced_answer" if state.get("forced") else "answered",
                                    cache_creation_tokens=usage.cache_creation_tokens,
                                    cache_read_tokens=usage.cache_read_tokens
                )}