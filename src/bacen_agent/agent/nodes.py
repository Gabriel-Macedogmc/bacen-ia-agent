from langfuse import observe

from bacen_agent.agent.state import AgentState
from bacen_agent.llm.provider import LLMProvider, LLMUsage
from bacen_agent.rag.pipeline import RagResponse
from bacen_agent.rag.prompts import SYSTEM_PROMPT, SYSTEM_REWRITE_PROMPT, build_rewrite_message, build_user_message
from bacen_agent.rag.schemas import Answer, Rewrite
from bacen_agent.retrieval.retriever import SearchFn


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
        