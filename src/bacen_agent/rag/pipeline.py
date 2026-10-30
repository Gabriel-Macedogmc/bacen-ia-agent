from langfuse import get_client, observe
from pydantic import BaseModel

from bacen_agent.config import get_settings
from bacen_agent.embeddings.embedder import EmbedderModel
from bacen_agent.llm.anthropic import AnthropicProvider
from bacen_agent.llm.provider import LLMProvider, LLMUsage
from bacen_agent.observability import make_langfuse
from bacen_agent.opensearch import make_client
from bacen_agent.rag.prompts import SYSTEM_PROMPT, build_user_message
from bacen_agent.rag.schemas import Answer
from bacen_agent.retrieval.reranker import Reranker
from bacen_agent.retrieval.retriever import Retriever, SearchFn, SearchResult, search_methods

NOT_FOUND_MESSAGE = "Não encontrei essa informação nas normas consultadas."

class RagResponse(BaseModel):
    found: bool
    answer: str
    sources: list[SearchResult]
    retrieved: list[SearchResult]
    input_tokens: int
    output_tokens: int
    
    @classmethod
    def not_found(cls, retrieved: list[SearchResult], usage: LLMUsage) -> "RagResponse":
        return cls(
            found=False,
            answer=NOT_FOUND_MESSAGE,
            sources=[],
            retrieved=retrieved,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens
        )
    
class RagPipeline:
    def __init__(self, search: SearchFn, retrieval_method: str, llm: LLMProvider, k: int = 5):
        self.search = search
        self.retrieval_method = retrieval_method
        self.llm = llm
        self.k = k
        self.runner = "pipeline" 
    
    @observe(name="rag", as_type="chain")
    def answer(self, question: str) -> RagResponse:
        chunks = self.search(question, self.k)
        answer, usage = self.llm.generate(SYSTEM_PROMPT, build_user_message(question, chunks), Answer)
        
        retrieved_ids = [c.chunk_id for c in chunks]
        valid_ids = set(retrieved_ids)
        
        if (answer is None 
                or not answer.found
                or not answer.citations
                or not set(answer.citations) <= valid_ids):
            return RagResponse.not_found(chunks, usage)
        
        return RagResponse(
            found=answer.found,
            sources=[c for c in chunks if c.chunk_id in answer.citations],
            retrieved=chunks,
            answer=answer.answer,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
        )

if __name__ == "__main__":
    settings = get_settings()
    make_langfuse(settings)
    
    retriever = Retriever(make_client(settings), EmbedderModel(settings), settings.index_name, Reranker(settings))
    provider = AnthropicProvider(settings)
        
    methods = search_methods(retriever, settings)
        
    ragPipeline = RagPipeline(methods[settings.retrieval_method], settings.retrieval_method, provider)
    
    for q in ["O que é o Mecanismo Especial de Devolução?",
          #"Caí num golpe no Pix, consigo o dinheiro de volta?",  
          #"Qual banco tem o Pix mais rápido?"
          ]:                   
        result = ragPipeline.answer(q)
    get_client().flush()