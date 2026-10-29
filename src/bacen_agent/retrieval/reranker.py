from sentence_transformers import CrossEncoder
import torch

from bacen_agent.config import Settings
from bacen_agent.device import device
from bacen_agent.retrieval.schemas import SearchResult
from bacen_agent.time import measure_time

class Reranker:
    def __init__(self, settings: Settings):
        self.device = device()
        self.model = CrossEncoder(settings.reranker_model, device=self.device,  model_kwargs={"torch_dtype": torch.float16} if self.device == "mps" else {}, max_length=512)
    
    @measure_time
    @torch.inference_mode()
    def rerank(self, query: str, candidates: list[SearchResult], k: int) -> list[SearchResult]:
        pairs = [(query, c.context) for c in candidates]
        notes = self.model.predict(pairs, batch_size=16)
        
        result = [c.model_copy(update={"score": float(score)}) for c, score in zip(candidates, notes)]
        
        return sorted(result, key=lambda x: x.score, reverse=True)[:k]
        