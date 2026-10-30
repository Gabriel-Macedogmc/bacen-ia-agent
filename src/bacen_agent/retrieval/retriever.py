from collections import defaultdict
from functools import partial
from typing import Callable

from langfuse import observe
from opensearchpy import OpenSearch
from pydantic import BaseModel

from bacen_agent.config import Settings, get_settings
from bacen_agent.embeddings.embedder import EmbedderModel
import logging

from bacen_agent.observability import make_langfuse
from bacen_agent.opensearch import make_client
from bacen_agent.retrieval.reranker import Reranker
from bacen_agent.retrieval.schemas import SearchResult

logger = logging.getLogger(__name__) 

class Retriever:
    def __init__(self, client: OpenSearch, embedder: EmbedderModel, index_name: str, reranker: Reranker | None = None):
        self.client = client
        self.embedder = embedder
        self.index_name = index_name
        self.reranker = reranker
        
    def search_bm25(self, query: str, k: int = 5, only_active: bool = True) -> list[SearchResult]:

        filters = [{"term": {"repealed": False}}] if only_active else []

        query_body = {
            "size": k,
            "_source": {"excludes": ["embedding"]},
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["context", "text"]
                            }
                        }
                    ],
                    "filter": filters,
                }
            },
        }
        
        response = self.client.search(
                        body=query_body,
                        index=self.index_name
                    )

        hits = response['hits']['hits']
                
        return [SearchResult(**hit["_source"], score=hit["_score"]) for hit in hits]
    
    def search_knn(self, query: str, k: int = 100, only_active: bool = True) -> list[SearchResult]:
        embedded = self.embedder.encode(documents=[query])[0]
        
        filters = [{"term": {"repealed": False}}] if only_active else []
        
        query_body = {
                    "size": k,
                    "_source": {"excludes": ["embedding"]},
                    "query": {
                        "knn": {
                            "embedding": {
                                "vector": embedded, "k": max(k, 100), "filter": {
                                    "bool": {
                                        "filter": filters
                                        }
                                    }
                                }
                            }
                    },
                }
        
        response = self.client.search(
                                body=query_body,
                                index=self.index_name
                            )
        
        hits = response['hits']['hits']
                        
        return [SearchResult(**hit["_source"], score=hit["_score"]) for hit in hits]

    def search_hybrid(self, query: str, k: int = 5, candidates: int = 50, rrf_k: int = 60, only_active: bool = True, bm25_weight: float = 0, knn_weight: float = 1.0) -> list[SearchResult]:
        bm25 = self.search_bm25(query, k=candidates, only_active=only_active)
        knn = self.search_knn(query, k=candidates, only_active=only_active)
        
        scores: dict[str, float] = defaultdict(float)
        docs: dict[str, SearchResult] = {}
        
        for results, weight in (bm25, bm25_weight), (knn, knn_weight):
            for rank, result in enumerate(results, start=1):
                scores[result.chunk_id] += weight / (rrf_k + rank)
                docs.setdefault(result.chunk_id, result)
        
        ranked = sorted(scores, key=scores.get, reverse=True)[:k]

        return [docs[cid].model_copy(update={"score": scores[cid]}) for cid in ranked]
    
    @observe(as_type="retriever")
    def search_rerank(self, query: str, k: int = 5, candidates: int = 50, only_active: bool = True) -> list[SearchResult]:
        if self.reranker is None:
            raise ValueError("search_rerank precisa de um Reranker no construtor do Retriever")
            
        knn = self.search_knn(query, candidates, only_active)
            
        rerank = self.reranker.rerank(query, knn, len(knn))
            
        return rerank[:k]
    
    @observe(as_type="retriever")
    def search_rerank_with_rrf(self, query: str, k: int = 5, candidates: int = 50, only_active: bool = True) -> list[SearchResult]:
        if self.reranker is None:
            raise ValueError("search_rerank_rrf precisa de um Reranker no construtor do Retriever")
        
        knn = self.search_knn(query, candidates, only_active)
        
        rerank = self.reranker.rerank(query, knn, len(knn))
        
        scores: dict[str, float] = defaultdict(float)
        docs: dict[str, SearchResult] = {}
                
        for results in (knn, rerank):
            for rank, result in enumerate(results, start=1):
                scores[result.chunk_id] += 1 / (60 + rank)
                docs.setdefault(result.chunk_id, result)
                
        ranked = sorted(scores, key=scores.get, reverse=True)[:k]
        
        return [docs[cid].model_copy(update={"score": scores[cid]}) for cid in ranked]

SearchFn = Callable[[str, int], list[SearchResult]]

def search_methods(retriever: Retriever, settings: Settings) -> dict[str, SearchFn]:
    c = settings.reranker_candidates
    w = settings.hybrid_bm25_weight          # 0.25

    methods: dict[str, SearchFn] = {
        "bm25": retriever.search_bm25,
        "knn": retriever.search_knn,
        f"hybrid_w{w}_c{c}": partial(retriever.search_hybrid, candidates=c, bm25_weight=w),
    }
    if retriever.reranker is not None:
        methods[f"rerank_rrf_c{c}"] = partial(retriever.search_rerank_with_rrf, candidates=c)
        methods[f"rerank_c{c}"] = partial(retriever.search_rerank, candidates=c)
        
    return methods

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    settings = get_settings()
    make_langfuse(settings)
    retriever = Retriever(make_client(settings), EmbedderModel(settings), settings.index_name)
    
    response = retriever.search_hybrid(query="Qual a taxa Selic atual?", k=10)
    logger.info("Results: %s", response)

    #print(EmbedderModel(settings).encode(documents=["O que acontece com um banco que descumpre as regras do Pix?"])[0])

