from opensearchpy import OpenSearch
from pydantic import BaseModel

from bacen_agent.config import get_settings
from bacen_agent.embeddings.embedder import EmbedderModel
import logging

from bacen_agent.opensearch import make_client

logger = logging.getLogger(__name__) 

class SearchResult(BaseModel):
    chunk_id: str
    type: str
    number: int
    article: str | None
    part: str
    title: str
    text: str
    date: str
    norma_id: int
    score: float
    
class Retriever:
    def __init__(self, client: OpenSearch, embedder: EmbedderModel, index_name: str):
        self.client = client
        self.embedder = embedder
        self.index_name = index_name
    
    def search_bm25(self, query: str, k: int = 5, only_active: bool = True) -> list[SearchResult]:

        filters = [{"term": {"repealed": False}}] if only_active else []

        query_body = {
            "size": k,
            "_source": {"excludes": ["embedding"]},
            "query": {
                "bool": {
                    "must": [{"match": {"text": query}}],
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
        
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    settings = get_settings()
    retriever = Retriever(make_client(settings), EmbedderModel(settings), settings.index_name)
    
    response = retriever.search_knn(query="Qual a taxa Selic atual?", k=10)
    logger.info("Results: %s", response)

    #print(EmbedderModel(settings).encode(documents=["O que acontece com um banco que descumpre as regras do Pix?"])[0])

