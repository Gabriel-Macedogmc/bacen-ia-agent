from typing import Iterable, Iterator

from opensearchpy import OpenSearch, helpers

from bacen_agent.config import get_settings
from bacen_agent.embeddings.embedder import EmbedderModel
from bacen_agent.ingestion.chunker import Chunk, read_chunks
from bacen_agent.opensearch import make_client
import logging
import time

logger = logging.getLogger(__name__) 

class Indexer:
    def __init__(self, client: OpenSearch, embedder: EmbedderModel, index_name: str):
        self.client = client
        self.embedder = embedder
        self.index_name = index_name
        
    def create_index(self, dimension: int = 1024) -> None:
        if not self.client.indices.exists(index=self.index_name):
            logger.info("Creating index...")
                    
            index_body = {
                        "settings": {
                            "index": {
                                "knn": True,
                            }
                        },
                        "mappings": {
                            "properties": {
                                "chunk_id": {"type": "keyword"},
                                "norma_id": {"type": "integer"},
                                "type":     {"type": "keyword"},
                                "number":   {"type": "integer"},
                                "title":    {"type": "text", "analyzer": "brazilian"},
                                "date":     {"type": "date"},
                                "repealed": {"type": "boolean"},
                                "part":     {"type": "keyword"},
                                "chapter":  {"type": "keyword"},
                                "article":  {"type": "keyword"},
                                "text":     {"type": "text", "analyzer": "brazilian"},
                                "context":  {"type": "text", "analyzer": "brazilian"},
                                "embedding": {
                                    "type": "knn_vector",
                                    "dimension": dimension,
                                    "method": {"name": "hnsw", "engine": "lucene", "space_type": "cosinesimil"},
                                },
                            }
                        }
            }
            self.client.indices.create(index=self.index_name, body=index_body)
            
            logger.info("KNN Index %d Created - Dimensions Vector %f", self.index_name, dimension)
            
    def build_actions(self,batchs: Iterable[list[Chunk]]) -> Iterator[dict]:
        embedded = 0
        
        for batch in batchs:
            start = time.perf_counter()
            vectors = self.embedder.encode([chunk.context for chunk in batch])
            embedded += len(vectors)
            
            logger.info("Embbedings: %d gerados (lote de %d em %.1fs)", embedded, len(batch), time.perf_counter() - start)
            
            for chunk, vector in zip(batch, vectors):
                yield {
                    "_index": self.index_name,
                    "_id": chunk.chunk_id,
                    "_source": {
                        **chunk.model_dump(),
                        "embedding": vector
                    }
                }

    def save_chunks(self, chunks: Iterable[list[Chunk]], log_every: int = 500) -> int:
        indexed = 0
        for success, _ in helpers.streaming_bulk(
            self.client,
            self.build_actions(chunks),
            chunk_size=100,
            max_chunk_bytes=5 * 1024 * 1024,
        ):
            
            if success:
                indexed += 1
                if indexed % log_every == 0:
                    logger.info("OpenSearch: %d documentos gravados", indexed)
        
        self.client.indices.refresh(index=self.index_name)
        total = self.client.count(index=self.index_name)["count"]
        
        logger.info("Concluído: %d gravados nesta execução | %d no índice", indexed, total)
        
        return indexed
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("opensearch").setLevel(logging.WARNING)
    
    settings = get_settings()
    indexer = Indexer(make_client(settings), EmbedderModel(settings), settings.index_name)
    indexer.create_index(settings.embedding_dimension)
    indexer.save_chunks(read_chunks(500))