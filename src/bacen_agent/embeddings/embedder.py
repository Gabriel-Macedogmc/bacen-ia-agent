from sentence_transformers import SentenceTransformer

from bacen_agent.config import Settings

class EmbedderModel:
    def __init__(self, settings: Settings):
        self.model = SentenceTransformer(settings.embedding_model, token=settings.hf_access_token)

    def encode(self, documents: list[str], batch_size: int = 64) -> list[list[float]]:
        return self.model.encode(documents, batch_size=batch_size, normalize_embeddings=True).tolist()
        
        
        
        