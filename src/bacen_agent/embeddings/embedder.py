import torch
from sentence_transformers import SentenceTransformer

from bacen_agent.config import Settings

class EmbedderModel:
    def __init__(self, settings: Settings):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = SentenceTransformer(
            settings.embedding_model,
            device=self.device,
            model_kwargs={"torch_dtype": torch.float16} if self.device == "mps" else {},
        )
        # seus chunks têm até ~500 tokens; o padrão do bge-m3 é 8192
        self.model.max_seq_length = 512

    @torch.inference_mode()
    def encode(self, documents: list[str], batch_size: int = 16) -> list[list[float]]:
        vectors = self.model.encode(
            documents,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).tolist()
        if self.device == "mps":
            torch.mps.empty_cache()
        return vectors

