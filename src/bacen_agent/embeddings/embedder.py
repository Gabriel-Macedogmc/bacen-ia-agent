import torch
from sentence_transformers import SentenceTransformer

from bacen_agent.config import Settings

class EmbedderModel:
    def __init__(self, settings: Settings):
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.model = SentenceTransformer(
            settings.embedding_model,
            device=self.device,
            model_kwargs={"torch_dtype": torch.float16} if self.device == "mps" else {},
        )

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

