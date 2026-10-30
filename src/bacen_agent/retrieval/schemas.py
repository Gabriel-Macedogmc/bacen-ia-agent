from pydantic import BaseModel

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
    context: str
    retrieval_method: str | None = None