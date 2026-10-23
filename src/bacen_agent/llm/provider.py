from typing import Protocol
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar('T')

class LLMUsage(BaseModel):
    input_tokens: int
    output_tokens: int

class LLMProvider(Protocol):
    def generate(self, system: str, user: str, output_type: type[T]) -> tuple[T, LLMUsage]:...
        