import anthropic

from bacen_agent.config import Settings

from typing import TypeVar

from bacen_agent.llm.provider import LLMUsage

T = TypeVar('T')

class AnthropicProvider:
    def __init__(self, settings: Settings):
        self.client = anthropic.Anthropic(api_key=settings.anthropic_key)
        self.model = settings.llm_model
        self.effort = settings.llm_effort
    
    def generate(self, system: str, user: str, output_type: type[T]) -> tuple[T | None, LLMUsage]:
        response = self.client.messages.parse(
            model=self.model,
            max_tokens=16000,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": self.effort},
            output_format=output_type
        )
        
        return response.parsed_output, response.usage