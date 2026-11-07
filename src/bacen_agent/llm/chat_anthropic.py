from langchain_anthropic import ChatAnthropic

from bacen_agent.config import Settings

class ChatAnthropicProvider:

    def __init__(self, settings: Settings):
        self.model = ChatAnthropic(
            model=settings.llm_model,
            api_key=settings.anthropic_key,
            max_tokens=16000,
            output_config={
                "effort": settings.llm_effort
            }
        )