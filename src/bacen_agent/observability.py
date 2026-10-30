from langfuse import Langfuse

from bacen_agent.config import Settings

def make_langfuse(settings: Settings) -> Langfuse:
    return Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        base_url=settings.langfuse_base_url
    )