from opensearchpy import OpenSearch

from bacen_agent.config import Settings


def make_client(settings: Settings) -> OpenSearch:
    client = OpenSearch(
            hosts=[{"host": settings.opensearch_host, "port": settings.opensearch_port}, ], 
            use_ssl=settings.opensearch_use_ssl,
            verify_certs=settings.opensearch_use_ssl,
        )
    
    if not client.ping():
        raise ConnectionAbortedError("Open Search OFF")
    
    return client