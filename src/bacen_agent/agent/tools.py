from langchain_core.tools import BaseTool, tool
from langfuse import get_client

from bacen_agent.config import get_settings
from bacen_agent.embeddings.embedder import EmbedderModel
from bacen_agent.observability import make_langfuse
from bacen_agent.opensearch import make_client
from bacen_agent.rag.prompts import format_chunk
from bacen_agent.retrieval.reranker import Reranker
from bacen_agent.retrieval.retriever import Retriever, SearchFn, search_methods
from bacen_agent.retrieval.schemas import SearchResult

def make_buscar_normas(search: SearchFn, k: int = 5) -> BaseTool:
    
    @tool(name_or_callable="buscar_normas", response_format="content_and_artifact", parse_docstring=True)
    def buscar_normas(query: str) ->  tuple[str, list[SearchResult]]:
        """Busca trechos das normas do Banco Central do Brasil sobre Pix e Open Finance (resoluções, instruções normativas, circulares e manuais em vigor).

        Use esta ferramenta antes de responder qualquer pergunta sobre regras, prazos, limites de valor, tarifas, obrigações, direitos ou definições do Pix e do Open Finance. Não responda perguntas desse tipo com conhecimento próprio: as normas mudam com frequência.

        Devolve até 5 trechos dentro de <trechos>. Cada <trecho> traz os atributos id, norma, tipo, data, parte e artigo. Use o id para citar a fonte na resposta final.

        Como escrever a consulta:
        - Use os termos técnicos das normas, e não as palavras do dia a dia. Exemplos: "devolver o Pix por golpe" vira "Mecanismo Especial de Devolução fraude"; "banco" vira "instituição participante"; "débito automático pelo Pix" vira "Pix Automático".
        - Se a pergunta tiver mais de uma parte, faça uma busca para cada parte.
        - Se os trechos devolvidos não responderem à pergunta, chame a ferramenta de novo com outros termos (sinônimos técnicos, ou o tema mais geral em vez do detalhe). Não repita uma consulta já feita.
        - Se após duas ou três buscas os trechos continuarem sem a resposta, pare de buscar e informe que não encontrou.

        Não use esta ferramenta para perguntas sem relação com Pix ou Open Finance, nem para dados que mudam com o tempo (cotações, taxas do dia, estatísticas), porque as normas não trazem esses valores.

        Args:
            query: Frase curta e afirmativa em português com os termos técnicos do assunto buscado, por exemplo "prazo do bloqueio cautelar de recursos recebidos". Não copie a pergunta inteira do usuário.
        """

        chunks = search(query, k)
        content = [format_chunk(c) for c in chunks]
        
        if not content:
            return "<trechos></trechos>\nNenhum trecho encontrado para esta consulta.", chunks
        
        return f"<trechos>{'\n'.join(content)}</trechos>", chunks
    
    return buscar_normas
        
if __name__ == "__main__":
    settings = get_settings()
    make_langfuse(settings)
    
    retriever = Retriever(make_client(settings), EmbedderModel(settings), settings.index_name, Reranker(settings))
    search = search_methods(retriever, settings)[settings.retrieval_method]

    buscar = make_buscar_normas(search)
    print(buscar.args_schema.model_json_schema())

    msg = buscar.invoke({"name": "buscar_normas", "args": {"query": "Mecanismo Especial de Devolução"},
                         "id": "teste-1", "type": "tool_call"})
    print(msg.content[:500])
    print([c.chunk_id for c in msg.artifact])
    
    get_client().flush()