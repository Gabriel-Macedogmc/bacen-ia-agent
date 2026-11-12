from pydantic import BaseModel, Field

class Answer(BaseModel):
    """Resposta final ao usuário. Chame esta ferramenta sempre para encerrar, uma única vez, depois de buscar os trechos necessários.

    Use found=true quando os trechos devolvidos por buscar_normas trazem o que foi pedido, e found=false quando não trazem ou quando a pergunta está fora do assunto. Nos dois casos, a resposta deve ser dada por esta ferramenta, e nunca em texto fora dela.
    """
    found: bool = Field(description="true somente se os trechos respondem a pergunta")
    answer: str = Field(description="Resposta em portugues simples, ou vazio se found=False")
    citations: list[str] = Field(description="ids dos trechos usados na resposta")
    
class Rewrite(BaseModel): 
    query: str = Field(description="Query reescrita para o motor de busca, ou vazia se não houver reescrita")