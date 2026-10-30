from pydantic import BaseModel, Field

class Answer(BaseModel):
    found: bool = Field(description="true somente se os trechos respondem a pergunta")
    answer: str = Field(description="Resposta em portugues simples, ou vazio se found=False")
    citations: list[str] = Field(description="ids dos trechos usados na resposta")
    
class Rewrite(BaseModel): 
    query: str