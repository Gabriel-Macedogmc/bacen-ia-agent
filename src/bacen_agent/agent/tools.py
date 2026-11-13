import ast
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
import json
import operator
from pathlib import Path
from typing import Literal

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

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}

def _eval(node: ast.AST) -> Decimal:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return Decimal(repr(node.value))
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval(node.operand)
    raise ValueError(f"Elemento Não Permitido : {type(node).__name__}")

MAX_EXPR_LEN = 200
CENTAVOS = Decimal("0.01")

@tool(parse_docstring=True)
def calcular(expression: str) -> str:
    """Calcula expressões aritméticas com valores em reais de forma exata.

    Use esta ferramenta para qualquer conta com valores: somas de transações, divisões, descontos, diferenças entre um limite e o valor já usado, quantidades acima de uma franquia. Não faça contas de cabeça, mesmo as simples: um erro de centavos numa resposta financeira é um erro.

    Devolve a expressão e o resultado arredondado para 2 casas decimais, por exemplo "3*349.90 + 1287.45 = 2337.15". Se a expressão for inválida, devolve uma mensagem de erro; corrija a expressão e chame de novo.

    Como escrever a expressão:
    - Use apenas números, os operadores + - * / e parênteses.
    - Use ponto como separador decimal e não use separador de milhar: R$ 1.287,45 vira 1287.45.
    - Escreva porcentagens como multiplicação: 7% de desconto sobre 2149.90 vira 2149.90 * 0.93; 7% de 2149.90 vira 2149.90 * 0.07.
    - Não escreva "R$", unidades, nomes de variáveis ou texto.

    Use apenas números que estejam na pergunta, nos dados da conta ou nos trechos das normas. Nunca suponha um valor que não foi informado, como uma taxa de juros, um rendimento ou um limite que você não encontrou: nesse caso, não calcule e informe que o dado não está disponível.

    Args:
        expression: Expressão aritmética com números, + - * / e parênteses, por exemplo "(820 * 1.50) + (415 * 2.25)".
    """
    
    if len(expression) > MAX_EXPR_LEN:
        return f"Erro: expressão com mais de {MAX_EXPR_LEN} caracteres. Divida a conta em partes menores."

    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval(tree).quantize(CENTAVOS, rounding=ROUND_HALF_UP)
    except SyntaxError:
        return "Erro: expressão inválida. Use apenas números, + - * / e parênteses, com ponto como separador decimal (1287.45)."
    except ValueError as e:
        return f"Erro: {e}. Use apenas números e + - * /, com ponto decimal e sem separador de milhar (1.287,45 vira 1287.45)."
    except ZeroDivisionError:
        return "Erro: divisão por zero."
    except (InvalidOperation, RecursionError):
        return "Erro: não foi possível calcular esta expressão. Simplifique a conta."
    
    return f"{expression} = {result}"

def make_consultar_conta(conta: dict) -> BaseTool:
    @tool(parse_docstring=True)
    def consultar_conta(campo: Literal["saldo", "limites", "transacoes", "chaves", "saques_troco_mes", "consentimentos_open_finance"]) -> str:
        """Consulta os dados da conta Pix do usuário que está fazendo a pergunta: saldo, limites, chaves, transações, saques do mês e consentimentos do Open Finance.

        Use esta ferramenta quando a pergunta for sobre a própria conta de quem pergunta, por exemplo "meu saldo", "minhas chaves", "quanto ainda posso enviar hoje", "esse Pix que recebi". Não responda esse tipo de pergunta sem consultar: os dados da conta não estão nas normas nem no seu conhecimento.

        Não use esta ferramenta:
        - para perguntas hipotéticas ou genéricas ("uma pessoa já enviou...", "quanto alguém pode..."): nesse caso os valores estão na pergunta ou nas normas;
        - para dados de outra pessoa. A ferramenta só acessa a conta de quem pergunta; pedidos sobre a conta de terceiros devem ser recusados sem chamar a ferramenta.

        Devolve um JSON com o campo pedido e a data_referencia, que é o momento atual da consulta. Use a data_referencia para interpretar "hoje", "agora", "este mês" e para contar prazos.

        O que cada campo traz:
        - saldo: saldo disponível e saldo bloqueado.
        - limites: limites de Pix diurno e noturno da conta, o horário do período noturno e quanto já foi usado hoje em cada período.
        - transacoes: últimos lançamentos com data, tipo ("enviado" ou "recebido"), valor, contraparte e status (por exemplo "liquidado" ou "bloqueio_cautelar"). Filtre pelo tipo e pelo status antes de somar ou analisar.
        - chaves: chaves Pix cadastradas, com tipo e valor.
        - saques_troco_mes: quantidade de saques e trocos pelo Pix feitos no mês.
        - consentimentos_open_finance: instituições com consentimento de compartilhamento de dados, os dados compartilhados, o status ("ativo" ou "revogado") e a data de expiração.

        Peça um campo por chamada; se precisar de mais de um, chame a ferramenta de novo. A conta traz os números, mas não as regras: quando a resposta depender de uma regra (limite padrão, quantidade de saques gratuitos, prazo de bloqueio, direito de revogar), busque também em buscar_normas e faça as contas com calcular.

        Args:
            campo: Campo da conta a consultar.
        """
        
        if campo not in conta:
            return f"Erro: campo '{campo}' não encontrado na conta."
        
        return str(conta[campo])
        
    return consultar_conta
    
        
if __name__ == "__main__":
    settings = get_settings()
    make_langfuse(settings)
    
    retriever = Retriever(make_client(settings), EmbedderModel(settings), settings.index_name, Reranker(settings))
    search = search_methods(retriever, settings)[settings.retrieval_method]

    # buscar = make_buscar_normas(search)
    # print(buscar.args_schema.model_json_schema())

    # msg = buscar.invoke({"name": "buscar_normas", "args": {"query": "Mecanismo Especial de Devolução"},
    #                      "id": "teste-1", "type": "tool_call"})
    # print(msg.content[:500])
    # print([c.chunk_id for c in msg.artifact])
    
    # for e in ["3*349.90 + 1287.45", "2149.90 * 0.93", "3780 / 7", "(5 + 6) - 8",
    #       "1.287,45", "1/0", "__import__('os')", "9**9**9", "x + 1"]:
    #     print(calcular.invoke({"expression": e}))
    
    conta = json.loads(Path("data/mock/conta.json").read_text())
    consultar = make_consultar_conta(conta)
    print(consultar.args_schema.model_json_schema())   # confira o enum
    for c in ["saldo", "limites", "transacoes"]:
        print(consultar.invoke({"campo": c}))
    consultar.invoke({"campo": "titular"})
    
    get_client().flush()