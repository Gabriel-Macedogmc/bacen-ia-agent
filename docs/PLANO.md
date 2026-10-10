# Plano do projeto: Assistente Regulatório BACEN (RAG + Agente)

## Contexto

Projeto de portfólio para a vaga de Engenheiro de IA pleno (Itaú), baseado em
`Projeto_Assistente_Regulatorio_RAG.docx`.

**Objetivo:** um assistente que responde perguntas sobre normas públicas do Banco Central
(Pix, Open Finance, resoluções e circulares). Ele sempre cita a fonte, mascara dados pessoais
e executa ações simples por meio de ferramentas.

**Por que este projeto:** ele reúne, num único sistema, o que a vaga pede: RAG, agentes,
ML clássico, guardrails, avaliação de LLM, deploy na AWS e observabilidade.

---

## Arquitetura

Primeiro roda tudo localmente com Docker Compose. A AWS entra só na última fase.

```
Pergunta
  │
  ▼
FastAPI
  │
  ▼
Guardrail de entrada ── mascara CPF, CNPJ, conta etc. antes de qualquer chamada ao LLM
  │
  ▼
Classificador de intenção (scikit-learn) ── barato, evita gastar LLM à toa
  ├─ fora_escopo ──► resposta padrão (sem LLM)
  └─ norma / cálculo
        │
        ▼
     Agente LangGraph ── decide quais ferramentas usar
        ├─ tool: buscar_normas  (OpenSearch híbrido: BM25 + vetorial)
        ├─ tool: calculadora
        └─ tool: conta_mock     (API fake de conta)
        │
        ▼
     LLM gera a resposta com citações
        │
        ▼
Guardrail de saída ── a resposta tem fonte? Se não tiver, responde "não encontrei"
```

**Partes transversais**
- **Observabilidade:** Langfuse ou OpenTelemetry, com trace do agente, latência, tokens e custo por requisição.
- **Avaliação:** cerca de 50 perguntas com métricas RAGAS, rodando no GitHub Actions a cada PR.
- **Infraestrutura como código:** Terraform na fase da AWS. Logs no CloudWatch, sem dados pessoais.
- **Ingestão por eventos (opcional):** uma norma nova publicada vira um evento no SQS ou Kafka e é reindexada automaticamente.

### Decisões principais

1. **LLM e embeddings atrás de uma interface** (`llm/provider.py`). Localmente você usa uma API
   direta (Anthropic ou OpenAI) e embeddings locais. Na fase da AWS troca para o Bedrock sem
   mexer no agente.
2. **OpenSearch no Docker Compose desde o início.** É o mesmo motor que vai para a AWS
   (OpenSearch Service), então nada muda na migração.
3. **Metadados em cada trecho indexado:** `norma_id`, `tipo` (resolução, circular),
   `numero`, `data`, `artigo`, `url`. As citações na resposta dependem deles.

---

## Estrutura de pastas

```
bacen-ai-agent/
├── pyproject.toml            # uv ou poetry; ruff + mypy + pytest
├── docker-compose.yml        # opensearch, opensearch-dashboards, langfuse, api
├── .env.example
├── README.md                 # diagrama, como rodar, resultados da avaliação
├── docs/                     # este plano + ADRs (por que LangGraph, por que busca híbrida etc.)
├── data/
│   ├── raw/                  # PDFs/HTML baixados do BACEN (no .gitignore)
│   ├── processed/            # trechos em JSONL
│   └── eval/                 # ~50 perguntas + resposta + fonte esperada
├── src/bacen_agent/
│   ├── config.py             # pydantic-settings
│   ├── ingestion/            # scraper.py, parser.py, chunker.py, indexer.py
│   ├── retrieval/            # retriever.py (híbrido), reranker.py (opcional)
│   ├── llm/                  # provider.py (interface), anthropic.py, bedrock.py
│   ├── guardrails/           # pii.py (entrada), citation_check.py (saída)
│   ├── classifier/           # train.py, model.py, modelo .joblib
│   ├── agent/                # graph.py (LangGraph), state.py, prompts.py, tools/
│   ├── api/                  # main.py (FastAPI), schemas.py, routes/
│   └── observability/        # configuração do Langfuse ou OpenTelemetry
├── evaluation/               # run_ragas.py, métricas de busca (recall@k, MRR)
├── notebooks/                # exploração (não é código de produção)
├── tests/                    # unit/ e integration/
├── infra/terraform/          # fase da AWS
└── .github/workflows/        # lint + testes + avaliação em PR
```

---

## Ordem das fases (por onde começar)

**Não comece pelo agente.** O agente só orquestra as ferramentas. Sem dados bons e uma busca
medida, você não consegue saber se ele funciona nem provar que melhorou algo.

Na entrevista, contar essa ordem ("medi primeiro, depois adicionei o agente e mostrei o ganho")
demonstra maturidade de engenharia.

### Fase 1: dados e ingestão (comece aqui)
- Escolha um recorte pequeno: regulamento do Pix (Resolução BCB nº 1/2020 e anexos) e umas 10 normas.
- Baixe as normas, extraia o texto do PDF e divida por artigo e parágrafo.
- Salve os trechos em JSONL com os metadados.
- Indexe no OpenSearch com BM25 e vetores.

### Fase 2: conjunto de avaliação e RAG simples (linha de base)
- Escreva de 20 a 50 perguntas com a fonte esperada **antes** de otimizar qualquer coisa.
- Monte uma versão linear: busca → prompt → resposta com citação.
- Meça recall@k e as métricas RAGAS (faithfulness, context precision).
- O resultado é a sua linha de base.

### Fase 3: agente LangGraph
- Transforme a busca em ferramenta (`buscar_normas`).
- Adicione a `calculadora` e a `conta_mock`.
- Guarde o histórico da conversa no estado do agente.
- Compare os resultados com a linha de base.

### Fase 4: guardrails e classificador
- Mascaramento de dados pessoais com regex para CPF, CNPJ e conta (Presidio é opcional).
- Checagem de citação na saída.
- Classificador scikit-learn (TF-IDF + LogisticRegression) treinado com perguntas sintéticas.

### Fase 5: API, observabilidade e CI
- FastAPI.
- Langfuse para os traces.
- GitHub Actions rodando lint, testes e avaliação a cada PR.

### Fase 6: AWS
- Bedrock (LLM e embeddings), OpenSearch Service, S3, ECS Fargate.
- Terraform para toda a infraestrutura.
- SQS para reindexação automática (opcional).

---

## Primeiro passo concreto (Fase 1)

1. `pyproject.toml` com as dependências iniciais e `docker-compose.yml` só com o OpenSearch.
2. `ingestion/scraper.py`: baixa as normas do recorte pela busca de normativos do BACEN.
3. `ingestion/parser.py` e `ingestion/chunker.py`: divide por artigo e parágrafo, mantendo os metadados.
4. Um notebook rápido para inspecionar os trechos antes de indexar.

---

## Como verificar cada fase

- **Fase 1:** `docker compose up opensearch`, rode o indexer e confira se uma consulta manual
  retorna o artigo certo para 5 perguntas conhecidas.
- **Fase 2 em diante:** `python evaluation/run_ragas.py` gera um relatório. Os números vão para o README.
- **Toda fase:** `pytest` e `ruff` passando.

---

## Stack

Python, FastAPI, LangGraph, scikit-learn, OpenSearch, Amazon Bedrock (LLM e embeddings), S3,
Lambda, ECS Fargate, Docker, Terraform, Langfuse, RAGAS e GitHub Actions.
