from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

from bacen_agent.config import Settings, get_settings
from bacen_agent.embeddings.embedder import EmbedderModel
from bacen_agent.llm.anthropic import AnthropicProvider
from bacen_agent.opensearch import make_client
from bacen_agent.rag.pipeline import RagPipeline
from bacen_agent.rag.prompts import PROMPT_VERSION
from bacen_agent.retrieval.retriever import Retriever
from evaluation.common import Question, git_commit, load_questions, matches, save_run

class AnswerResult(BaseModel):
    id: int
    scope: str
    style: str
    found: bool
    retrieval_hit: bool
    citation_hit: bool
    answer: str
    citations: list[str]
    input_tokens: int
    output_tokens: int

class AnswerRunSummary(BaseModel):
    run_id: str
    created_at: str
    git_commit: str
    prompt_version: str
    llm_model: str
    llm_effort: str
    embedding_model: str
    cost_usd: float
    input_tokens: int
    output_tokens: int
    k: int
    n_in: int
    n_off: int
    response_rate: float
    citation_rate: float
    citation_rate_when_retrieved: float
    retrieval_hit_rate: float
    abstention_rate: float
    wrong_citations: list[int]
    wrong_answers_off: list[int]
    
def response_rate(results: list[AnswerResult]) -> float:
    return sum(r.found for r in results) / len(results) if results else 0.0

def citation_rate(results: list[AnswerResult]) -> float:
    return sum(r.citation_hit for r in results) / len(results) if results else 0.0

def citation_rate_when_retrieved(results: list[AnswerResult]) -> float:
    retrieved = [r for r in results if r.retrieval_hit]
    
    return sum(r.citation_hit for r in retrieved) / len(retrieved) if retrieved else 0.0

def abstention_rate(results: list[AnswerResult]) -> float:
    return sum(not r.found for r in results) / len(results) if results else 0.0

def evaluate(pipeline: RagPipeline, questions: list[Question]) -> list[AnswerResult]:
    answer_result: list[AnswerResult] = []
    
    for q in questions:
        result = pipeline.answer(q.question)
        answer_result.append(AnswerResult(
            answer=result.answer,
            scope=q.scope,
            retrieval_hit=any(matches(r, e) for r in result.retrieved for e in q.expected),
            citation_hit=any(matches(r, e) for r in result.sources for e in q.expected),
            found=result.found,
            citations=[s.chunk_id for s in result.sources],
            style=q.style,
            id=q.id,
            output_tokens=result.output_tokens,
            input_tokens=result.input_tokens
        ))

    return answer_result

def summarize(results: list[AnswerResult], settings: Settings, k: int = 5) -> AnswerRunSummary:
    in_scope  = [r for r in results if r.scope == "in"]
    off_scope = [r for r in results if r.scope != "in"]

    input_tokens = sum(r.input_tokens for r in results)
    output_tokens = sum(r.output_tokens for r in results)
    
    now = datetime.now(timezone.utc)
    return AnswerRunSummary(
        run_id=f"{now:%Y%m%d-%H%M%S}_answers",
        created_at=now.isoformat(timespec="seconds"),
        git_commit=git_commit(),
        prompt_version=PROMPT_VERSION,
        llm_model=settings.llm_model,
        llm_effort=settings.llm_effort,
        embedding_model=settings.embedding_model,
        cost_usd= (input_tokens * settings.llm_input_price + output_tokens * settings.llm_output_price) / 1_000_000,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        k=k,
        n_in=len(in_scope),
        n_off=len(off_scope),
        response_rate=response_rate(in_scope),
        citation_rate=citation_rate(in_scope),
        citation_rate_when_retrieved=citation_rate_when_retrieved(in_scope),
        retrieval_hit_rate=sum(r.retrieval_hit for r in in_scope) / len(in_scope),
        abstention_rate=abstention_rate(off_scope),
        wrong_citations=[r.id for r in in_scope if r.retrieval_hit and not r.citation_hit],
        wrong_answers_off=[r.id for r in off_scope if r.found]
    )

if __name__ == "__main__":
    settings = get_settings()
    retriever = Retriever(make_client(settings), EmbedderModel(settings), settings.index_name)
    provider = AnthropicProvider(settings)
    
    ragPipeline = RagPipeline(retriever, provider)
    
    questions = load_questions(Path("data/eval/questions.jsonl"))
    
    results = evaluate(pipeline=ragPipeline, questions=questions)
    
    save_run(summarize(results, settings, ragPipeline.k), results, Path("data/eval/results"), "answers_history.jsonl" )
    
        