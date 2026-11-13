from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import time

from langfuse import get_client, propagate_attributes
from pydantic import BaseModel

from bacen_agent.agent.graph import Answerer, RagGraph
from bacen_agent.agent.nodes import RagAgent, ToolAgent
from bacen_agent.agent.tool_graph import AgentGraph
from bacen_agent.config import Settings, get_settings
from bacen_agent.embeddings.embedder import EmbedderModel
from bacen_agent.llm.anthropic import AnthropicProvider
from bacen_agent.llm.chat_anthropic import ChatAnthropicProvider
from bacen_agent.observability import make_langfuse
from bacen_agent.opensearch import make_client
from bacen_agent.rag.pipeline import RagPipeline
from bacen_agent.rag.prompts import AGENT_PROMPT_VERSION, PROMPT_VERSION
from bacen_agent.retrieval.reranker import Reranker
from bacen_agent.retrieval.retriever import Retriever, search_methods
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
    attempts: int | None = None
    queries: list[str] | None = None
    latency_s: float | None = None
    task: str
    tools_called: list[str] | None = None
    reason: str | None = None
    tool_selection_hit: bool | None = None
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    
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
    tool_selection_accuracy: float | None = None
    mean_tool_calls: float | None = None
    wrong_citations: list[int]
    wrong_answers_off: list[int]
    retrieval_method: str
    runner: str
    max_latency_s: float | None = None
    mean_latency_s: float | None = None
    reasons_in: dict[str, int] | None = None
    reasons_off: dict[str, int] | None = None
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    
def response_rate(results: list[AnswerResult]) -> float:
    return sum(r.found for r in results) / len(results) if results else 0.0

def citation_rate(results: list[AnswerResult]) -> float:
    return sum(r.citation_hit for r in results) / len(results) if results else 0.0

def citation_rate_when_retrieved(results: list[AnswerResult]) -> float:
    retrieved = [r for r in results if r.retrieval_hit]
    
    return sum(r.citation_hit for r in retrieved) / len(retrieved) if retrieved else 0.0

def abstention_rate(results: list[AnswerResult]) -> float:
    return sum(not r.found for r in results) / len(results) if results else 0.0

def tool_selection_hit(called: list[str], q: Question) -> bool:
    called_set = set(called)
    required = set(q.expected_tools)
    allowed = required | set(q.optional_tools)
    
    return required <= called_set and called_set <= allowed

def tool_selection_accuracy(results: list[AnswerResult]) -> float | None:
    scored = [r for r in results if r.tool_selection_hit is not None]
    return sum(r.tool_selection_hit for r in scored) / len(scored) if scored else None

def mean_tool_calls(results: list[AnswerResult]) -> float | None:
    scored = [r for r in results if r.tools_called is not None]
    return sum(len(r.tools_called) for r in scored) / len(scored) if scored else None

def reasons(results: list[AnswerResult]) -> dict[str, int] | None:
    scored = [r.reason for r in results if r.reason is not None]
    return dict(Counter(scored)) if scored else None

def evaluate(pipeline: Answerer, questions: list[Question], now: datetime) -> list[AnswerResult]:
    answer_result: list[AnswerResult] = []
    
    with propagate_attributes(
    session_id=f"{now:%Y%m%d-%H%M%S}_answers",
    tags=[AGENT_PROMPT_VERSION, pipeline.retrieval_method, pipeline.runner],
    metadata={"llm_model": settings.llm_model, "llm_effort": settings.llm_effort, "runner": pipeline.runner, "git_commit": git_commit()},
    ):
        langfuse = get_client()
        for q in questions:
                with langfuse.start_as_current_observation(
                    name=f"q{q.id:02d}-{q.scope}", as_type="span",
                    input=q.question, metadata={"id": q.id, "scope": q.scope, "style": q.style},
                ):
                    start = time.perf_counter()
                    result = pipeline.answer(q.question)
                    latency = time.perf_counter() - start
                    langfuse.score_current_trace(name="latency", value=latency)
                
                    answer = AnswerResult(
                        answer=result.answer,
                        scope=q.scope,
                        retrieval_hit=any(matches(r, e) for r in result.retrieved for e in q.expected),
                        citation_hit=any(matches(r, e) for r in result.sources for e in q.expected),
                        found=result.found,
                        citations=[s.chunk_id for s in result.sources],
                        style=q.style,
                        id=q.id,
                        output_tokens=result.output_tokens,
                        input_tokens=result.input_tokens,
                        attempts=result.attempts,
                        queries=result.queries,
                        latency_s=latency,
                        task=q.task,
                        tools_called=result.tools_called,
                        reason=result.reason,
                        tool_selection_hit=tool_selection_hit(result.tools_called, q) if result.tools_called is not None else None,
                        cache_read_tokens=result.cache_read_tokens,
                        cache_creation_tokens=result.cache_creation_tokens,
                    )
                    langfuse.score_current_trace(name="citation_hit", value=float(answer.citation_hit))
                    langfuse.score_current_trace(name="retrieval_hit", value=float(answer.retrieval_hit))
                    langfuse.score_current_trace(name="found", value=float(answer.found))
                    if answer.tool_selection_hit is not None:
                        langfuse.score_current_trace(name="tool_selection_hit", value=float(answer.tool_selection_hit))
                    answer_result.append(answer)

    return answer_result

def summarize(results: list[AnswerResult], settings: Settings, pipeline: Answerer, now: datetime) -> AnswerRunSummary:
    in_scope  = [r for r in results if r.scope == "in"]
    off_scope = [r for r in results if r.scope != "in"]

    input_tokens = sum(r.input_tokens for r in results)
    output_tokens = sum(r.output_tokens for r in results)
    
    cache_read = sum(r.cache_read_tokens for r in results)
    cache_creation = sum(r.cache_creation_tokens for r in results)
    uncached = input_tokens - cache_read - cache_creation
    
    cost_usd = (uncached * settings.llm_input_price
            + cache_read * settings.llm_cache_read_price
            + cache_creation * settings.llm_cache_write_price
            + output_tokens * settings.llm_output_price) / 1_000_000
    
    return AnswerRunSummary(
        run_id=f"{now:%Y%m%d-%H%M%S}_answers",
        created_at=now.isoformat(timespec="seconds"),
        git_commit=git_commit(),
        prompt_version=AGENT_PROMPT_VERSION,
        llm_model=settings.llm_model,
        llm_effort=settings.llm_effort,
        embedding_model=settings.embedding_model,
        cost_usd= cost_usd,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read,
        cache_creation_tokens=cache_creation,
        k=pipeline.k,
        n_in=len(in_scope),
        n_off=len(off_scope),
        retrieval_method=pipeline.retrieval_method,
        response_rate=response_rate(in_scope),
        citation_rate=citation_rate(in_scope),
        citation_rate_when_retrieved=citation_rate_when_retrieved(in_scope),
        retrieval_hit_rate=sum(r.retrieval_hit for r in in_scope) / len(in_scope),
        abstention_rate=abstention_rate(off_scope),
        tool_selection_accuracy=tool_selection_accuracy(results),
        mean_tool_calls=mean_tool_calls(results),
        reasons_in=reasons(in_scope),
        reasons_off=reasons(off_scope),
        wrong_citations=[r.id for r in in_scope if r.retrieval_hit and not r.citation_hit],
        wrong_answers_off=[r.id for r in off_scope if r.found],
        runner=pipeline.runner,
        max_latency_s=max((r.latency_s for r in results if r.latency_s is not None), default=None),
        mean_latency_s=sum((r.latency_s for r in results if r.latency_s is not None), 0.0) / sum((1 for r in results if r.latency_s is not None), 1)
    )

if __name__ == "__main__":
    settings = get_settings()
    make_langfuse(settings)
    
    retriever = Retriever(make_client(settings), EmbedderModel(settings), settings.index_name, Reranker(settings))
    #provider = AnthropicProvider(settings)
    provider = ChatAnthropicProvider(settings)
    methods = search_methods(retriever, settings)
    
    ragPipeline = RagPipeline(methods[settings.retrieval_method], settings.retrieval_method, provider)
    
    #agent = RagAgent(methods[settings.retrieval_method], settings.retrieval_method, provider)
    agent = ToolAgent(provider.model, methods[settings.retrieval_method], settings.retrieval_method)
    #ragPipeline = RagGraph(agent, settings)
    ragPipeline = AgentGraph(agent)
    
    questions = [q for q in load_questions(Path("data/eval/questions.jsonl")) if q.task == "norma"]
    #questions = [q for q in load_questions(Path("data/eval/questions.jsonl")) if q.id in {9, 33, 34, 38, 44}]

    now = datetime.now(timezone.utc)
    
    results = evaluate(pipeline=ragPipeline, questions=questions, now=now)
    
    save_run(summarize(results, settings, ragPipeline, now), results, Path("data/eval/results"), "answers_history.jsonl" )
    
    get_client().flush()
    
        