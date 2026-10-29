from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from pydantic import BaseModel

from bacen_agent.config import Settings, get_settings
from bacen_agent.embeddings.embedder import EmbedderModel
from bacen_agent.opensearch import make_client
from bacen_agent.retrieval.reranker import Reranker
from bacen_agent.retrieval.retriever import Retriever
from bacen_agent.retrieval.schemas import SearchResult
from evaluation.common import Expected, Question, QuestionResult, git_commit, load_questions, matches, save_run

SearchFn = Callable[[str, int], list[SearchResult]]
STYLES = ["literal", "user", "reference"]
CUTOFFS = (1, 5, 10)

class RetrievalRunSummary(BaseModel):
    run_id: str
    method: str
    created_at: str
    git_commit: str
    embedding_model: str
    index_name: str
    n_questions: int
    recall: dict[str, float]
    mrr: float
    recall5_by_style: dict[str, float]
    misses: list[int]

def print_header() -> None:
    styles = "  ".join(f"{s:>9}" for s in STYLES)
    print(f"{'método':<8} {'n':>3} {'R@1':>6} {'R@5':>6} {'R@10':>6} {'MRR':>6} | R@5 por estilo: {styles}")

def report(name: str, results: list[QuestionResult]) -> None:
    by_styles = []
    
    for style in STYLES:
        group = [r for r in results if r.style == style]
        by_styles.append(f"{recall_at(group, 5):>9.2f}" if group else f"{'-':>9}")

    print(
        f"{name:<8} {len(results):>3} {recall_at(results, 1):>6.2f} {recall_at(results, 5):>6.2f} "
        f"{recall_at(results, 10):>6.2f} {mrr(results):>6.2f} |                 {'  '.join(by_styles)}"
    )

    misses = [r.id for r in results if r.position is None or r.position > 10]
    print(f"{'':<8} não encontradas no top 10: {misses}")

def summarize(method: str, results: list[QuestionResult], settings: Settings) -> RetrievalRunSummary:
    now = datetime.now(timezone.utc)
    return RetrievalRunSummary(
        run_id=f"{now:%Y%m%d-%H%M%S}_{method}",
        method=method,
        created_at=now.isoformat(timespec="seconds"),
        git_commit=git_commit(),
        embedding_model=settings.embedding_model,
        index_name=settings.index_name,
        n_questions=len(results),
        recall={f"@{k}": round(recall_at(results, k), 4) for k in CUTOFFS},
        mrr=round(mrr(results), 4),
        recall5_by_style={s: round(recall_at(g, 5), 4) for s in STYLES if (g := [r for r in results if r.style == s])},
        misses=[r.id for r in results if r.position is None or r.position > 10],
    )
    
def first_hit(results: list[SearchResult], expected: list[Expected]) -> int | None:
    for index, r in enumerate(results, start=1):
        if any(matches(r, e) for e in expected):
            return index
    return None
 
def recall_at(results: list[QuestionResult], k: int) -> float:
    count = len([r for r in results if r.position is not None and r.position <= k])
    
    return count / len(results)

def mrr(results: list[QuestionResult], k: int = 5) -> float:
    return sum(1 / p.position for p in results if p.position is not None and p.position <= k) / len(results)

def evaluate(search: SearchFn, questions: list[Question], k: int = 100) -> list[QuestionResult]:
    response: list[QuestionResult] = []
    
    for q in questions:
        if q.scope == "in":
            search_result = search(q.question, k)
            position = first_hit(results=search_result, expected=q.expected)
            
            response.append(QuestionResult(
                position=position,
                id=q.id,
                style=q.style,
                retrieved=[sr.chunk_id for sr in search_result]
            ))
    
    return response

if __name__ == "__main__":
    settings = get_settings()
    retriever = Retriever(make_client(settings), EmbedderModel(settings), settings.index_name, reranker=Reranker(settings))
    questions = load_questions(Path("data/eval/questions.jsonl"))

    print_header()
    for name, search in {"bm25": retriever.search_bm25, "knn": retriever.search_knn, "hybrid": retriever.search_hybrid, f"rerank_c{settings.reranker_candidates}": retriever.search_rerank}.items():
        results = evaluate(search, questions)
        report(name, results)
        save_run(summarize(name, results, settings), results, Path("data/eval/results"), "retrieval_history.jsonl")
        