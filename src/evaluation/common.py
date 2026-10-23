from pathlib import Path
import subprocess
from typing import Optional, Sequence

from pydantic import BaseModel

from bacen_agent.retrieval.retriever import SearchResult

class QuestionResult(BaseModel):
    id: int
    style: str
    position: int | None
    retrieved: list[str]

class Expected(BaseModel):
    type: str
    number: int
    article: Optional[str] = None
    part: Optional[str] = None

class Question(BaseModel):
    id: int
    question: str
    expected: list[Expected]
    scope: str
    style: str
    author: str
    notes: Optional[str] = None

def git_commit() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout.strip()
        
        return f"{sha}{'-dirty' if dirty else ''}"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 'unkown'
    
def save_run(summary: BaseModel, results: Sequence[BaseModel], out_dir: Path, history: str) -> None:
    runs = out_dir / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    (runs / f"{summary.run_id}.jsonl").write_text(
        "".join(r.model_dump_json() + "\n" for r in results), encoding="utf-8"
    )
    
    with (out_dir / history).open("a", encoding="utf-8") as f:
        f.write(summary.model_dump_json() + "\n")
        
def load_questions(path: Path) -> list[Question]:
    return [Question.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def matches(result: SearchResult, expected: Expected) -> bool:
    return (
        result.type == expected.type
        and result.number == expected.number
        and (expected.article is None or result.article == expected.article)
        and (expected.part is None or result.part.startswith(expected.part))
    )