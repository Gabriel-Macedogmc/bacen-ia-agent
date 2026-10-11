import collections
from dataclasses import dataclass, field
import json
from pathlib import Path
import re

from mypy.nodes import Block
from pydantic import BaseModel

from bacen_agent.ingestion.parser import Norma

ART = re.compile(r"^Art\. ?(\d+[º°]?(?:-[A-Z]+)?)")
PAR = re.compile(r"^(§ ?\d+|Parágrafo único)")

@dataclass
class Block:
    article: str | None
    lines: list[str] = field(default_factory=list)
    
    @property
    def text(self) -> str:
        return "\n".join(self.lines).strip()
    
class Chunk(BaseModel):
    chunk_id: str
    norma_id: int
    type: str
    number: int
    title: str
    date: str
    repealed: bool
    part: str
    chapter: str | None
    article: str | None
    text: str
    context: str

def split_blocks(text: str) -> list[Block]: 
    blocks: list[Block] = []
    current = Block(article=None)
    
    for line in text.split("\n"):
        match = ART.match(line)
       
        if match:
            blocks.append(current)
            current = Block(article=match.group(1).replace("º", "").replace("º", ""))
       
        current.lines.append(line)
        
    blocks.append(current)
    return [b for b in blocks if b.text]

def split_long(block: Block, max_chars: int) -> list[str]:
    max_chars_valid = sum(len(line) for line in block.lines) <= max_chars
    
    if max_chars_valid:
         

def split_by_size(text: str, max_chars: int) -> list[str]: ...  

def chunk_norma(norma: Norma) -> list[Chunk]: ...

def chunk_all(in_path: Path, out_path: Path) -> None:
    for l in Path(in_path).read_text(encoding="utf-8").splitlines():
            data = Norma.model_validate(json.loads(l))
            splitedBlocks = split_blocks(data.text)
            print(splitedBlocks[1])

chunk_all(Path("data/interim/normas.jsonl") , Path("data/processed/chunks.jsonl"))