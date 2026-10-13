from dataclasses import dataclass, field
import json
from pathlib import Path
import re

from pydantic import BaseModel

from bacen_agent.ingestion.parser import Norma

ART = re.compile(r"^Art\. ?(\d+)[º°]?(-[A-Z]+)?")
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
            current = Block(article=match.group(1) + (match.group(2) or ""))
       
        current.lines.append(line)
        
    blocks.append(current)
    return [b for b in blocks if b.text]

def split_long(block: Block, max_chars: int) -> list[str]:
    if len(block.text) <= max_chars:
        return [block.text]
    
    parts: list[str] = []
    current: list[str] = []
    
    for line in block.lines:
        if PAR.match(line):
            parts.append("\n".join(current).strip())
            current = []
        current.append(line)
    
    parts.append("\n".join(current).strip())
    
    chunks: list[str] = []
    
    for part in parts:
        if len(part) <= max_chars:
            chunks.append(part)
        else:
            print(f"Parte do Texto {part}")
            chunks.append(part)
    
    return chunks
    
def split_by_size(text: str, max_chars: int) -> list[str]: ...  

def chunk_norma(norma: Norma) -> list[Chunk]: ...

def chunk_all(in_path: Path, out_path: Path) -> None:
    for l in Path(in_path).read_text(encoding="utf-8").splitlines():
            data = Norma.model_validate(json.loads(l))
            
            for block in split_blocks(data.text):
                for text in split_long(block, 1500):
                    text            
            

chunk_all(Path("data/interim/normas.jsonl") , Path("data/processed/chunks.jsonl"))