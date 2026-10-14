from dataclasses import dataclass, field
from itertools import islice
import json
from pathlib import Path
import re
from typing import Iterator

from pydantic import BaseModel

from bacen_agent.ingestion.parser import Norma

ART = re.compile(r"^Art\. ?(\d+)[º°]?(-[A-Z]+)?")
PAR = re.compile(r"^(§ ?\d+|Parágrafo único)")
PART = re.compile(r"^(ANEXO|Anexo|REGULAMENTO)\b")
SENTENCE = re.compile(r"(?<=[.;:!?])\s+")

@dataclass
class Block:
    article: str | None
    lines: list[str] = field(default_factory=list)
    part: str = "corpo"
    
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
    article: str | None
    text: str
    context: str

def split_blocks(text: str) -> list[Block]: 
    blocks: list[Block] = []
    current = Block(article=None)
    part = "corpo"
    
    for line in text.split("\n"):
        match = ART.match(line)
       
        if match:
            blocks.append(current)
            current = Block(article=match.group(1) + (match.group(2) or ""), part=part)
        elif PART.match(line):
            blocks.append(current)
            part = line.strip()[:120]
            current = Block(article=None, part=part)
        
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
            chunks.extend(split_by_size(part, max_chars))
    
    return pack(chunks, max_chars)

def split_line(line: str, max_chars: int) -> list[str]:
    if len(line) <= max_chars:
        return [line]

    pices: list[str] = []
    for setence in SENTENCE.split(line):
        if len(setence) <= max_chars:
            pices.append(setence)
        else:
            pices.extend(pack(setence.split(" "), max_chars, sep= " "))
    
    return pack(pices, max_chars, sep=" ")

def pack(units: list[str], max_chars: int, sep: str="\n") -> list[str]:
    chunks: list[str] = []
    current = ""
    
    for unit in units:
        if current and len(current) + len(sep) + len(unit) > max_chars:
            chunks.append(current)
            current = ""
        
        current = f"{current}{sep}{unit}" if current else unit
        
    if current:
        chunks.append(current)
        
    return chunks
    
def split_by_size(text: str, max_chars: int) -> list[str]:
    units: list[str] = []
    
    for line in text.split("\n"):
        if line.strip():
            units.extend(split_line(line, max_chars))
            
    return pack(units, max_chars)

def build_context(norma: Norma, block: Block, text: str) -> str:
    header = [norma.title.strip()]
    
    if block.part != "corpo":
        header.append(block.part)
    if block.article:
        header.append(f"Art. {block.article}")
        
    return " > ".join(header) + "\n" + text

def chunk_norma(norma: Norma, max_chars: int =1500) -> list[Chunk]:
    chunks: list[Chunk] = []
    
    for block in split_blocks(norma.text):
        for text in split_long(block, max_chars):
            chunks.append(
                Chunk(
                    chunk_id=f"{norma.id}:{len(chunks)}",
                    norma_id=norma.id,
                    type=norma.type,
                    number=norma.number,
                    title=norma.title.strip(),
                    date=norma.date,
                    repealed=norma.repealed,
                    article=block.article,
                    part=block.part,
                    text=text,
                    context=build_context(norma, block, text)
                )
            )
            
    return chunks

def chunk_all(in_path: Path, out_path: Path, max_chars: int = 1500) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    
    with out_path.open("w", encoding="utf-8") as f:
        for line in in_path.read_text(encoding="utf-8").splitlines():
            norma = Norma.model_validate_json(line)
            for chunk in chunk_norma(norma, max_chars):
                f.write(chunk.model_dump_json() + "\n")
                total += 1
    
    print(f"Saved Total: {total}")

def read_chunks(chunk_size: int) -> Iterator[list[Chunk]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size deve ser maior que zero")
    
    with open(Path("data/processed/chunks.jsonl"), "r", encoding="utf-8") as file:
        while lines := list(islice(file, chunk_size)):
            yield [
                Chunk.model_validate_json(line)
                for line in lines
                if line.strip()
            ]
           
if __name__ == "__main__":
    chunk_all(Path("data/interim/normas.jsonl"), Path("data/processed/chunks.jsonl"))