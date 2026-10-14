import json
from pathlib import Path
import re
from typing import TypedDict
import unicodedata
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field, field_validator

BLOCKS = ["p", "br", "div", "li", "tr", "h1", "h2", "h3", "h4"]

class NormaFile(TypedDict, total=False):
    Texto: str | None
    Revogado: bool
    NormasVinculadas: str | None
    Atualizacoes: str | None
    Data: str
    DataTexto: str
    Id: int
    Assunto: str
    Titulo: str
    Tipo: str
    Numero: int

class NormaVinculada(BaseModel):
    type: str
    number: str
    year: str

class Norma(BaseModel):
    id: int
    type: str
    number: int
    title: str
    date: str
    date_text: str
    subject: str 
    text: str 
    repealed: bool
    updates: str | None = Field(default=None)
    status: str
    linked_standards: list[NormaVinculada] = Field(default_factory=list)
    
    @field_validator("linked_standards", mode="before")
    @classmethod
    def split_linkeds(cls, value: str | None) -> list[dict]:
        if not value:
            return []
        
        if isinstance(value, list):
            return value

        itens = [item.split(";@") for item in value.split(";#") if item.strip()]

        return [{"type": t.strip().upper(), "number": n, "year": a} for t, n, a in itens]

def read_files(raw_dir: str) -> list[NormaFile]:
    folder = Path(raw_dir)
    
    if not folder.is_dir():
        raise FileNotFoundError(f"Diretorio nao encontrado {folder.resolve()}")
    
    data_list = []
    
    for path in folder.glob("*.json"):
        try:
            with path.open("r", encoding="utf-8") as file:
                data = json.load(file)
                data["path"] = path
                data_list.append(data)
                
        except json.JSONDecodeError as exc:
            print(f"JSON invalido em {path} {exc}")

    return data_list

def remove_html_tags(raw_text: str) -> str:
    soup = BeautifulSoup(raw_text, 'html.parser')
            
    for tag in soup(["script", "style", "title", "s"]):
        tag.decompose()
    
    for node in soup.find_all(string=True):
        node.replace_with(re.sub(r"\s+", " ", node))
    
    for tr in soup.find_all("tr"):
        celulas = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
        tr.replace_with("\n" + " | ".join(celulas) + "\n")
    
    for tag in soup.find_all(BLOCKS):
        tag.insert_before("\n")
  
    text = soup.get_text()
    text = unicodedata.normalize("NFC", text).replace("\xa0", " ")
    
    text = re.sub(r"[ \t]+", " ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    
    return clean_dots(text.strip())

DOTS = re.compile(r"[ \t]*\.{5,}[ \t]*")

def clean_dots(text: str) -> str:
    lines: list[str] = []
    
    for line in DOTS.sub(" [...] ", text).split("\n"):
        line = line.strip()
        
        if line == "[...]" and lines and lines[-1].endswith("[...]"):
            continue
        
        lines.append(line)
    
    return "\n".join(lines)

def parse(raw_dir: Path, out_path: Path) -> None:
    files = read_files(raw_dir)
    
    normas: list[Norma] = []

    for file in files:
        text_html = file.get("Texto") or ""
        
        text = remove_html_tags(text_html)
        subject = remove_html_tags(file.get("Assunto"))
        
        revoked = file.get("Revogado", False)

        if not text:
            status = "sem_texto"
        elif len(text) < 200:
            status = "suspeito"
        else:
            status = "ok"

        norma = Norma(
            type=file.get("Tipo").upper(),
            number=file.get("Numero"),
            status=status,
            text=text,
            date=file.get("Data"),
            date_text=file.get("DataTexto"),
            id=file.get("Id"),
            linked_standards=file.get("NormasVinculadas"),
            repealed=revoked,
            subject=subject,
            title=file.get("Titulo"),
            updates=file.get("Atualizacoes"),
        )
        normas.append(norma)
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for norma in normas:
            f.write(norma.model_dump_json() + "\n")
        print(f"Saved total: {len(normas)}")

if __name__ == "__main__":
    parse(Path("data/raw"), Path("data/interim/normas.jsonl"))    
    
        