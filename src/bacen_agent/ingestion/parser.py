import json
from pathlib import Path
import re
import unicodedata
from bs4 import BeautifulSoup
from pydantic import BaseModel

BLOCKS = ["p", "br", "div", "li", "tr", "h1", "h2", "h3", "h4"]
BOILERPLATE = [
    re.compile(r"este text não substitui o publicado.*", re.I),
]

class Norma(BaseModel):
    id: str
    title: str
    date: str
    date_text: str
    subject: str
    text: str
    repealed: bool
    updates: str
    status: str
    linked_standards: str
    
def read_files(raw_dir: str) -> list:
    folder = Path(raw_dir)
    
    if not folder.is_dir():
        raise FileNotFoundError(f"Diretorio nao encontrado ${folder.resolve()}")
    
    data_list = []
    
    for path in folder.glob("*.json"):
        try:
            with path.open("r", encoding="utf-8") as file:
                data = json.load(file)
                data["path"] = path
                data_list.append(data)
                
        except json.JSONDecodeError as exc:
            print(f"JSON invalido em ${path} ${exc}")
        
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
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    
    return text.strip()

def parse():
    files = read_files("./data/raw/")
    
    for file in files:
        text_html = file["Texto"] or ""
        text = remove_html_tags(text_html) if text_html else ""
        if not text:
            status = "revogada" if file["Revogado"] else "sem_texto"
        elif len(text) < 200:
            status = "suspeito"
        else:
            status = "ok"
    
    

if __name__ == "__main__":
    files = read_files("./data/raw/")
    
    for file in files:
        remove_html_tags(file["Texto"])
        