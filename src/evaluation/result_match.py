import re
from datetime import datetime
from decimal import Decimal

# referências legais têm números que não são resposta: "art. 3", "nº 19", "§ 4º", "inciso I"
_LEGAL = re.compile(r"(art(igo)?\.?|n[ºo°]\.?|§|inciso|resolu[çc][ãa]o( bcb| cmn)?( n[ºo°])?)\s*\d+[\w-]*", re.IGNORECASE)
# 1.287,45 | 1287,45 | 2.337 | 540 | 1287.45
_NUM = re.compile(r"\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d+(?:[.,]\d+)?")

def parse_numbers(text: str) -> list[Decimal]:
    clean = _LEGAL.sub("", text)
    out = []
    
    for raw in _NUM.findall(clean):
        if "," in raw:
            raw = raw.replace(".", "").replace(",", ".")
        elif re.fullmatch(r"\d{1,3}(\.\d{3})+", raw):   # "2.337" sem vírgula = milhar
            raw = raw.replace(".", "")
        out.append(Decimal(raw))
        
    return out

MOUNTHS = ["janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho",
         "agosto", "setembro", "outubro", "novembro", "dezembro"]

def _date_hit(text: str, expected: str) -> bool:
    dt = datetime.fromisoformat(expected)
    t = text.lower()
    
    day = {f"{dt:%d/%m}", f"{dt.day}/{dt.month:02d}", f"{dt.day} de {MOUNTHS[dt.month - 1]}"}
    hour = {f"{dt:%Hh%M}", f"{dt:%H:%M}", f"{dt.hour}h{dt:%M}"}
    
    return any(d in t for d in day) and any(h in t for h in hour)

def result_hit(answer: str, found: bool, expected, forbidden: list[str]) -> bool | None:
    if expected is None:
        return None
    if not found:
        return False
    if any(f.lower() in answer.lower() for f in forbidden):
        return False
    if isinstance(expected, list):
        return all(item.lower() in answer.lower() for item in expected)
    if isinstance(expected, str):
        return _date_hit(answer, expected)
    
    target = Decimal(str(expected))
    return any(abs(n - target) <= Decimal("0.01") for n in parse_numbers(answer))