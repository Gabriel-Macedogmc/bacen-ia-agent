import json
from pathlib import Path
import re
import time
import unicodedata
from pydantic import BaseModel
import httpx

class NormaRef(BaseModel):
    tipo: str
    numero: str
       
class BacenClient:
    def __init__(self, timeout: float = 30, delay: float = 1.0, max_retries: int = 3):
        self.timeout = timeout
        self.delay = delay
        self.max_retries = max_retries
        self.client = httpx.Client(timeout=self.timeout)

    def get_norma(self) -> list[NormaRef]:
        params = {
            "querytext": 'ContentType:normativo AND contentSource:normativos AND (pix OR Open Finance OR resoluções OR circulares)',
            "rowlimit": 500,
            "startrow": 0,
            "sortlist": "Data1OWSDATE:descending",
            "refinementfilters": "Data:range(datetime(min),datetime(max))",
        }
        
        requests = 0
        normaRef = []
        retries = 0
        
        while (True):
            response = self.client.get("https://www.bcb.gov.br/api/search/app/normativos/buscanormativos", 
                                               params=params, headers={"Accept": "application/json"})
            requests += 1
            
            if(response.status_code != 200):
                self.retry_request(self.delay, retries, requests, response.status_code)
                retries += 1
                continue
                
            params["startrow"] = int(params["startrow"]) + int(params["rowlimit"])
            
            normaRef.extend([NormaRef(tipo=item["TipodoNormativoOWSCHCS"], numero=item["NumeroOWSNMBR"]) for item in response.json().get("Rows", [])])
            
            print(f"Request {requests} - Retrieved {len(normaRef)} normas")
            
            if(response.json().get("TotalRows") == len(normaRef)):
                break
            
        print(f"Total normas retrieved: {len(normaRef)}")
        
        return normaRef
    
    def retry_request(self, delay: float, retries: int, requests: int, status_code: int):
        time.sleep(delay)    
                           
        print(f"Request {requests} failed with status code {status_code}. Retrying... ({self.max_retries} retries left)")
                        
        if retries == self.max_retries:
            raise Exception("Max retries exceeded")
        
    
    def get_norma_details(self, norma_ref: NormaRef, retries: int, request: int) -> dict | None:
        time.sleep(1)
        if(norma_ref.tipo == "Comunicado"):
            response = self.client.get("https://www.bcb.gov.br/api/conteudo/app/normativos/exibeoutrasnormas/", params={"p1": norma_ref.tipo, "p2": norma_ref.numero})
        else: 
            response = self.client.get("https://www.bcb.gov.br/api/conteudo/app/normativos/exibenormativo", params={"p1": norma_ref.tipo, "p2": norma_ref.numero})
        
        request += 1
        
        if(response.status_code != 200):
            self.retry_request(delay= self.delay, retries=retries, requests=request, status_code=response.status_code)
            retries += 1
        
        conteudo = response.json().get("conteudo", [])

        if not conteudo:
            return None

        content = conteudo[0]
        
        return content

def raw_path(ref: NormaRef) -> str:
    ascii_text = unicodedata.normalize("NFKD", f"`{ref.tipo}_${ref.numero}").encode("ascii", "ignore").decode()
    text_clean = re.sub(r"[^a-z0-9]+", "-", ascii_text.lower()).strip("-")
    return f"{text_clean}.json"

def download_norma(refs: list[NormaRef], client: BacenClient, out_dir: Path, force: bool = False) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    details = []
    retries = 0
    request = 0
    
    for ref in refs:
        out_path = out_dir / raw_path(ref)
        
        if not force and out_path.exists():
            print(f"Skipping {ref.tipo} {ref.numero}, file already exists.")
            continue
        
        response = client.get_norma_details(ref, retries, request)
        
        if not response:
            print(f"Skipping {ref.tipo} {ref.numero}, no details found.")
            continue
        
        details.append(response)
        
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(response, ensure_ascii=False, indent=2))
        print(f"Downloaded {ref.tipo} {ref.numero} to {out_path}")
    
    print(f"Downloaded {len(details)} normas") 

if __name__ == "__main__":
        client = BacenClient(delay=2.0)
        norma_refs = client.get_norma()
        download_norma(norma_refs, client, Path("data/raw"), force=False)