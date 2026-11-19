import json
from re import search
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
from fastapi import FastAPI, File, HTTPException, UploadFile
from ddgs import DDGS
from fastapi.responses import JSONResponse, FileResponse
from starlette.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pypdf import PdfReader

OLLAMA_URL = "http://localhost:11434"
CHAT_MODEL = "llama3.2"
EMBED_MODEL = "nomic-embed-text"
MAX_TOOL_ROUNDS = 3
MAX_UPLOAD_MB = 20
ALOWWED_EXT = [".pdf", ".txt", ".csv", ".json", ".md"]

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title = "Document Q&A")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

DOCS: dict[str, dict] = {}

@app.exception_handler(requests.RequestException)
async def ollama_down(_,exc):
    return JSONResponse(
        status_code=503,
        content = {"detail":f"Could not connect to Ollama at {OLLAMA_URL}. Please ensure Ollama is running and accessible."}
    )

#-----Document preocessing------

def extract_text(filename:str,data : bytes) ->str:
    if filename.lower().endswith(".pdf"):
        reader = PdfReader(io_bytes(data))
        return "\n".join(page.extract_text() for page in reader.pages)
    return data.decode("utf-8", errors="replace")

def io_bytes(data: bytes):
    import io
    return io.BytesIO(data)

def chunk_text(text:str,size :int = 800, overlap:int = 150) -> list[str]:
    chunks,start = [],0
    while start < len(text):
        chunks.append(text[start:start+size])
        start += size - overlap
    return [c.strip() for c in chunks if c.strip()]

#------Embediing and Retrievals------

def embed(text: list[str]) -> np.ndarray:
    vectors: list[list[float]] =  []
    for i in range(0, len(text),64):
        r = requests.post(
            f"{OLLAMA_URL}/api/embed",
            json = {
                "model": EMBED_MODEL,
                "input": text[i:i+64]
            }
        )
        r.raise_for_status()
        vectors.extend(r.json()["embeddings"])
    arr = np.array(vectors, dtype=np.float32)
    return arr / np.linalg.norm(arr, axis=1, keepdims=True)


def retrieve(query:str, docs: dict, k: int)-> list[tuple[int,float,str]]:
    q = embed([query])[0]
    scores = docs["vectors"] @ q
    top = np.argsort(scores)[::-1][:k]
    return [(int(i), float(scores[i]), docs["chunks"][i]) for i in top]

#----- Web Search Tool-----

def web_search(query:str, max_results:int = 5)->list[dict]:
    try:
        with DDGS() as ddgs:
            return [
                {
                    "title": result["title"],
                    "url": result["href"],
                    "snippet": result.get("body", "")
                }
                for result in ddgs.text(query, max_results=max_results)
            ]
    except Exception as e:
        return [{"error":f"Web search failed: {e}"}]

WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Perform a web search and return the top results.",
        "parameters":{
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query."
                    }},
            "required": ["query"],
            
        },
    },
}


def chat_with_tools(messages: list[dict], use_web: bool):
    """Run Ollama chat loop ,executing web_search calls as the model requests them"""
    searches: list[str] = []
    web_sources:dict[str, dict] = {}
    answer =""
    for _ in range(MAX_TOOL_ROUNDS):
        payload = {"model":CHAT_MODEL, "messages": messages,"stream": False}
        if use_web:
            payload["tools"] = [WEB_SEARCH_TOOL]

        r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout = 600)
        r.raise_for_status()
        msg = r.json()["message"]
        messages.append(msg)
        answer = msg.get("content", "")

        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            break
        for call in tool_calls:
            fn = call["function"]
            if fn["name"] != "web_search":
                continue
            args = fn.get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args)
            query = args.get("query", "")
            searches.append(query)

            results = web_search(query)
            for res in results:
                if "url" in res:
                    web_sources.setdefault(res["url"], res)
            messages.append({
                "role": "tool",
                "content": json.dumps(results)
            })
    return answer or "No answer generated.", searches, list(web_sources.values())


#------API Routes-----

def doc_summary(doc_id:str)-> dict:
    d = DOCS[doc_id]
    return {
        "id": doc_id,
        "name": d["name"],
        "chunk_count": len(d["chunks"]),
        "chars":len(d["text"]),
        "uploaded_at": d["uploaded_at"],
    }


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")

@app.post("/api/documents")
def upload_document(file: UploadFile = File(...)):
    name = file.filename or "untitled"
    ext = Path(name).suffix.lower()
    if ext not in ALOWWED_EXT:
        raise HTTPException(status_code=400, detail=f"File type {ext} not allowed. Allowed types: {ALOWWED_EXT}")

    data = file.file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File size exceeds {MAX_UPLOAD_MB} MB limit.")

    text = extract_text(name,data)
    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(status_code=422, detail="No text could be extracted from the document.")

    doc_id = uuid.uuid4().hex[:12]
    DOCS[doc_id] = {
        "name": name,
        "text": text,
        "chunks": chunks,
        "vectors": embed(chunks),
        "uploaded_at": datetime.now(timezone.utc).isoformat(timespec ="seconds")
    }
    return doc_summary(doc_id)


@app.get("/api/documents")
def list_documents():
    return [doc_summary(doc_id) for doc_id in DOCS]

@app.get("/api/documents/{doc_id}")
def get_document(doc_id:str):
    if doc_id not in DOCS:
        raise HTTPException(status_code=404, detail="Document not found.")
    return doc_summary(doc_id)

@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id:str):
    if DOCS.pop(doc_id, None) is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    return {"deleted": doc_id}

class AskRequest(BaseModel):
    doc_id: str
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=10)
    use_web: bool = True


@app.post("/api/ask")
def ask(req: AskRequest):
    doc = DOCS.get(req.doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    hits = retrieve(req.question, doc, req.top_k)
    context = "\n\n---\n\n".join(
        f"[Excerpt {n+1}]\n{text}" for n, (_,_, text) in enumerate(hits)
    )

    if req.use_web:
        system_prompt = (
            "You Answer questions using the document excerpts first. Cite excerpts as [Excerpt N]."
            "If the Excerpts do not contain the answer, perform a web search and cite sources as [Web Source N]."
            "facts, call the web_search tool to find relevant information. If you cannot find an answer, respond with 'I don't know.'"
            "claims, and provide a summary of the answer based on the information available."
        )
    else:
        system_prompt = (
            "You Answer questions using the document excerpts first. Cite excerpts as [Excerpt N]."
            "If the Excerpts do not contain the answer, respond with 'I don't know.'"
        )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Document Excerpts:\n\n{context}\n\nQuestion: {req.question}"}
    ]

    answer, searches, web_sources = chat_with_tools(messages, req.use_web)

    return{
        "answer": answer,
        "searches": searches,
        "doc_sources":[
            {"excerpt": n+1, "score": round(score, 4), "text": text}
            for n, (_, score, text) in enumerate(hits)
        ],
        "web_sources": web_sources,
    }