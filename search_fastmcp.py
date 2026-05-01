from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List

from fastapi_mcp import FastApiMCP

# ✅ import CORE functions (NOT tools)
from tools import (
    ingest_pdf,
    hybrid_search,
    keyword_search,
    doc_map
)

# ---------------------------------------------------
# FASTAPI APP
# ---------------------------------------------------

app = FastAPI(title="Hybrid RAG MCP Server")


# ---------------------------------------------------
# MODELS
# ---------------------------------------------------

class SearchRequest(BaseModel):
    query: str
    k: int = 5


class SearchResult(BaseModel):
    id: int
    text: str
    score: float
    source: str


# ===================================================
# 1. HYBRID SEARCH ROUTE
# ===================================================

@app.post("/search/", response_model=List[SearchResult])
async def hybrid_search_api(req: SearchRequest):

    results = hybrid_search(req.query, req.k)

    formatted = []

    for i, r in enumerate(results):
        formatted.append({
            "id": r["id"],
            "text": r["text"],
            "score": 1.0 / (i + 1),   # simple rank-based score
            "source": "Hybrid"
        })

    return formatted


# ===================================================
# 2. KEYWORD SEARCH ROUTE
# ===================================================

@app.post("/search/keyword", response_model=List[SearchResult])
async def keyword_search_api(req: SearchRequest):

    ids = keyword_search(req.query, req.k)

    formatted = []

    for i, doc_id in enumerate(ids):
        if doc_id in doc_map:
            formatted.append({
                "id": doc_id,
                "text": doc_map[doc_id]["text"],
                "score": 1.0 / (i + 1),
                "source": "Keyword"
            })

    return formatted


# ===================================================
# 3. PDF INGESTION ROUTE
# ===================================================

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):

    file_path = f"./temp_{file.filename}"

    with open(file_path, "wb") as f:
        f.write(await file.read())

    chunks = ingest_pdf(file_path)

    return {
        "message": "PDF ingested successfully",
        "chunks_added": chunks
    }


# ===================================================
# MCP WRAPPER (LATEST STYLE)
# ===================================================

mcp = FastApiMCP(
    app,
    name="HybridRAG-MCP",
    description="Hybrid RAG system with vector + BM25 + keyword search"
)

mcp.mount_http()


# ===================================================
# RUN SERVER
# ===================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8003)