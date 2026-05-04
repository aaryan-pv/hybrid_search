from fastapi import FastAPI, UploadFile, File
import feedparser
from pydantic import BaseModel
from typing import List, Optional

from fastapi_mcp import FastApiMCP
import requests

# ✅ import CORE functions (NOT tools)
from tools import (
    finance_tool,
    ingest_pdf,
    ingest_pdf_tool,
    hybrid_search,
    keyword_search,
    keyword_search_tool,
    medical_tool,
    search_tool,
    web_search,
    news_tool,
    newsletter_tool,
    get_weather,
    doc_map
)

# ---------------------------------------------------
# FASTAPI APP
# ---------------------------------------------------

app = FastAPI(title="Hybrid RAG MCP Server")


# ---------------------------------------------------
# MODELS
# ---------------------------------------------------
class NewsSearchRequest(BaseModel):
    query: str
    k: int = 3


class WeatherRequest(BaseModel):
    city: str


class NewsletterRequest(BaseModel):
    topic: str


class SearchRequest(BaseModel):
    query: str
    k: int = 5


class SearchResult(BaseModel):
    id: str
    text: str
    score: float
    source: str
    url: Optional[str] = None


class ToolTextRequest(BaseModel):
    query: str


class FileIngestRequest(BaseModel):
    file_path: str


class ToolResponse(BaseModel):
    result: str


# ===================================================
# 1. HYBRID SEARCH ROUTE
# ===================================================

@app.post("/search/", response_model=List[SearchResult])
async def hybrid_search_api(req: SearchRequest):

    results = hybrid_search(req.query, req.k)

    formatted = []

    for i, r in enumerate(results):
        formatted.append({
            "id": str(r["id"]),
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
                "id": str(doc_id),
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


@app.post("/tools/ingest_pdf", response_model=ToolResponse)
def ingest_pdf_tool_api(req: FileIngestRequest):
    return {"result": ingest_pdf_tool.invoke({"file_path": req.file_path})}


@app.post("/tools/search", response_model=ToolResponse)
def search_tool_api(req: ToolTextRequest):
    return {"result": search_tool.invoke({"query": req.query})}


@app.post("/tools/keyword_search", response_model=ToolResponse)
def keyword_search_tool_api(req: ToolTextRequest):
    return {"result": keyword_search_tool.invoke({"query": req.query})}


@app.post("/search/web", response_model=List[SearchResult])
def web_api(req: SearchRequest):
    results = web_search.invoke({"query": req.query})

    return [
        {
            "id": f"web_{i}",
            "text": f"{r['title']}\n{r['content']}",
            "url": r["url"],
            "score": r["score"],
            "source": r["source"]
        }
        for i, r in enumerate(results)
    ]


# =========================
# 📰 NEWS API
# =========================
@app.post("/search/news", response_model=List[SearchResult])
def news_api(req: SearchRequest):
    results = news_tool.invoke({"query": req.query, "k": req.k})

    return [
        {
            "id": f"news_{i}",
            "text": f"{r['title']}\n{r['content']}",
            "url": r.get("url", ""),
            "score": r["score"],
            "source": r["source"]
        }
        for i, r in enumerate(results)
    ]


# =========================
# 🌤️ WEATHER API
# =========================
@app.post("/weather", response_model=SearchResult)
def weather_api(req: WeatherRequest):
    result = get_weather.invoke({"city": req.city})

    if "error" in result:
        return {
            "id": "weather_error",
            "text": result["error"],
            "score": 0.0,
            "source": "weather"
        }

    return {
        "id": f"weather_{req.city}",
        "text": f"{result['temperature']}°C, {result['description']}, humidity {result['humidity']}",
        "score": result["score"],
        "source": "weather"
    }


# =========================
# 🧠 NEWSLETTER API
# =========================
@app.post("/search/newsletter", response_model=List[SearchResult])
def newsletter_api(req: NewsletterRequest):
    results = newsletter_tool.invoke({"topic": req.topic})

    return [
        {
            "id": f"newsletter_{i}",
            "text": f"{r['title']}\n{r['summary']}",
            "score": 1.0 / (i + 1),
            "source": r["source"],
            "url": r.get("url", "")
        }
        for i, r in enumerate(results)
    ]

class FinanceRequest(BaseModel):
    query: str


@app.post("/finance")
def finance_api(req: FinanceRequest):
    result = finance_tool.invoke(req.query)

    return {
        "id": f"finance_{hash(req.query)}",
        "text": result,
        "source": "finance_tool"
    }


class MedicalRequest(BaseModel):
    query: str


@app.post("/medical")
def medical_api(req: MedicalRequest):
    result = medical_tool.invoke(req.query)

    return {
        "id": f"medical_{hash(req.query)}",
        "text": result,
        "source": "medical_tool"
    }



# MCP WRAPPER (LATEST STYLE)
# ===================================================

mcp = FastApiMCP(
    app,
    name="HybridRAG-MCP",
    description="Hybrid RAG system with vector + BM25 + keyword search"
)

mcp.mount_http()
 

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8003)