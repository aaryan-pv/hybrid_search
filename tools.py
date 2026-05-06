import numpy as np
import lancedb
import sqlite3
import requests
import yake
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

from tavily import TavilyClient
import os
from langchain_core.tools import tool
from typing import List, Dict
import feedparser
from dotenv import load_dotenv
load_dotenv()

# API keys from environment
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
 

tavily = TavilyClient(api_key=TAVILY_API_KEY)

model = SentenceTransformer("all-MiniLM-L6-v2")

db = lancedb.connect("./lancedb")
table = None

documents = []
doc_map = {}
bm25 = None

conn = sqlite3.connect("keywords.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS chunk_keywords (
    chunk_id INTEGER,
    keyword TEXT
)
""")

cursor.execute("CREATE INDEX IF NOT EXISTS idx_keyword ON chunk_keywords(keyword)")

kw_extractor = yake.KeywordExtractor(lan="en", n=2, top=5)

def load_pdf(file_path: str):
    try:
        reader = PdfReader(file_path)
        try:
            return " ".join([page.extract_text() or "" for page in reader.pages])
        except Exception:
            return ""
    except Exception:
        return ""


def chunk_text(text, chunk_size=300, overlap=50):
    try:
        words = text.split()
        try:
            return [
                " ".join(words[i:i + chunk_size])
                for i in range(0, len(words), chunk_size - overlap)
            ]
        except Exception:
            return []
    except Exception:
        return []

def ingest_pdf(file_path: str):
    global documents, bm25, table, doc_map

    try:
        text = load_pdf(file_path)
    except Exception:
        text = ""

    try:
        chunks = chunk_text(text)
    except Exception:
        chunks = []

    new_docs = []
    start_id = len(documents) + 1

    for i, chunk in enumerate(chunks):
        doc_id = start_id + i

        new_docs.append({
            "id": doc_id,
            "text": chunk
        })

        # keyword extraction
        try:
            keywords = kw_extractor.extract_keywords(chunk)
            for kw, _ in keywords:
                try:
                    cursor.execute(
                        "INSERT INTO chunk_keywords VALUES (?, ?)",
                        (doc_id, kw.lower())
                    )
                except Exception:
                    continue
        except Exception:
            pass

    try:
        conn.commit()
    except Exception:
        pass

    try:
        documents.extend(new_docs)
    except Exception:
        pass

    try:
        for d in new_docs:
            doc_map[d["id"]] = d
    except Exception:
        pass

    # BM25
    try:
        tokenized = [d["text"].split() for d in documents]
        bm25 = BM25Okapi(tokenized)
    except Exception:
        bm25 = None

    # embeddings
    try:
        embeddings = model.encode([d["text"] for d in new_docs]).tolist()
    except Exception:
        embeddings = []

    try:
        rows = [
            {"id": new_docs[i]["id"], "text": new_docs[i]["text"], "vector": embeddings[i]}
            for i in range(len(new_docs))
        ]
    except Exception:
        rows = []

    try:
        if table is None:
            table = db.create_table("docs", data=rows, mode="overwrite")
        else:
            table.add(rows)
    except Exception:
        pass

    return len(new_docs)



def keyword_search(query: str, k: int = 5):
    try:
        query_terms = query.lower().split()
    except Exception:
        query_terms = []

    scores = {}

    for term in query_terms:
        try:
            cursor.execute(
                "SELECT chunk_id FROM chunk_keywords WHERE keyword = ?",
                (term,)
            )
            try:
                results = cursor.fetchall()
                for (chunk_id,) in results:
                    try:
                        scores[chunk_id] = scores.get(chunk_id, 0) + 1
                    except Exception:
                        continue
            except Exception:
                continue
        except Exception:
            continue

    try:
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    except Exception:
        ranked = []

    try:
        return [doc_id for doc_id, _ in ranked[:k]]
    except Exception:
        return []

def reciprocal_rank_fusion(rank_lists, k=60):
    scores = {}
    try:
        for rank_list in rank_lists:
            try:
                for rank, doc_id in enumerate(rank_list):
                    try:
                        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
                    except Exception:
                        continue
            except Exception:
                continue
    except Exception:
        return []

    try:
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)
    except Exception:
        return []

def hybrid_search(query: str, k: int = 5):
    global bm25, table

    try:
        if bm25 is None or table is None:
            return []
    except Exception:
        return []

    # BM25
    try:
        tokenized_query = query.split()
        bm25_scores = bm25.get_scores(tokenized_query)
        bm25_top_k = np.argsort(bm25_scores)[::-1][:k]
        bm25_ids = [documents[idx]["id"] for idx in bm25_top_k]
    except Exception:
        bm25_ids = []

    # Vector
    try:
        query_embedding = model.encode(query).tolist()
        vector_results = table.search(query_embedding).limit(k).to_list()
        vector_ids = [r["id"] for r in vector_results]
    except Exception:
        vector_ids = []

    # Keyword
    try:
        keyword_ids = keyword_search(query, k)
    except Exception:
        keyword_ids = []

    # Fusion
    try:
        fused = reciprocal_rank_fusion([
            bm25_ids,
            vector_ids,
            keyword_ids
        ])
    except Exception:
        fused = []

    try:
        return [
            {
                "id": doc_id,
                "text": doc_map[doc_id]["text"]
            }
            for doc_id, _ in fused[:k]
        ]
    except Exception:
        return []

@tool
def finance_tool(query: str) -> str:
    """Finance domain tool using trusted sources only."""

    trusted_domains = [
        "rbi.org.in",
        "federalreserve.gov",
        "imf.org",
        "worldbank.org",
        "investopedia.com"
    ]

    try:
        results = tavily.search(
            query=query,
            max_results=5,
            include_domains=trusted_domains
        )
    except Exception:
        return ""

    output = []
    try:
        for r in results["results"]:
            try:
                output.append(f"{r['title']}\n{r['content']}\nSource: {r['url']}")
            except Exception:
                continue
    except Exception:
        return ""

    try:
        return "\n\n".join(output)
    except Exception:
        return ""
 
@tool
def web_search(query: str) -> List[Dict]:
    """Search the web for recent or unknown information."""

    try:
        response = tavily.search(query=query, max_results=3)
    except Exception:
        return []

    results = []

    try:
        for i, r in enumerate(response["results"]):
            try:
                results.append({
                    "source": "web",
                    "title": r.get("title", ""),
                    "content": r.get("content", ""),
                    "url": r.get("url", ""),
                    "score": 1.0 / (i + 1)
                })
            except Exception:
                continue
    except Exception:
        return []

    try:
        return results
    except Exception:
        return []

@tool
def news_tool(query: str, k: int = 3) -> List[Dict]:
    """Use for latest news, recent events, or current updates."""

    api_key ="NEWS_API_KEY"
    try:
        if not api_key:
            return [
                {
                    "source": "news",
                    "title": "News API key not configured",
                    "content": "Set NEWS_API_KEY in the environment before calling news_tool.",
                    "url": "",
                    "score": 0.0,
                }
            ]
    except Exception:
        return []

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "apiKey": api_key,
        "pageSize": k,
        "language": "en",
        "sortBy": "publishedAt",
    }

    try:
        response = requests.get(url, params=params, timeout=10)
    except Exception:
        return []

    try:
        if response.status_code != 200:
            try:
                error = response.json().get("message", "Unknown News API error")
            except Exception:
                error = "Unknown News API error"
            return [
                {
                    "source": "news",
                    "title": "News API request failed",
                    "content": error,
                    "url": "",
                    "score": 0.0,
                }
            ]
    except Exception:
        return []

    try:
        res = response.json()
    except Exception:
        return []

    try:
        articles = res.get("articles", [])[:k]
    except Exception:
        articles = []

    results = []
    try:
        for i, a in enumerate(articles):
            try:
                results.append({
                    "source": "news",
                    "title": a.get("title", ""),
                    "content": a.get("description", ""),
                    "url": a.get("url", ""),
                    "score": 1.0 / (i + 1)
                })
            except Exception:
                continue
    except Exception:
        return []

    try:
        return results
    except Exception:
        return []
    

@tool
def get_weather(city: str) -> Dict:
    """Get current weather of a city."""

    try:
        api_key = OPENWEATHER_API_KEY
    except Exception:
        return {}

    try:
        if not api_key:
            return {
                "source": "weather",
                "city": city,
                "error": "OpenWeather API key is missing. Set OPEN_WEATHER_API_KEY in the environment.",
                "score": 0.0
            }
    except Exception:
        return {}

    url = (
        f"http://api.openweathermap.org/data/2.5/weather"
        f"?q={city}&appid={api_key}&units=metric"
    )

    try:
        response = requests.get(url)
    except Exception:
        return {}

    try:
        data = response.json()
    except Exception:
        data = {}

    try:
        if response.status_code != 200:
            try:
                error_msg = data.get("message", "Could not fetch weather")
            except Exception:
                error_msg = "Could not fetch weather"
            return {
                "source": "weather",
                "city": city,
                "error": error_msg,
                "score": 0.0
            }
    except Exception:
        return {}

    try:
        return {
            "source": "weather",
            "city": city,
            "temperature": data["main"]["temp"],
            "description": data["weather"][0]["description"],
            "humidity": data["main"]["humidity"],
            "score": 1.0
        }
    except Exception:
        return {}
    

@tool
def medical_tool(query: str) -> str:
    """Medical domain tool using trusted health sources."""

    trusted_domains = [
        "who.int",
        "cdc.gov",
        "nih.gov",
        "mayoclinic.org"
    ]

    try:
        results = tavily.search(
            query=query,
            max_results=5,
            include_domains=trusted_domains
        )
    except Exception:
        return ""

    output = []

    try:
        for r in results["results"]:
            try:
                output.append(f"{r['title']}\n{r['content']}\nSource: {r['url']}")
            except Exception:
                continue
    except Exception:
        return ""

    try:
        return "\n\n".join(output)
    except Exception:
        return ""
    

@tool
def newsletter_tool(topic: str) -> List[Dict]:
    """
    Use for curated insights and summaries from high-quality AI/tech newsletters.
    Best for trends, not breaking news.
    """

    sources = {
        "tldr_ai": "https://tldr.tech/ai/rss",
        "ai_weekly": "https://aiweekly.co/rss",
    }

    results = []

    try:
        for name, url in sources.items():
            try:
                feed = feedparser.parse(url)
            except Exception:
                continue

            try:
                for entry in feed.entries[:2]:
                    try:
                        results.append({
                            "source": name,
                            "title": entry.title,
                            "summary": entry.summary,
                            "topic": topic
                        })
                    except Exception:
                        continue
            except Exception:
                continue
    except Exception:
        return []

    try:
        return results
    except Exception:
        return []
    


@tool
def ingest_pdf_tool(file_path: str) -> str:
    """Load and index a PDF file."""
    try:
        count = ingest_pdf(file_path)
        return f"Ingested {count} chunks."
    except Exception:
        return "PDF ingestion failed."
 

@tool
def keyword_search_tool(query: str) -> str:
    """
    Retrieve document chunks using keyword-based SQLite search.
    """

    try:
        chunk_ids = keyword_search(query)
    except Exception:
        return "Keyword search failed."

    try:
        if not chunk_ids:
            return "No keyword matches found."
    except Exception:
        return "No keyword matches found."

    results = []

    try:
        for cid in chunk_ids:
            try:
                if cid in doc_map:
                    results.append(f"[{cid}] {doc_map[cid]['text'][:300]}")
            except Exception:
                continue
    except Exception:
        return "Keyword search failed."

    try:
        return "\n\n".join(results)
    except Exception:
        return "Keyword search failed."


@tool
def search_tool(query: str) -> str:
    """Search relevant chunks from indexed documents."""

    try:
        results = hybrid_search(query)
    except Exception:
        return "Search failed due to an internal error."

    try:
        if not results:
            return "No documents found. Please ingest a PDF first."
    except Exception:
        return "Search failed due to an internal error."

    try:
        return "\n\n".join([
            f"[{r['id']}] {r['text'][:300]}"
            for r in results
        ])
    except Exception:
        return "Search failed due to an internal error."