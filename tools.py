import numpy as np
import lancedb
import sqlite3
import yake

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from langchain.tools import tool

# ------------------------
# GLOBALS
# ------------------------

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


# ------------------------
# CORE FUNCTIONS
# ------------------------

def load_pdf(file_path: str):
    reader = PdfReader(file_path)
    return " ".join([page.extract_text() or "" for page in reader.pages])


def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    return [
        " ".join(words[i:i + chunk_size])
        for i in range(0, len(words), chunk_size - overlap)
    ]


def ingest_pdf(file_path: str):
    global documents, bm25, table, doc_map

    text = load_pdf(file_path)
    chunks = chunk_text(text)

    new_docs = []
    start_id = len(documents) + 1

    for i, chunk in enumerate(chunks):
        doc_id = start_id + i

        new_docs.append({
            "id": doc_id,
            "text": chunk
        })

        # keyword extraction
        keywords = kw_extractor.extract_keywords(chunk)
        for kw, _ in keywords:
            cursor.execute(
                "INSERT INTO chunk_keywords VALUES (?, ?)",
                (doc_id, kw.lower())
            )

    conn.commit()

    documents.extend(new_docs)
    for d in new_docs:
        doc_map[d["id"]] = d

    # BM25
    tokenized = [d["text"].split() for d in documents]
    bm25 = BM25Okapi(tokenized)

    # embeddings
    embeddings = model.encode([d["text"] for d in new_docs]).tolist()

    rows = [
        {"id": new_docs[i]["id"], "text": new_docs[i]["text"], "vector": embeddings[i]}
        for i in range(len(new_docs))
    ]

    if table is None:
        table = db.create_table("docs", data=rows, mode="overwrite")
    else:
        table.add(rows)

    return len(new_docs)


def keyword_search(query: str, k: int = 5):
    query_terms = query.lower().split()
    scores = {}

    for term in query_terms:
        cursor.execute(
            "SELECT chunk_id FROM chunk_keywords WHERE keyword = ?",
            (term,)
        )
        for (chunk_id,) in cursor.fetchall():
            scores[chunk_id] = scores.get(chunk_id, 0) + 1

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in ranked[:k]]


def reciprocal_rank_fusion(rank_lists, k=60):
    scores = {}
    for rank_list in rank_lists:
        for rank, doc_id in enumerate(rank_list):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def hybrid_search(query: str, k: int = 5):
    global bm25, table

    if bm25 is None or table is None:
        return []

    # BM25
    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_top_k = np.argsort(bm25_scores)[::-1][:k]
    bm25_ids = [documents[idx]["id"] for idx in bm25_top_k]

    # Vector
    query_embedding = model.encode(query).tolist()
    vector_results = table.search(query_embedding).limit(k).to_list()
    vector_ids = [r["id"] for r in vector_results]

    # Keyword
    keyword_ids = keyword_search(query, k)

    # Fusion
    fused = reciprocal_rank_fusion([
        bm25_ids,
        vector_ids,
        keyword_ids
    ])

    return [
        {
            "id": doc_id,
            "text": doc_map[doc_id]["text"]
        }
        for doc_id, _ in fused[:k]
    ]


# ------------------------
# TOOLS
# ------------------------

@tool
def ingest_pdf_tool(file_path: str) -> str:
    """Load and index a PDF file."""
    count = ingest_pdf(file_path)
    return f"Ingested {count} chunks."
 

@tool
def keyword_search_tool(query: str) -> str:
    """
    Retrieve document chunks using keyword-based SQLite search.
    """

    chunk_ids = keyword_search(query)

    if not chunk_ids:
        return "No keyword matches found."

    results = []
    for cid in chunk_ids:
        if cid in doc_map:
            results.append(f"[{cid}] {doc_map[cid]['text'][:300]}")

    return "\n\n".join(results)
@tool
def search_tool(query: str) -> str:
    """Search relevant chunks from indexed documents."""
    results = hybrid_search(query)

    if not results:
        return "No documents found. Please ingest a PDF first."

    return "\n\n".join([
        f"[{r['id']}] {r['text'][:300]}"
        for r in results
    ])