# SEarch_MCP

1. Overview

This document describes the design, implementation, and evaluation of a Hybrid Search Service integrated with a LangGraph Agent and exposed via a FastAPI MCP server.

The system enables:

PDF ingestion and chunking
Hybrid retrieval (Dense + Sparse + Keyword)
Agent-based querying using tools
External integration via MCP-compatible interfaces

2. System Architecture
2.1 High-Level Components
User → LangGraph Agent → Tools → Retrieval Layer → Storage
                                      ↓
                         (LanceDB + BM25 + SQLite)
                                      ↓
                               FastAPI MCP Server

## Repository Structure

- `app.py` - Agent orchestration using `langgraph` and a tool-enabled LLM loop.
- `tools.py` - Core indexing and search pipeline implementation, plus LangChain-style tool wrappers.
- `search_fastmcp.py` - FastAPI application entrypoint with REST routes and MCP mounting.
 

## Components

### `tools.py`

This module contains the primary search logic and ingestion pipeline.

#### Core functions

- `load_pdf(file_path: str)`
  - Reads a PDF using `pypdf` and returns the concatenated text for all pages.

- `chunk_text(text, chunk_size=300, overlap=50)`
  - Splits raw text into overlapping chunks suitable for indexing.

- `ingest_pdf(file_path: str)`
  - Ingests a PDF into the system.
  - Extracts chunked text.
  - Stores extracted keywords in SQLite for exact keyword matching.
  - Rebuilds BM25 indexes over all loaded documents.
  - Computes embeddings using `sentence_transformers` and stores them in `LanceDB`.

- `keyword_search(query: str, k: int = 5)`
  - Performs exact keyword retrieval using SQLite indexes.
  - Returns matching chunk IDs ranked by keyword frequency.

- `hybrid_search(query: str, k: int = 5)`
  - Executes a hybrid retrieval pipeline:
    - BM25 ranking over chunk text.
    - Vector search over LanceDB embeddings.
    - Keyword search fallback.
  - Applies reciprocal rank fusion to merge the three result lists.
  - Returns enriched document chunks for downstream consumption.

#### Tool wrappers

These wrappers expose the core search functions as LangChain-compatible tools.

- `ingest_pdf_tool(file_path: str) -> str`
  - Uses `ingest_pdf` to index a new PDF and returns a chunk ingestion summary.

- `keyword_search_tool(query: str) -> str`
  - Returns keyword-based results as formatted text snippets.

- `search_tool(query: str) -> str`
  - Executes the `hybrid_search` pipeline and returns best-matching chunks.

### `search_fastmcp.py`

This file defines the HTTP API for the hybrid retrieval service.

#### FastAPI routes

- `POST /search/`
  - Accepts `query` and optional `k`.
  - Uses `hybrid_search` and returns ranked search results.

- `POST /search/keyword`
  - Accepts `query` and optional `k`.
  - Uses `keyword_search` and returns exact keyword matches.

- `POST /upload-pdf/`
  - Accepts file uploads via multipart form data.
  - Writes the uploaded file locally and ingests it through `ingest_pdf`.

#### MCP integration

- `FastApiMCP` is used to mount the FastAPI app as an MCP service.
- The MCP wrapper is configured with the name `HybridRAG-MCP` and descriptive metadata.

### `app.py`

This module builds an interactive tool-enabled agent using `langgraph` and  Qwen3.5 4B AWQ 4bit model.  

- Compiles an agent graph and exposes a CLI loop for direct user interaction.
-The agent can take document from the user and can embed it, then the user can ask queries from it.


## Data and Storage

- `lancedb/` is the persistent storage directory for vector embeddings.
- `keywords.db` is a local SQLite database used for keyword indexing.
- Document metadata is held in-memory (`documents`, `doc_map`) but persisted across tool lifetime via the LanceDB store.
 
 