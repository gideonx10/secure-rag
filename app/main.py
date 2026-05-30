# app/main.py
# PURPOSE: FastAPI server with rate limiting, CORS, static frontend, and input validation

import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from app.rag import query_rag

# ── Logging ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ── Rate limiter ──────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="Secure RAG API")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── CORS ──────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://localhost:8000",
        "null",
        "https://secure-rag-app-adi.azurewebsites.net",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# ── Serve frontend static files ───────────────────────────────────────
# Serves index.html at / and static assets at /frontend/
import os
if os.path.exists("frontend"):
    app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

# ── Schema ────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer:  str
    sources: list[str]
    blocked: bool      = False
    flags:   list[str] = []

# ── Routes ────────────────────────────────────────────────────────────
@app.get("/")
def serve_frontend():
    """Serve the frontend UI at root URL."""
    if os.path.exists("frontend/index.html"):
        return FileResponse("frontend/index.html")
    return {"message": "Secure RAG API is running. Frontend not found."}

@app.post("/query", response_model=QueryResponse)
@limiter.limit("10/minute")
def query(request: Request, body: QueryRequest):
    q = body.question.strip()

    if not q:
        raise HTTPException(status_code=400, detail="Empty query")
    if len(q) > 1000:
        raise HTTPException(status_code=400, detail="Query too long — max 1000 chars")

    logger.info(f"QUERY | ip={request.client.host} | q={q[:80]}")

    result = query_rag(q)

    logger.info(
        f"RESPONSE | blocked={result.get('blocked', False)} | "
        f"flags={result.get('flags', [])}"
    )

    return QueryResponse(
        answer  = result["answer"],
        sources = result["sources"],
        blocked = result.get("blocked", False),
        flags   = result.get("flags", [])
    )

@app.get("/health")
def health():
    return {"status": "ok"}