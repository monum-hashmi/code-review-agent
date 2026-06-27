import hmac
import hashlib
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, HTTPException, Header, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl

from src.config import settings
from src.graph import review_graph

logger = structlog.get_logger()


# ── Request / Response Schemas ─────────────────────────────────────────────────

class ReviewRequest(BaseModel):
    pr_url: str
    index_repo: bool = False  # if True, index the repo before reviewing


class ReviewResponse(BaseModel):
    pr_url: str
    final_review: str
    correctness_findings: list[str]
    style_findings: list[str]
    security_findings: list[str]
    error: str | None = None


class IndexRequest(BaseModel):
    repo_url: str  # e.g. https://github.com/owner/repo


class IndexResponse(BaseModel):
    repo_url: str
    chunks_stored: int


# ── App Setup ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Code Review Agent",
    description="Multi-agent LangGraph system for automated PR code review",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Auth ───────────────────────────────────────────────────────────────────────

def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Health check — first thing you hit after EC2 deployment."""
    return {"status": "ok", "model": settings.model_name}


@app.post("/review", response_model=ReviewResponse)
def review_pr(
    request: ReviewRequest,
    api_key: str = Depends(verify_api_key),
):
    """
    Main endpoint. Takes a GitHub PR URL, runs the full
    LangGraph pipeline, returns structured review findings.
    """
    logger.info("review_requested", pr_url=request.pr_url)

    # Optionally index the repo first
    if request.index_repo:
        from src.indexer import index_repo
        repo_url = _pr_url_to_repo_url(request.pr_url)
        index_repo(repo_url)

    # Run the LangGraph pipeline
    initial_state = {
        "pr_url": request.pr_url,
        "pr_data": None,
        "retrieved_context": [],
        "correctness_findings": [],
        "style_findings": [],
        "security_findings": [],
        "final_review": "",
        "token_usage": 0,
        "error": None,
    }

    result = review_graph.invoke(initial_state)

    if result.get("error"):
        logger.error("review_failed", error=result["error"])
        raise HTTPException(status_code=400, detail=result["error"])

    logger.info("review_complete", pr_url=request.pr_url)

    return ReviewResponse(
        pr_url=request.pr_url,
        final_review=result["final_review"],
        correctness_findings=result["correctness_findings"],
        style_findings=result["style_findings"],
        security_findings=result["security_findings"],
        error=result.get("error"),
    )


@app.post("/index", response_model=IndexResponse)
def index_repository(
    request: IndexRequest,
    api_key: str = Depends(verify_api_key),
):
    """
    Index a repo into ChromaDB so the review agents
    have codebase context when reviewing PRs.
    """
    from src.indexer import index_repo

    logger.info("index_requested", repo_url=request.repo_url)
    chunks = index_repo(request.repo_url)

    return IndexResponse(
        repo_url=request.repo_url,
        chunks_stored=chunks,
    )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _pr_url_to_repo_url(pr_url: str) -> str:
    """
    Converts https://github.com/owner/repo/pull/123
    to      https://github.com/owner/repo
    """
    parts = pr_url.split("/pull/")
    return parts[0]