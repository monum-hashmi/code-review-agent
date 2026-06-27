from typing import TypedDict, Annotated
import operator

from langgraph.graph import StateGraph, END

from src.pr_fetcher import PRData
from src.indexer import retrieve_context
from src.config import settings


# ── State ─────────────────────────────────────────────────────────────────────
# This TypedDict is the single object that flows through every node.
# Each node reads from it and writes back to it.

class ReviewState(TypedDict):
    pr_url: str
    pr_data: PRData | None
    retrieved_context: list[str]
    correctness_findings: list[str]
    style_findings: list[str]
    security_findings: list[str]
    final_review: str
    token_usage: int
    error: str | None


# ── Nodes ──────────────────────────────────────────────────────────────────────
# Each function is one node in the graph.
# They receive the full state and return a dict of fields to update.

def fetch_pr_node(state: ReviewState) -> dict:
    """
    Node 1: Fetch PR data from GitHub.
    Calls pr_fetcher.fetch_pr() and stores result in state.
    """
    from src.pr_fetcher import fetch_pr

    print(f"[fetch_pr] Fetching {state['pr_url']}")

    if state.get("token_usage", 0) > settings.max_tokens_budget:
        return {"error": "Token budget exceeded"}

    try:
        pr_data = fetch_pr(state["pr_url"])

        if pr_data.lines_changed > settings.max_lines_per_pr:
            return {
                "error": f"PR too large: {pr_data.lines_changed} lines changed. Max is {settings.max_lines_per_pr}."
            }

        return {"pr_data": pr_data, "error": None}

    except Exception as e:
        return {"error": str(e)}


def retrieve_context_node(state: ReviewState) -> dict:
    """
    Node 2: Retrieve relevant codebase context from ChromaDB.
    Uses the PR diff as the query so we get related code.
    """
    print("[retrieve_context] Querying ChromaDB...")

    pr_data = state["pr_data"]
    if not pr_data:
        return {"retrieved_context": []}

    # Use changed file names + first 500 chars of diff as query
    query = " ".join(pr_data.changed_files) + "\n" + pr_data.diff[:500]
    context = retrieve_context(query, k=5)

    print(f"[retrieve_context] Got {len(context)} chunks")
    return {"retrieved_context": context}


def correctness_node(state: ReviewState) -> dict:
    """
    Node 3a: Check for logic bugs, null checks, edge cases.
    Runs in parallel with style and security nodes.
    Built fully in Week 5 Day 30.
    """
    from src.agents.correctness import run as run_correctness
    print("[correctness] Running correctness agent...")
    findings = run_correctness(state["pr_data"], state["retrieved_context"])
    return {"correctness_findings": findings}


def style_node(state: ReviewState) -> dict:
    """
    Node 3b: Check for style consistency with existing codebase.
    Runs in parallel with correctness and security nodes.
    Built fully in Week 5 Day 31.
    """
    from src.agents.style import run as run_style
    print("[style] Running style agent...")
    findings = run_style(state["pr_data"], state["retrieved_context"])
    return {"style_findings": findings}


def security_node(state: ReviewState) -> dict:
    """
    Node 3c: Check for security issues — injection, secrets, unsafe patterns.
    Runs in parallel with correctness and style nodes.
    Built fully in Week 5 Day 31.
    """
    from src.agents.security import run as run_security
    print("[security] Running security agent...")
    findings = run_security(state["pr_data"], state["retrieved_context"])
    return {"security_findings": findings}


def synthesize_node(state: ReviewState) -> dict:
    """
    Node 4: Combine all findings into one clean markdown review.
    Deduplicates, ranks by severity, formats for GitHub comment.
    Built fully in Week 5 Day 32.
    """
    print("[synthesize] Combining findings...")

    all_findings = (
        state.get("correctness_findings", []) +
        state.get("style_findings", []) +
        state.get("security_findings", [])
    )

    if not all_findings:
        review = "✅ No issues found. LGTM!"
    else:
        lines = ["## 🤖 Automated Code Review\n"]
        for i, finding in enumerate(all_findings, 1):
            lines.append(f"{i}. {finding}")
        review = "\n".join(lines)

    return {"final_review": review}


# ── Router ─────────────────────────────────────────────────────────────────────

def should_continue(state: ReviewState) -> str:
    """
    After fetch_pr_node, check if we hit an error.
    If yes, go straight to END. If no, continue to retrieve_context.
    """
    if state.get("error"):
        return "end"
    return "continue"


# ── Build the Graph ────────────────────────────────────────────────────────────

def build_graph():
    graph = StateGraph(ReviewState)

    # Add all nodes
    graph.add_node("fetch_pr", fetch_pr_node)
    graph.add_node("retrieve_context", retrieve_context_node)
    graph.add_node("correctness", correctness_node)
    graph.add_node("style", style_node)
    graph.add_node("security", security_node)
    graph.add_node("synthesize", synthesize_node)

    # Entry point
    graph.set_entry_point("fetch_pr")

    # After fetch_pr — check for errors first
    graph.add_conditional_edges(
        "fetch_pr",
        should_continue,
        {"continue": "retrieve_context", "end": END}
    )

    # After retrieve_context — fan out to 3 agents in parallel
    graph.add_edge("retrieve_context", "correctness")
    graph.add_edge("retrieve_context", "style")
    graph.add_edge("retrieve_context", "security")

    # All 3 agents feed into synthesize
    graph.add_edge("correctness", "synthesize")
    graph.add_edge("style", "synthesize")
    graph.add_edge("security", "synthesize")

    # Synthesize is the final node
    graph.add_edge("synthesize", END)

    return graph.compile()


# Compile once at import time
review_graph = build_graph()