from typing import TypedDict

from langgraph.graph import StateGraph, END

from src.pr_fetcher import PRData
from src.indexer import retrieve_context
from src.config import settings


# ── State ─────────────────────────────────────────────────────────────────────

class ReviewState(TypedDict):
    pr_url: str
    pr_data: PRData | None
    retrieved_context: list[str]
    correctness_findings: list[dict]     # changed from list[str] to list[dict]
    style_findings: list[dict]
    security_findings: list[dict]
    final_review: str
    token_usage: int
    error: str | None


# ── Nodes ──────────────────────────────────────────────────────────────────────

def fetch_pr_node(state: ReviewState) -> dict:
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
    print("[retrieve_context] Querying ChromaDB...")

    pr_data = state["pr_data"]
    if not pr_data:
        return {"retrieved_context": []}

    query = " ".join(pr_data.changed_files) + "\n" + pr_data.diff[:500]
    context = retrieve_context(query, k=5)

    print(f"[retrieve_context] Got {len(context)} chunks")
    return {"retrieved_context": context}


def correctness_node(state: ReviewState) -> dict:
    from src.agents.correctness import run as run_correctness
    print("[correctness] Running correctness agent...")
    findings = run_correctness(state["pr_data"], state["retrieved_context"])
    return {"correctness_findings": findings}


def style_node(state: ReviewState) -> dict:
    from src.agents.style import run as run_style
    print("[style] Running style agent...")
    findings = run_style(state["pr_data"], state["retrieved_context"])
    return {"style_findings": findings}


def security_node(state: ReviewState) -> dict:
    from src.agents.security import run as run_security
    print("[security] Running security agent...")
    findings = run_security(state["pr_data"], state["retrieved_context"])
    return {"security_findings": findings}


def synthesize_node(state: ReviewState) -> dict:
    """Combine all findings into a structured markdown review, grouped by severity."""
    print("[synthesize] Combining findings...")

    cor = state.get("correctness_findings", [])
    sty = state.get("style_findings", [])
    sec = state.get("security_findings", [])

    if not cor and not sty and not sec:
        return {"final_review": "✅ No issues found. LGTM!"}

    lines = ["## 🤖 Automated Code Review\n"]

    # Security first (highest priority), then correctness, then style
    sections = [
        ("🔴 Security", sec),
        ("🟠 Correctness", cor),
        ("🟡 Style", sty),
    ]

    for label, findings in sections:
        if not findings:
            continue
        lines.append(f"### {label} ({len(findings)} issue{'s' if len(findings) != 1 else ''})\n")
        for f in findings:
            if isinstance(f, dict):
                sev = f.get("severity", "medium").upper()
                issue = f.get("issue", "")
                file = f.get("file", "")
                line = f.get("line", "")
                suggestion = f.get("suggestion", "")
                lines.append(f"- **[{sev}]** `{file}`")
                if line:
                    lines.append(f"  Code: `{line}`")
                lines.append(f"  Issue: {issue}")
                if suggestion:
                    lines.append(f"  Fix: {suggestion}")
                lines.append("")
            else:
                lines.append(f"- {f}\n")

    total = len(cor) + len(sty) + len(sec)
    lines.append(f"---\n*{total} issue{'s' if total != 1 else ''} found across {len(state['pr_data'].changed_files)} files.*")

    return {"final_review": "\n".join(lines)}


# ── Router ─────────────────────────────────────────────────────────────────────

def should_continue(state: ReviewState) -> str:
    if state.get("error"):
        return "end"
    return "continue"


# ── Build the Graph ────────────────────────────────────────────────────────────

def build_graph():
    graph = StateGraph(ReviewState)

    graph.add_node("fetch_pr", fetch_pr_node)
    graph.add_node("retrieve_context", retrieve_context_node)
    graph.add_node("correctness", correctness_node)
    graph.add_node("style", style_node)
    graph.add_node("security", security_node)
    graph.add_node("synthesize", synthesize_node)

    graph.set_entry_point("fetch_pr")

    graph.add_conditional_edges(
        "fetch_pr",
        should_continue,
        {"continue": "retrieve_context", "end": END}
    )

    graph.add_edge("retrieve_context", "correctness")
    graph.add_edge("retrieve_context", "style")
    graph.add_edge("retrieve_context", "security")

    graph.add_edge("correctness", "synthesize")
    graph.add_edge("style", "synthesize")
    graph.add_edge("security", "synthesize")

    graph.add_edge("synthesize", END)

    return graph.compile()


review_graph = build_graph()