from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.config import settings
from src.pr_fetcher import PRData


# ── Output Schema ──────────────────────────────────────────────────────────────

class CorrectnessFindings(BaseModel):
    findings: list[str] = Field(
        description="List of correctness issues found. Empty list if none."
    )


# ── LLM Setup ─────────────────────────────────────────────────────────────────

def _get_llm():
    return ChatOpenAI(
        model=settings.model_name,
        openai_api_key=settings.openrouter_api_key,
        openai_api_base=settings.openrouter_base_url,
        temperature=0,
    ).with_structured_output(CorrectnessFindings)


# ── Prompt ─────────────────────────────────────────────────────────────────────

PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a senior software engineer reviewing a pull request for correctness.
Your job is to find real bugs — not style issues, not security issues, just logic errors.

Look for:
- Logic bugs and wrong conditions
- Null/None checks that are missing
- Off-by-one errors
- Unhandled edge cases
- Incorrect error handling
- Functions that don't do what their name says

Be specific. Reference the exact file and line when possible.
If you find nothing, return an empty list. Do not invent issues."""),

    ("human", """PR Title: {title}
PR Description: {description}
Author: {author}

Changed Files:
{changed_files}

Diff:
{diff}

Relevant codebase context:
{context}

Find correctness issues only. Return a list of specific findings.""")
])


# ── Entry Point ────────────────────────────────────────────────────────────────

def run(pr_data: PRData, context: list[str]) -> list[str]:
    """
    Called by correctness_node in graph.py.
    Takes PR data + retrieved codebase context.
    Returns list of correctness issues as strings.
    """
    llm = _get_llm()
    chain = PROMPT | llm

    result = chain.invoke({
        "title": pr_data.title,
        "description": pr_data.description,
        "author": pr_data.author,
        "changed_files": "\n".join(pr_data.changed_files),
        "diff": pr_data.diff[:3000],  # cap to avoid token blowup
        "context": "\n---\n".join(context) if context else "No context available.",
    })

    return result.findings