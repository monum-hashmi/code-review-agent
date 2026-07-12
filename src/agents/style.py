from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.config import settings
from src.pr_fetcher import PRData


class StyleFindings(BaseModel):
    findings: list[str] = Field(
        description="List of style issues found. Empty list if none."
    )


def _get_llm():
    return ChatOpenAI(
        model=settings.model_name,
        openai_api_key=settings.openrouter_api_key,
        openai_api_base=settings.openrouter_base_url,
        temperature=0,
    )


PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a senior software engineer reviewing a pull request for style consistency.
Your job is to find style issues only — not bugs, not security issues.

Look for:
- Naming inconsistencies with the existing codebase
- Missing or inconsistent docstrings
- Functions that are too long or do too many things
- Duplicate code that already exists in the codebase
- Inconsistent formatting patterns

Use the codebase context to judge what the existing style actually is.
If the PR matches existing style, return an empty list. Do not invent issues."""),

    ("human", """PR Title: {title}

Changed Files:
{changed_files}

Diff:
{diff}

Existing codebase style (for reference):
{context}

Find style issues only. You MUST respond with ONLY this JSON format, nothing else:
{{"findings": ["issue 1", "issue 2"]}}
If no issues found: {{"findings": []}}""")
])


def run(pr_data: PRData, context: list[str]) -> list[str]:
    import json, re
    llm = _get_llm()
    chain = PROMPT | llm

    result = chain.invoke({
        "title": pr_data.title,
        "changed_files": "\n".join(pr_data.changed_files),
        "diff": pr_data.diff[:3000],
        "context": "\n---\n".join(context) if context else "No context available.",
    })

    text = result.content
    text = re.sub(r"```json|```", "", text).strip()
    
    if not text:
        return []
    
    try:
        parsed = json.loads(text)
        return parsed.get("findings", [])
    except json.JSONDecodeError:
        # LLM returned plain text instead of JSON — extract lines as findings
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        return lines if lines else []