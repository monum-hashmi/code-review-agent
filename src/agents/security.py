from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.config import settings
from src.pr_fetcher import PRData


class SecurityFindings(BaseModel):
    findings: list[str] = Field(
        description="List of security issues found. Empty list if none."
    )


def _get_llm():
    return ChatOpenAI(
        model=settings.model_name,
        openai_api_key=settings.openrouter_api_key,
        openai_api_base=settings.openrouter_base_url,
        temperature=0,
    )


PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a senior security engineer reviewing a pull request.
Your job is to find security vulnerabilities only — not bugs, not style issues.

Look for:
- SQL injection or prompt injection risks
- Hardcoded secrets, API keys, or passwords
- Missing input validation
- Unsafe deserialization
- Missing authentication or authorization checks
- Exposed sensitive data in logs or responses
- Path traversal vulnerabilities
- Use of unsafe functions

Be specific. Reference the exact file and pattern.
If you find nothing, return an empty list. Do not invent issues."""),

    ("human", """PR Title: {title}

Changed Files:
{changed_files}

Diff:
{diff}

Relevant codebase context:
{context}

Find security issues only. You MUST respond with ONLY this JSON format, nothing else:
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