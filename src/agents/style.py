import json
import re

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from src.config import settings
from src.pr_fetcher import PRData


def _get_llm():
    return ChatOpenAI(
        model=settings.model_name,
        openai_api_key=settings.openrouter_api_key,
        openai_api_base=settings.openrouter_base_url,
        temperature=0,
    )


PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a senior software engineer reviewing a pull request for code quality and style.
Find style and maintainability issues only — not bugs, not security.

Check for:
- Functions longer than ~40 lines that should be split
- Deeply nested code (3+ levels of if/for/try) that should be flattened
- Duplicated logic that should be extracted into a helper
- Naming that doesn't match the codebase conventions (compare with context)
- Missing or misleading docstrings on public functions/classes
- Dead code, commented-out code, or leftover debug prints
- Magic numbers/strings that should be named constants
- Inconsistent patterns (e.g., some functions use early return, others don't)
- Missing type hints where the rest of the codebase uses them
- Overly complex list comprehensions that hurt readability
- Imports that are unused or could be more specific

RULES:
- Use the codebase context to judge existing style. Don't enforce YOUR preferences — enforce THEIR patterns.
- If the PR matches the existing codebase style, return empty findings.
- Be specific: quote the code pattern, not just "naming is bad".

Respond with ONLY valid JSON, no markdown fences."""),

    ("human", """PR: {title}

File being reviewed: {current_file}

Diff for this file:
```
{file_diff}
```

Existing codebase style (for reference):
{context}

Return ONLY this JSON:
{{
  "findings": [
    {{
      "file": "exact/filename.py",
      "line": "the problematic code or pattern",
      "severity": "high|medium|low",
      "issue": "what the style problem is",
      "suggestion": "how to improve it"
    }}
  ]
}}
If no issues: {{"findings": []}}""")])


MAX_DIFF_CHARS = 6000


def run(pr_data: PRData, context: list[str]) -> list[dict]:
    llm = _get_llm()
    chain = PROMPT | llm
    all_findings = []

    file_diffs = pr_data.file_diffs if pr_data.file_diffs else []

    if not file_diffs:
        file_diffs = [type("F", (), {"filename": "unknown", "patch": pr_data.diff[:MAX_DIFF_CHARS]})]

    for fd in file_diffs:
        if _skip_file(fd.filename):
            continue

        patch = fd.patch if isinstance(fd.patch, str) else ""
        if not patch.strip():
            continue

        try:
            result = chain.invoke({
                "title": pr_data.title,
                "current_file": fd.filename,
                "file_diff": patch[:MAX_DIFF_CHARS],
                "context": "\n---\n".join(context) if context else "No context available.",
            })
            findings = _parse_findings(result.content, fd.filename)
            all_findings.extend(findings)
        except Exception as e:
            print(f"[style] Error reviewing {fd.filename}: {e}")

    return all_findings


def _skip_file(filename: str) -> bool:
    skip_ext = {".md", ".txt", ".json", ".yml", ".yaml", ".toml", ".cfg",
                ".ini", ".lock", ".png", ".jpg", ".svg", ".ico", ".gif",
                ".woff", ".woff2", ".ttf", ".eot", ".map", ".min.js", ".min.css"}
    skip_names = {"package-lock.json", "yarn.lock", "poetry.lock", ".gitignore",
                  "LICENSE", "CHANGELOG.md", "requirements.txt"}
    lower = filename.lower()
    if any(lower.endswith(ext) for ext in skip_ext):
        return True
    if filename.split("/")[-1] in skip_names:
        return True
    return False


def _parse_findings(text: str, fallback_file: str) -> list[dict]:
    text = re.sub(r"```json|```", "", text).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        findings = parsed.get("findings", [])
        result = []
        for f in findings:
            if isinstance(f, str):
                result.append({"file": fallback_file, "issue": f, "severity": "medium", "line": "", "suggestion": ""})
            elif isinstance(f, dict):
                result.append({
                    "file": f.get("file", fallback_file),
                    "line": f.get("line", ""),
                    "severity": f.get("severity", "medium"),
                    "issue": f.get("issue", str(f)),
                    "suggestion": f.get("suggestion", ""),
                })
        return result
    except json.JSONDecodeError:
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        return [{"file": fallback_file, "issue": l, "severity": "medium", "line": "", "suggestion": ""} for l in lines]