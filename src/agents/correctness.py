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
    ("system", """You are a senior software engineer doing a correctness review of a pull request.
Find REAL BUGS only. Not style, not security — only logic errors.

Check systematically for:
- Wrong boolean conditions (using 'and' instead of 'or', inverted checks)
- Off-by-one errors in loops, slicing, range()
- Null/None/undefined access without guards
- Variables used before assignment or after reassignment
- Exception handling that swallows errors silently (bare except, catching too broad)
- Return values that are never used or wrong type returned
- Functions that mutate input arguments unexpectedly
- Race conditions or shared mutable state
- Missing break/continue/return causing fall-through
- Incorrect string formatting (f-string with wrong variable, .format mismatches)
- Integer division when float was needed (or vice versa)
- Comparison with 'is' when '==' was needed
- Mutable default arguments (def f(x=[]))

RULES:
- Only report issues you are confident about. Do NOT invent problems.
- Reference the exact filename and the code line/pattern.
- Explain WHY it's a bug and what the fix is.
- If the diff is clean, return an empty findings array.

Respond with ONLY valid JSON, no markdown fences, no explanation outside the JSON."""),

    ("human", """PR: {title}
Description: {description}
Author: {author}

File being reviewed: {current_file}

Diff for this file:
```
{file_diff}
```

Other changed files in this PR: {other_files}

Codebase context (related existing code):
{context}

Return ONLY this JSON:
{{
  "findings": [
    {{
      "file": "exact/filename.py",
      "line": "the problematic code line or pattern",
      "severity": "high|medium|low",
      "issue": "what the bug is",
      "suggestion": "how to fix it"
    }}
  ]
}}
If no issues: {{"findings": []}}""")])


MAX_DIFF_CHARS = 6000  # per file


def run(pr_data: PRData, context: list[str]) -> list[dict]:
    llm = _get_llm()
    chain = PROMPT | llm
    all_findings = []

    file_diffs = pr_data.file_diffs if pr_data.file_diffs else []

    # If no per-file diffs available, fall back to full diff
    if not file_diffs:
        file_diffs = [type("F", (), {"filename": "unknown", "patch": pr_data.diff[:MAX_DIFF_CHARS]})]

    for fd in file_diffs:
        # Skip non-code files
        if _skip_file(fd.filename):
            continue

        patch = fd.patch if isinstance(fd.patch, str) else ""
        if not patch.strip():
            continue

        other_files = [f for f in pr_data.changed_files if f != fd.filename]

        try:
            result = chain.invoke({
                "title": pr_data.title,
                "description": pr_data.description,
                "author": pr_data.author,
                "current_file": fd.filename,
                "file_diff": patch[:MAX_DIFF_CHARS],
                "other_files": ", ".join(other_files) if other_files else "None",
                "context": "\n---\n".join(context) if context else "No context available.",
            })
            findings = _parse_findings(result.content, fd.filename)
            all_findings.extend(findings)
        except Exception as e:
            print(f"[correctness] Error reviewing {fd.filename}: {e}")

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
        # Normalize each finding
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