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
    ("system", """You are a senior security engineer doing a security-focused review of a pull request.
Find security vulnerabilities only — not bugs, not style.

Check EVERY line of the diff for:

INPUT HANDLING:
- SQL injection: string concatenation or f-strings building SQL queries
- Command injection: os.system(), subprocess with shell=True, eval(), exec()
- Path traversal: user input joined to file paths without sanitization
- XSS: user input rendered in HTML without escaping
- Template injection: user input in Jinja2/Mako templates unsafely
- Unsafe deserialization: pickle.loads(), yaml.load() without SafeLoader
- Regex DoS: user input in re.compile() without timeout

SECRETS & AUTH:
- Hardcoded passwords, API keys, tokens, or connection strings
- Secrets in comments, variable names like 'password = "..."'
- Missing authentication on endpoints that modify data
- Missing authorization checks (user A accessing user B's data)
- JWT tokens without expiry or with weak algorithms (none, HS256 with weak key)
- CORS set to allow_origins=["*"] in production

DATA EXPOSURE:
- Sensitive data in log statements (passwords, tokens, PII)
- Stack traces or internal errors exposed to users
- Debug mode enabled in production configs
- Sensitive fields not excluded from API responses

CRYPTO & NETWORK:
- Use of MD5 or SHA1 for password hashing (should be bcrypt/argon2)
- HTTP URLs where HTTPS should be used
- TLS verification disabled (verify=False)
- Weak random (random.random() instead of secrets module for tokens)

RULES:
- Be specific: quote the exact dangerous code pattern.
- Explain the attack scenario (how would an attacker exploit this).
- Classify severity: high = exploitable, medium = needs specific conditions, low = defense-in-depth.
- Do NOT flag test files unless they contain real hardcoded production secrets.
- If the code is clean, return empty findings.

Respond with ONLY valid JSON, no markdown fences."""),

    ("human", """PR: {title}

File being reviewed: {current_file}

Diff for this file:
```
{file_diff}
```

Other changed files: {other_files}

Codebase context:
{context}

Return ONLY this JSON:
{{
  "findings": [
    {{
      "file": "exact/filename.py",
      "line": "the vulnerable code line or pattern",
      "severity": "high|medium|low",
      "issue": "what the vulnerability is",
      "suggestion": "how to fix it"
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

        other_files = [f for f in pr_data.changed_files if f != fd.filename]

        try:
            result = chain.invoke({
                "title": pr_data.title,
                "current_file": fd.filename,
                "file_diff": patch[:MAX_DIFF_CHARS],
                "other_files": ", ".join(other_files) if other_files else "None",
                "context": "\n---\n".join(context) if context else "No context available.",
            })
            findings = _parse_findings(result.content, fd.filename)
            all_findings.extend(findings)
        except Exception as e:
            print(f"[security] Error reviewing {fd.filename}: {e}")

    return all_findings


def _skip_file(filename: str) -> bool:
    # Security agent should still scan config files — only skip binary/media
    skip_ext = {".png", ".jpg", ".svg", ".ico", ".gif",
                ".woff", ".woff2", ".ttf", ".eot", ".map",
                ".min.js", ".min.css", ".lock"}
    skip_names = {"package-lock.json", "yarn.lock", "poetry.lock"}
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
                result.append({"file": fallback_file, "issue": f, "severity": "high", "line": "", "suggestion": ""})
            elif isinstance(f, dict):
                result.append({
                    "file": f.get("file", fallback_file),
                    "line": f.get("line", ""),
                    "severity": f.get("severity", "high"),
                    "issue": f.get("issue", str(f)),
                    "suggestion": f.get("suggestion", ""),
                })
        return result
    except json.JSONDecodeError:
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        return [{"file": fallback_file, "issue": l, "severity": "high", "line": "", "suggestion": ""} for l in lines]