import re
from dataclasses import dataclass, field

from github import Github
from src.config import settings


@dataclass
class FileChange:
    filename: str
    patch: str          # the diff for this file
    additions: int
    deletions: int


@dataclass
class PRData:
    url: str
    title: str
    description: str
    author: str
    repo_full_name: str
    changed_files: list[str]
    diff: str                              # full combined diff (kept for backward compat)
    file_diffs: list[FileChange] = field(default_factory=list)  # per-file diffs
    commits: list[str] = field(default_factory=list)
    lines_changed: int = 0


def fetch_pr(pr_url: str) -> PRData:
    """
    Takes a GitHub PR URL like:
    https://github.com/owner/repo/pull/123
    Returns a PRData object with everything the agents need.
    """
    owner, repo_name, pr_number = _parse_pr_url(pr_url)

    g = Github(settings.github_token)
    repo = g.get_repo(f"{owner}/{repo_name}")
    pr = repo.get_pull(pr_number)

    # Single get_files() call — cache the list
    files = list(pr.get_files())

    changed_files = [f.filename for f in files]

    # Per-file diffs
    file_diffs = []
    diff_parts = []
    for f in files:
        patch = f.patch or ""
        file_diffs.append(FileChange(
            filename=f.filename,
            patch=patch,
            additions=f.additions,
            deletions=f.deletions,
        ))
        if patch:
            diff_parts.append(f"--- {f.filename} ---\n{patch}")

    diff = "\n\n".join(diff_parts)
    commits = [c.commit.message for c in pr.get_commits()]
    lines_changed = sum(f.additions + f.deletions for f in files)

    return PRData(
        url=pr_url,
        title=pr.title,
        description=pr.body or "",
        author=pr.user.login,
        repo_full_name=f"{owner}/{repo_name}",
        changed_files=changed_files,
        diff=diff,
        file_diffs=file_diffs,
        commits=commits,
        lines_changed=lines_changed,
    )


def _parse_pr_url(pr_url: str) -> tuple[str, str, int]:
    pattern = r"github\.com/([^/]+)/([^/]+)/pull/(\d+)"
    match = re.search(pattern, pr_url)
    if not match:
        raise ValueError(f"Invalid GitHub PR URL: {pr_url}")
    owner, repo, number = match.groups()
    return owner, repo, int(number)