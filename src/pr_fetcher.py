import re
from dataclasses import dataclass

from github import Github
from src.config import settings


@dataclass
class PRData:
    url: str
    title: str
    description: str
    author: str
    repo_full_name: str
    changed_files: list[str]
    diff: str
    commits: list[str]
    lines_changed: int


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

    # Get changed files
    changed_files = [f.filename for f in pr.get_files()]

    # Get full diff
    diff_parts = []
    for f in pr.get_files():
        if f.patch:
            diff_parts.append(f"--- {f.filename} ---\n{f.patch}")
    diff = "\n\n".join(diff_parts)

    # Get commit messages
    commits = [c.commit.message for c in pr.get_commits()]

    # Total lines changed
    lines_changed = sum(
        f.additions + f.deletions for f in pr.get_files()
    )

    return PRData(
        url=pr_url,
        title=pr.title,
        description=pr.body or "",
        author=pr.user.login,
        repo_full_name=f"{owner}/{repo_name}",
        changed_files=changed_files,
        diff=diff,
        commits=commits,
        lines_changed=lines_changed,
    )


def _parse_pr_url(pr_url: str) -> tuple[str, str, int]:
    """
    Parses https://github.com/owner/repo/pull/123
    Returns (owner, repo, pr_number)
    """
    pattern = r"github\.com/([^/]+)/([^/]+)/pull/(\d+)"
    match = re.search(pattern, pr_url)
    if not match:
        raise ValueError(f"Invalid GitHub PR URL: {pr_url}")
    owner, repo, number = match.groups()
    return owner, repo, int(number)