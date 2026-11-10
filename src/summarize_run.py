"""Used by GitHub Actions: summarize the failed jobs of a finished workflow run."""
import os

from src.baseline import format_summary
from src.github_api import API, get_json, make_session
from src.summarizer import summarize


def main():
    repo, run_id = os.environ["GITHUB_REPOSITORY"], os.environ["RUN_ID"]
    backend = "anthropic" if os.getenv("ANTHROPIC_API_KEY") else "baseline"
    session = make_session()
    jobs = get_json(session, f"{API}/repos/{repo}/actions/runs/{run_id}/jobs")["jobs"]
    failed = [j for j in jobs if j["conclusion"] == "failure"]
    parts = [f"## Build failure summary (run {run_id}, backend: {backend})"]
    for job in failed:
        r = session.get(f"{API}/repos/{repo}/actions/jobs/{job['id']}/logs", timeout=60)
        summary = summarize(r.text, backend) if r.status_code == 200 else None
        body = format_summary(summary) if summary else f"Could not download logs (HTTP {r.status_code})"
        parts.append(f"### Job: {job['name']}\n```\n{body}\n```")
    if not failed:
        parts.append("No failed jobs found.")
    text = "\n\n".join(parts)
    print(text)
    path = os.getenv("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
