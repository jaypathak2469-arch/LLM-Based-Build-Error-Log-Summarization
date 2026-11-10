"""Download real failed-build logs from public repositories.

Usage:
    export GITHUB_TOKEN=<your token>
    python -m src.collect_logs pallets/flask psf/requests --max-logs 15

Labels for real logs are automatic (weak): the failed step comes from the API
and the error type from the log itself. Use manual rating for the real quality check.
"""
import argparse
import json
import re

import requests

from src.github_api import API, get_json, make_session
from src.preprocess import clean_lines, guess_failed_step

ERR_TYPE = re.compile(r"\b(\w+Error|\w+Exception|F\d{3}|E\d{3}|ERESOLVE)\b")


def collect_repo(session, repo, max_logs):
    rows, page = [], 1
    try:
        while len(rows) < max_logs and page <= 5:
            runs = get_json(session, f"{API}/repos/{repo}/actions/runs",
                            {"per_page": 30, "page": page, "status": "failure"}).get("workflow_runs", [])
            if not runs:
                break
            for run in runs:
                jobs = get_json(session, f"{API}/repos/{repo}/actions/runs/{run['id']}/jobs")["jobs"]
                for job in [j for j in jobs if j["conclusion"] == "failure"]:
                    r = session.get(f"{API}/repos/{repo}/actions/jobs/{job['id']}/logs", timeout=60)
                    if r.status_code != 200 or len(r.text) > 2_000_000:
                        continue
                    lines = clean_lines(r.text)
                    match = ERR_TYPE.search("\n".join(lines))
                    if not match:
                        continue
                    aliases = [s["name"] for s in job["steps"] if s["conclusion"] == "failure"]
                    aliases.append(guess_failed_step(lines))
                    rows.append({"id": f"{repo}#{job['id']}", "log": r.text,
                                 "failed_step_aliases": aliases, "error_type": match.group(1),
                                 "root_cause": "", "fix": ""})
                    print(f"{repo}: {len(rows)} logs collected", flush=True)
                    if len(rows) >= max_logs:
                        return rows
            page += 1
    except requests.RequestException as e:
        print(f"WARNING: {repo} stopped early after {len(rows)} logs ({type(e).__name__})")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("repos", nargs="+")
    ap.add_argument("--max-logs", type=int, default=15)
    args = ap.parse_args()
    session = make_session()
    all_rows = []
    for repo in args.repos:
        all_rows += collect_repo(session, repo, args.max_logs)
        with open("data/real_logs.jsonl", "w") as f:
            for row in all_rows:
                f.write(json.dumps(row) + "\n")
        print(f"Saved {len(all_rows)} logs so far to data/real_logs.jsonl", flush=True)
