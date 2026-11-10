"""Generate realistic, labelled GitHub Actions failure logs.

Used to test the pipeline and CI without downloads. Each sample carries the
true failed step, error type, root cause and fix so summaries can be scored.
"""
import json
import random
from datetime import datetime, timedelta

MODULES = ["xgboost", "seaborn", "yaml", "torch", "pydantic", "boto3"]
FUNCS = ["test_split_is_chronological", "test_features_have_no_nans", "test_parse_config",
         "test_retry_logic", "test_total_price", "test_login_flow"]
SYMS = ["buffer_len", "init_table", "max_retries", "cfg"]
PKGS = ["react", "webpack", "eslint", "typescript"]


def pip_noise(rng, n):
    names = ["numpy", "pandas", "scipy", "requests", "urllib3", "idna", "certifi", "six"]
    out = []
    for _ in range(n):
        p = rng.choice(names)
        out.append(rng.choice([f"Collecting {p}>=1.{rng.randint(0, 9)}",
                               f"  Downloading {p}-1.{rng.randint(0, 9)}.0-py3-none-any.whl (1.2 MB)",
                               f"Requirement already satisfied: {p} in /opt/hostedtoolcache/lib"]))
    return out


def t_pytest(rng):
    f, a, b = rng.choice(FUNCS), rng.randint(1, 9), rng.randint(10, 99)
    mod, ln = f.replace("test_", ""), rng.randint(10, 90)
    body = ["..F.", "=================================== FAILURES ===================================",
            f"_________________ {f} _________________", f"    def {f}():",
            f">       assert {a} == {b}", f"E       AssertionError: assert {a} == {b}",
            f"tests/test_{mod}.py:{ln}: AssertionError",
            "=========================== short test summary info ============================",
            f"FAILED tests/test_{mod}.py::{f} - AssertionError: assert {a} == {b}",
            "1 failed, 12 passed in 0.84s"]
    return ("pytest -q", body, "AssertionError", f"{f} failed because assert {a} == {b} is false",
            "Fix the code or the expected value in the failing test", 1)


def t_module(rng):
    m, ln = rng.choice(MODULES), rng.randint(3, 40)
    body = ["Traceback (most recent call last):",
            f'  File "/home/runner/work/app/src/train.py", line {ln}, in <module>',
            f"    import {m}", f"ModuleNotFoundError: No module named '{m}'"]
    return ("python -m src.train", body, "ModuleNotFoundError", f"No module named '{m}'",
            f"Add {m} to requirements.txt", 1)


def t_flake8(rng):
    f, imp, ln = rng.choice(["train", "features", "utils"]), rng.choice(["os", "sys", "json"]), rng.randint(1, 9)
    body = [f"src/{f}.py:{ln}:1: F401 '{imp}' imported but unused",
            f"src/{f}.py:{ln + 20}:80: E501 line too long (96 > 79 characters)"]
    return ("flake8 src tests", body, "F401", f"'{imp}' imported but unused in src/{f}.py",
            "Remove the unused import", 1)


def t_npm(rng):
    p, v = rng.choice(PKGS), rng.randint(15, 18)
    body = ["npm ERR! code ERESOLVE", "npm ERR! ERESOLVE unable to resolve dependency tree",
            f"npm ERR! Found: {p}@{v}.0.0", "npm ERR! Could not resolve peer dependency"]
    return ("npm ci", body, "ERESOLVE", f"unable to resolve dependency tree for {p}",
            "Align or pin the conflicting package versions", 1)


def t_compile(rng):
    f, s, ln = rng.choice(["main", "table", "net"]), rng.choice(SYMS), rng.randint(10, 200)
    body = [f"src/{f}.c: In function 'main':",
            f"src/{f}.c:{ln}:5: error: '{s}' undeclared (first use in this function)",
            "make: *** [Makefile:12: all] Error 1"]
    return ("make", body, "undeclared", f"'{s}' undeclared in src/{f}.c at line {ln}",
            "Declare the symbol or include the right header", 2)


def t_timeout(rng):
    body = [f"tests/integration/test_api.py::test_{i} PASSED" for i in range(rng.randint(3, 8))]
    body += ["Error: The operation was canceled.",
             "##[error]The job running on runner GitHub Actions 5 has exceeded "
             "the maximum execution time of 360 minutes."]
    return ("pytest tests/integration", body, "canceled",
            "the job exceeded the maximum execution time", "Find the hanging test and add a timeout", None)


def t_docker(rng):
    p = rng.choice(["scikit-learn", "pandas", "torch"])
    body = ["#8 [4/6] RUN pip install -r requirements.txt",
            f"#8 ERROR: Could not find a version that satisfies the requirement {p}==9.9.9",
            'ERROR: failed to solve: process "/bin/sh -c pip install -r requirements.txt" '
            "did not complete successfully: exit code: 1"]
    return ("docker build -t app .", body, "Could not find a version",
            f"no version of {p}==9.9.9 exists", "Pin an available version of the package", 1)


def t_permission(rng):
    s = rng.choice(["deploy.sh", "build.sh", "release.sh"])
    body = [f"bash: ./{s}: Permission denied"]
    return (f"./{s}", body, "Permission denied", f"{s} is not executable",
            f"Run chmod +x {s} and commit it", 126)


TEMPLATES = [t_pytest, t_module, t_flake8, t_npm, t_compile, t_timeout, t_docker, t_permission]


def build_log(rng, template):
    cmd, body, etype, root, fix, code = template(rng)
    lines = ["##[group]Run actions/checkout@v4", "Syncing repository: owner/app", "##[endgroup]",
             "##[group]Run actions/setup-python@v5", "Successfully set up Python 3.11"]
    if rng.random() < 0.4:
        lines.append("Error: connection reset by peer, retrying (1/3)")
    lines += ["##[endgroup]", "##[group]Run pip install -r requirements.txt"]
    lines += pip_noise(rng, rng.randint(30, 150))
    if rng.random() < 0.5:
        lines.append("Skipping error-handling tests (not configured)")
    lines += ["##[endgroup]", f"##[group]Run {cmd}"] + body
    if code is not None:
        lines.append(f"##[error]Process completed with exit code {code}.")
    lines += ["##[group]Post Run actions/checkout", "Cleaning up orphan processes"]
    if rng.random() < 0.4:
        lines.append("##[warning]Failed to save cache: error: cache service unavailable")
    t = datetime(2026, 11, 9, 10, 0, 0)
    stamped = []
    for i, line in enumerate(lines):
        ts = (t + timedelta(milliseconds=137 * i)).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        stamped.append(f"{ts} {line}")
    return {"log": "\n".join(stamped), "failed_step_aliases": [cmd], "error_type": etype,
            "root_cause": root, "fix": fix}


def make_dataset(n=60, seed=42):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        row = build_log(rng, TEMPLATES[i % len(TEMPLATES)])
        row["id"] = f"syn-{i:03d}"
        rows.append(row)
    return rows


def save(rows, path):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    data = make_dataset()
    save(data, "data/logs.jsonl")
    print(f"Wrote {len(data)} labelled logs to data/logs.jsonl")
