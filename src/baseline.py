"""Rule-based summarizer: no model, used as the baseline to compare the LLM against."""
import re

from src.preprocess import ERROR_HINT, guess_failed_step, prepare

EXIT_LINE = re.compile(r"Process completed with exit code|exceeded the maximum execution time")
CANDIDATES = [
    re.compile(r"\b\w*(?:Error|Exception)\b: .+"),
    re.compile(r"^E\s+.+"),
    re.compile(r"\bF\d{3}\b.+"),
    re.compile(r"npm ERR! .+"),
    re.compile(r"(?i)\b(?:error|fatal): .+"),
    re.compile(r"(?i)permission denied|no space left|killed"),
]
FIX_HINTS = [
    (re.compile(r"ModuleNotFoundError|ImportError|No module named"),
     "Add the missing package to requirements.txt and reinstall."),
    (re.compile(r"AssertionError|assert "),
     "Check the failing test and the code it covers, then fix the code or the expected value."),
    (re.compile(r"\bF\d{3}\b|flake8|E501"),
     "Fix the lint problems reported by flake8."),
    (re.compile(r"ERESOLVE|dependency tree"),
     "Resolve the dependency version conflict (align or pin the versions)."),
    (re.compile(r"undeclared|undefined reference|syntax error", re.I),
     "Fix the compile error at the reported file and line."),
    (re.compile(r"canceled|maximum execution time|timed out", re.I),
     "Find the step that hangs and add a timeout."),
    (re.compile(r"Could not find a version|did not complete successfully"),
     "Pin an available package version in the build file."),
    (re.compile(r"(?i)permission denied"),
     "Check file permissions (for example chmod +x on the script)."),
]


def pick_root_cause(lines):
    end = next((i for i, line in enumerate(lines) if EXIT_LINE.search(line)), len(lines) - 1)
    scope = lines[: end + 1]
    hits = [line for line in scope
            if not EXIT_LINE.search(line) and any(p.search(line) for p in CANDIDATES)]
    if not hits:
        hits = [line for line in scope if not EXIT_LINE.search(line) and ERROR_HINT.search(line)]
    if hits:
        return hits[-1].strip()
    return scope[-1].strip() if scope else "no error found"


def baseline_summary(log_text):
    lines, window = prepare(log_text)
    root_cause = pick_root_cause(window)
    fix = "Read the log around the error line."
    for pattern, hint in FIX_HINTS:
        if pattern.search(root_cause) or pattern.search("\n".join(window)):
            fix = hint
            break
    return {"failed_step": guess_failed_step(lines), "root_cause": root_cause,
            "suggested_fix": fix, "backend": "baseline"}


def format_summary(summary):
    return (f"Failed step: {summary['failed_step']}\n"
            f"Root cause: {summary['root_cause']}\n"
            f"Suggested fix: {summary['suggested_fix']}")
