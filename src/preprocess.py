"""Clean, redact and shorten raw CI logs before they are summarized."""
import re

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z\s?")
GROUP = re.compile(r"^##\[group\](.*)$")
SECRETS = [
    (re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"), "[REDACTED_TOKEN]"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "[REDACTED_AWS_KEY]"),
    (re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._\-]{15,}"), r"\1[REDACTED]"),
    (re.compile(r"(?i)((?:password|passwd|secret|token|api[_-]?key)\s*[=:]\s*)\S+"), r"\1[REDACTED]"),
]
ERROR_HINT = re.compile(
    r"(##\[error\]|traceback|exception|error\b|\bfailed\b|\bfailure\b|fatal|assert|"
    r"cannot|not found|denied|timed out|killed|canceled)", re.I)


def redact(line):
    for pattern, replacement in SECRETS:
        line = pattern.sub(replacement, line)
    return line


def clean_lines(text):
    """Strip colour codes and timestamps, drop empty lines, redact secrets."""
    lines = []
    for raw in text.lstrip("\ufeff").splitlines():
        line = TIMESTAMP.sub("", ANSI.sub("", raw)).rstrip()
        if line.strip():
            lines.append(redact(line))
    return lines


def collapse_repeats(lines):
    """Replace runs of identical lines with one line and a repeat count."""
    out, i = [], 0
    while i < len(lines):
        j = i
        while j + 1 < len(lines) and lines[j + 1] == lines[i]:
            j += 1
        count = j - i + 1
        out.append(lines[i] if count == 1 else f"{lines[i]} (repeated {count}x)")
        i = j + 1
    return out


def error_indices(lines):
    return [i for i, line in enumerate(lines) if ERROR_HINT.search(line)]


def extract_window(lines, max_lines=40, context=3):
    """Keep only the parts of a long log that surround error lines."""
    if len(lines) <= max_lines:
        return lines
    idx = error_indices(lines)
    if not idx:
        return lines[-max_lines:]
    keep = set()
    for i in idx[:4] + idx[-4:]:
        keep.update(range(max(0, i - context), min(len(lines), i + context + 1)))
    ordered = sorted(keep)
    if len(ordered) > max_lines:
        half = max_lines // 2
        ordered = ordered[:half] + ordered[-half:]
    out, prev = [], None
    for i in ordered:
        if prev is not None and i != prev + 1:
            out.append("...")
        out.append(lines[i])
        prev = i
    return out


def guess_failed_step(lines):
    """The last step header before the first ##[error] line."""
    last = None
    for line in lines:
        match = GROUP.match(line)
        if match:
            last = match.group(1).strip()
        if line.startswith("##[error]") and last:
            break
    if last is None:
        return "unknown"
    return re.sub(r"^Run\s+", "", last)


def prepare(text, max_lines=40):
    """Full preprocessing pipeline: returns (all cleaned lines, window)."""
    lines = collapse_repeats(clean_lines(text))
    return lines, extract_window(lines, max_lines=max_lines)
