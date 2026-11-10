"""LLM summarization with swappable backends (free local model or Claude API)."""
import os
import re

import requests

from src.preprocess import guess_failed_step, prepare

PROMPT = """You are a CI assistant. Summarize this failed build log for a developer.
Answer in exactly this format:
Failed step: <step name>
Root cause: <one sentence naming the exact error>
Suggested fix: <one sentence>

Log:
{log}
"""


class HFBackend:
    """Free open-source model that runs locally (needs requirements-llm.txt)."""
    name = "hf"
    max_log_chars = 1400

    def __init__(self, model="google/flan-t5-base"):
        from transformers import pipeline
        self.model = model
        self.pipe = pipeline("text2text-generation", model=model)

    def generate(self, prompt):
        out = self.pipe(prompt, max_new_tokens=96, truncation=True)
        return out[0]["generated_text"]


class AnthropicBackend:
    """Claude API. Needs the ANTHROPIC_API_KEY environment variable."""
    name = "anthropic"
    max_log_chars = 12000

    def __init__(self, model=None):
        self.model = model or os.getenv("LLM_MODEL", "claude-haiku-5-5")
        self.key = os.environ["ANTHROPIC_API_KEY"]

    def generate(self, prompt):
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": self.key, "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            json={"model": self.model, "max_tokens": 300,
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=60)
        r.raise_for_status()
        return r.json()["content"][0]["text"]


def parse_summary(text):
    """Turn the model's answer into a dict; fall back to the raw text."""
    def field(label):
        m = re.search(rf"{label}:\s*(.+)", text, re.I)
        return m.group(1).strip() if m else ""
    out = {"failed_step": field("Failed step"), "root_cause": field("Root cause"),
           "suggested_fix": field("Suggested fix")}
    if not out["root_cause"]:
        out["root_cause"] = text.strip().splitlines()[0] if text.strip() else ""
    return out


def llm_summary(log_text, backend):
    lines, window = prepare(log_text)
    excerpt = "\n".join(window)
    if len(excerpt) > backend.max_log_chars:
        excerpt = excerpt[-backend.max_log_chars:]
    raw = backend.generate(PROMPT.format(log=excerpt))
    summary = parse_summary(raw)
    if not summary["failed_step"]:
        summary["failed_step"] = guess_failed_step(lines)
    summary.update({"backend": backend.name, "raw": raw})
    return summary
