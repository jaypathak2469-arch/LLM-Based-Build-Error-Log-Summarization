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

# Small models copy placeholders like <step name>, so they get a worked example instead.
HF_PROMPT = """Summarize the failed build log.

Example log:
##[group]Run npm test
Error: Cannot find module 'express'
##[error]Process completed with exit code 1.
Example answer:
Failed step: npm test
Root cause: Cannot find module 'express'
Suggested fix: Install the express package.

Log:
{log}
Answer:
"""


class HFBackend:
    """Free open-source model that runs locally (needs requirements-llm.txt)."""
    name = "hf"
    max_log_chars = 1400
    prompt = HF_PROMPT
    log_token_budget = 360

    def __init__(self, model="google/flan-t5-base"):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self.model_name = model
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model)
        self.model.eval()

    def fit_excerpt(self, excerpt):
        """Keep the END of the log (where the error is) within the model's token budget."""
        ids = self.tokenizer(excerpt, add_special_tokens=False)["input_ids"]
        if len(ids) <= self.log_token_budget:
            return excerpt
        return self.tokenizer.decode(ids[-self.log_token_budget:], skip_special_tokens=True)

    def generate(self, prompt):
        import torch
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=96)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)


class AnthropicBackend:
    """Claude API. Needs the ANTHROPIC_API_KEY environment variable."""
    name = "anthropic"
    max_log_chars = 12000
    prompt = PROMPT

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
    if hasattr(backend, "fit_excerpt"):
        excerpt = backend.fit_excerpt(excerpt)
    elif len(excerpt) > backend.max_log_chars:
        excerpt = excerpt[-backend.max_log_chars:]
    raw = backend.generate(getattr(backend, "prompt", PROMPT).format(log=excerpt))
    summary = parse_summary(raw)
    if not summary["failed_step"]:
        summary["failed_step"] = guess_failed_step(lines)
    summary.update({"backend": backend.name, "raw": raw})
    return summary
