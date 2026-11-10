"""One entry point for all backends."""
from src.baseline import baseline_summary
from src.llm_summarizer import AnthropicBackend, HFBackend, llm_summary

_CACHE = {}


def get_backend(name, model=None):
    key = (name, model)
    if key not in _CACHE:
        if name == "hf":
            _CACHE[key] = HFBackend(model) if model else HFBackend()
        elif name == "anthropic":
            _CACHE[key] = AnthropicBackend(model)
        else:
            raise ValueError(f"unknown backend: {name}")
    return _CACHE[key]


def summarize(log_text, backend="baseline", model=None):
    if backend == "baseline":
        return baseline_summary(log_text)
    return llm_summary(log_text, get_backend(backend, model))
