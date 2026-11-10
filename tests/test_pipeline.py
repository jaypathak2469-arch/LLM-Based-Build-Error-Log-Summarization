from src.baseline import baseline_summary
from src.evaluate import rouge_l, score
from src.llm_summarizer import llm_summary, parse_summary
from src.make_synthetic_logs import make_dataset
from src.preprocess import (clean_lines, collapse_repeats, extract_window, guess_failed_step,
                            prepare, redact)

DATA = make_dataset(n=40, seed=1)


def test_clean_lines_strips_ansi_and_timestamps():
    out = clean_lines("2026-11-09T10:00:01.123Z \x1b[31mboom\x1b[0m\n\n   \nnext")
    assert out == ["boom", "next"]


def test_redact_hides_tokens_and_passwords():
    assert "ghp_" not in redact("auth ghp_abcdefghijklmnopqrstuvwxyz0123456789")
    assert "hunter2" not in redact("password=hunter2")


def test_collapse_repeats():
    assert collapse_repeats(["a", "a", "a", "b"]) == ["a (repeated 3x)", "b"]


def test_window_keeps_error_and_is_short():
    lines = [f"line {i}" for i in range(500)] + ["ValueError: boom"] + [f"tail {i}" for i in range(500)]
    window = extract_window(lines, max_lines=40)
    assert len(window) <= 45
    assert "ValueError: boom" in window


def test_guess_failed_step():
    lines = ["##[group]Run actions/checkout@v4", "##[group]Run pytest -q", "FAILED x",
             "##[error]Process completed with exit code 1."]
    assert guess_failed_step(lines) == "pytest -q"


def test_baseline_finds_step_and_error_on_most_samples():
    step = error = 0
    for s in DATA:
        sc = score(baseline_summary(s["log"]), s)
        step += sc["step_hit"]
        error += sc["error_hit"]
    assert step / len(DATA) >= 0.9
    assert error / len(DATA) >= 0.7


def test_prepare_shortens_logs():
    lines, window = prepare(DATA[0]["log"])
    assert len(window) < len(lines)


def test_rouge_l_bounds():
    assert rouge_l("a b c", "a b c") == 1.0
    assert rouge_l("a b c", "x y z") == 0.0


def test_parse_summary_reads_fields():
    out = parse_summary("Failed step: pytest\nRoot cause: assert 1 == 2\nSuggested fix: fix the test")
    assert out == {"failed_step": "pytest", "root_cause": "assert 1 == 2",
                   "suggested_fix": "fix the test"}


class FakeBackend:
    name = "fake"
    max_log_chars = 1000

    def generate(self, prompt):
        assert "Log:" in prompt
        return "Root cause: ModuleNotFoundError: No module named 'x'"


def test_llm_summary_with_fake_backend_falls_back_to_detected_step():
    sample = DATA[1]
    out = llm_summary(sample["log"], FakeBackend())
    assert out["failed_step"] == sample["failed_step_aliases"][0]
    assert "ModuleNotFoundError" in out["root_cause"]
