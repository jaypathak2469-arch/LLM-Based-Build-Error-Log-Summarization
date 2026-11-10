# LLM-Based Build/Error-Log Summarization

Turns long, noisy CI failure logs into a short summary (failed step, root cause, suggested fix)
and compares a rule-based baseline with language models. A GitHub Actions workflow summarizes
failed builds automatically.

## Quick start (no API key, no download)
```bash
pip install -r requirements.txt
pytest -q
python -m src.evaluate --backends baseline
python -m src.cli build.log      # summarize one log file
```

## Free local LLM
```bash
pip install -r requirements-llm.txt
python -m src.evaluate --backends baseline,hf --limit 24
```

## Real logs
```bash
export GITHUB_TOKEN=<your token>
python -m src.collect_logs pallets/flask psf/requests --max-logs 15
python -m src.evaluate --data data/real_logs.jsonl --backends baseline,hf
```

## Optional: Claude API
Set `ANTHROPIC_API_KEY` (locally, or as a repository secret for the workflow) and use `--backends anthropic`.

## Pipeline
1. `preprocess.py` cleans, redacts secrets, collapses repeats, keeps the window around errors
2. `baseline.py` rule-based summary
3. `llm_summarizer.py` prompt plus HF / Claude backends
4. `evaluate.py` step accuracy, error-type accuracy, ROUGE-L, compression, latency
5. Workflows: `ci.yml` (lint, tests, evaluation), `summarize-failed-build.yml` (auto-summary of failed runs),
   `demo-failing-build.yml` (manual, fails on purpose to trigger the summary)
