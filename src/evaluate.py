"""Score summarizers on labelled logs."""
import argparse
import csv
import json
import os
import re
import time

from src.baseline import format_summary
from src.make_synthetic_logs import make_dataset, save
from src.summarizer import summarize

OUT = "outputs"


def tokens(text):
    return re.findall(r"\w+", text.lower())


def lcs_len(a, b):
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def rouge_l(pred, ref):
    p, r = tokens(pred), tokens(ref)
    if not p or not r:
        return 0.0
    lcs = lcs_len(p, r)
    if lcs == 0:
        return 0.0
    prec, rec = lcs / len(p), lcs / len(r)
    return 2 * prec * rec / (prec + rec)


def score(summary, sample):
    text = format_summary(summary).lower()
    step_hit = any(a.lower() in text for a in sample["failed_step_aliases"])
    error_hit = sample["error_type"].lower() in text
    rouge = rouge_l(summary["root_cause"], sample["root_cause"]) if sample.get("root_cause") else None
    return {"step_hit": int(step_hit), "error_hit": int(error_hit), "rouge_l": rouge,
            "compression": len(text) / max(len(sample["log"]), 1)}


def load(path):
    if not os.path.exists(path):
        print(f"{path} not found, generating synthetic logs")
        rows = make_dataset()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        save(rows, path)
        return rows
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/logs.jsonl")
    ap.add_argument("--backends", default="baseline")
    ap.add_argument("--model", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    samples = load(args.data)[: args.limit]
    results, per_sample = {}, []
    for backend in args.backends.split(","):
        rows = []
        for s in samples:
            start = time.time()
            summary = summarize(s["log"], backend, args.model)
            latency = time.time() - start
            sc = score(summary, s)
            sc.update({"id": s["id"], "backend": backend, "latency_s": latency,
                       "summary": format_summary(summary).replace("\n", " | ")})
            rows.append(sc)
        per_sample += rows
        results[backend] = {
            "samples": len(rows),
            "step_accuracy": mean(r["step_hit"] for r in rows),
            "error_type_accuracy": mean(r["error_hit"] for r in rows),
            "rouge_l_root_cause": mean(r["rouge_l"] for r in rows),
            "compression_ratio": mean(r["compression"] for r in rows),
            "avg_latency_s": mean(r["latency_s"] for r in rows),
        }
        print(backend, json.dumps(results[backend], indent=1))

    with open(f"{OUT}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(f"{OUT}/per_sample.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_sample[0].keys()))
        w.writeheader()
        w.writerows(per_sample)
    with open(f"{OUT}/manual_rating_template.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "backend", "summary", "accuracy_1to5", "usefulness_1to5", "hallucination_yes_no"])
        for r in per_sample[:30]:
            w.writerow([r["id"], r["backend"], r["summary"], "", "", ""])


if __name__ == "__main__":
    main()
