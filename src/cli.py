"""Summarize one log file: python -m src.cli build.log --backend baseline"""
import argparse
import sys

from src.baseline import format_summary
from src.summarizer import summarize

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="log file, or - for stdin")
    ap.add_argument("--backend", default="baseline", choices=["baseline", "hf", "anthropic"])
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    text = sys.stdin.read() if args.path == "-" else open(args.path).read()
    print(format_summary(summarize(text, args.backend, args.model)))
