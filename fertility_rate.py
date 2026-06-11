#!/usr/bin/env python
# -*- coding: utf-8 -*-
# this script compute the fertility rate of a model using its tokenizer
#
# Copyright (C) 2026, AA
# Last Update: Thu May 14 12:20:10 +03 2026
#

"""
fertility_rate.py
-----------------
Compute and compare the tokenizer fertility rate for one or more
HuggingFace models given an input text file.

Fertility rate = (number of tokens produced by the tokenizer)
                 / (number of whitespace-split words in the source text)

Usage
-----
# Single model
python fertility_rate.py --input corpus.txt --models bert-base-uncased

# Multiple models
python fertility_rate.py --input corpus.txt \
    --models bert-base-uncased gpt2 meta-llama/Llama-2-7b-hf \
            mistralai/Mistral-7B-v0.1 google/flan-t5-base

# Save results to CSV
python fertility_rate.py --input corpus.txt \
    --models bert-base-uncased gpt2 \
    --output results.csv

# Process sentence-by-sentence and show per-sentence stats
python fertility_rate.py --input corpus.txt \
    --models gpt2 bert-base-uncased \
    --per-sentence \
    --output results.csv

# Chinese text (uses jieba segmentation)
python fertility_rate.py --input corpus_zh --models bert-base-chinese --chinese
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
# For Chinese: Install with pip install jieba
import jieba 

# Must be set before the HuggingFace tokenizers Rust library initialises its
# thread pool, which happens at the first AutoTokenizer import. Setting it to
# "false" disables parallelism (safe for a single-process CLI script) and
# silences the fork-safety warning entirely.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------
try:
    from transformers import AutoTokenizer
except ImportError:
    sys.exit(
        "transformers is not installed.\n"
        "Install it with:  pip install transformers\n"
        "For models that need sentencepiece: pip install transformers sentencepiece"
    )

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SentenceStats:
    sentence: str
    word_count: int
    token_count: int
    fertility_rate: float


@dataclass
class ModelResult:
    model_name: str
    total_words: int
    total_tokens: int
    fertility_rate: float
    per_sentence: List[SentenceStats] = field(default_factory=list)
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def count_words(text: str, is_chinese: bool = False) -> int:
    """
    Count words in text.
    
    For English/whitespace-delimited languages: uses simple split().
    For Chinese: uses jieba segmentation since Chinese has no whitespace between words.
    """
    if is_chinese:
        # Use jieba for Chinese word segmentation
        return len(list(jieba.cut(text)))
    return len(text.split())


def split_sentences(text: str) -> List[str]:
    """
    Naive sentence splitter: splits on '.', '!', '?' followed by whitespace.
    Suitable for a wide range of plain-text corpora without extra dependencies.
    For Chinese, also splits on Chinese sentence endings: 。！？
    """
    import re
    # Support both Western and Chinese sentence endings
    parts = re.split(r'(?<=[.!?。！？])\s+', text.strip())
    return [p for p in parts if p.strip()]


def compute_fertility(
    text: str,
    tokenizer,
    per_sentence: bool = False,
    is_chinese: bool = False,
) -> tuple:
    """
    Returns (total_words, total_tokens, fertility_rate, sentence_stats_list).
    
    Args:
        text: Input text to analyze
        tokenizer: HuggingFace tokenizer
        per_sentence: Whether to compute per-sentence statistics
        is_chinese: If True, use jieba for Chinese word segmentation
    """
    total_words = count_words(text, is_chinese=is_chinese)

    # Tokenize the full text at once for an accurate total
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False,
    )
    total_tokens = len(encoded["input_ids"])

    fertility = total_tokens / total_words if total_words > 0 else 0.0

    sentence_stats: List[SentenceStats] = []
    if per_sentence:
        for sentence in split_sentences(text):
            w = count_words(sentence, is_chinese=is_chinese)
            if w == 0:
                continue
            enc = tokenizer(
                sentence,
                add_special_tokens=False,
                truncation=False,
                return_attention_mask=False,
            )
            t = len(enc["input_ids"])
            sentence_stats.append(
                SentenceStats(
                    sentence=sentence,
                    word_count=w,
                    token_count=t,
                    fertility_rate=round(t / w, 4),
                )
            )

    return total_words, total_tokens, fertility, sentence_stats


def load_tokenizer(model_name: str):
    """
    Load a tokenizer from HuggingFace Hub.

    Strategy:
      1. Try the fast (Rust-based) tokenizer.
      2. If that raises a deserialization error (common with bleeding-edge
         models whose tokenizer.json uses a newer format than the installed
         `tokenizers` library supports), fall back to the slow (Python) tokenizer.
      3. If both fail, print an actionable error and return None.

    The most common fix for the 'data did not match any variant of untagged
    enum ModelWrapper' error is:
        pip install -U tokenizers transformers
    """
    _DESERIALIZE_HINTS = (
        "did not match any variant",
        "untagged enum",
        "ModelWrapper",
        "EOF while parsing",
        "expected value at line",
    )

    print(f"  Loading tokenizer: {model_name} ...", end=" ", flush=True)

    # --- attempt 1: fast tokenizer ---
    try:
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        print("OK (fast)")
        return tok
    except Exception as exc:
        exc_str = str(exc)
        is_deserialize_error = any(hint in exc_str for hint in _DESERIALIZE_HINTS)

        if not is_deserialize_error:
            # Unrelated error (auth, network, missing files) — no point retrying
            print(f"FAILED ({exc})")
            _print_load_hint(model_name, exc_str)
            return None

        # The Rust tokenizers library is too old to parse this tokenizer.json.
        # Try the pure-Python slow tokenizer instead.
        print(f"\n    fast tokenizer parse error — retrying with slow tokenizer ...", end=" ", flush=True)

    # --- attempt 2: slow (Python) tokenizer ---
    try:
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        print("OK (slow)")
        print(
            f"    Note: consider upgrading to silence this:  "
            f"pip install -U tokenizers transformers"
        )
        return tok
    except Exception as exc2:
        exc2_str = str(exc2)
        # SentencePiece / blobfile missing → give targeted pip hint then bail;
        # these are hard runtime import failures that a retry won't fix.
        if _is_missing_dep_error(exc2_str):
            print(f"FAILED ({exc2})")
            _print_load_hint(model_name, exc2_str)
            return None

        # Any other slow-tokenizer failure — fall through to hint + None
        print(f"FAILED ({exc2})")
        _print_load_hint(model_name, exc2_str)
        return None


def _is_missing_dep_error(exc_str: str) -> bool:
    """Return True when the error is a missing optional dependency."""
    _MISSING_DEP_HINTS = (
        "blobfile is not installed",
        "sentencepiece",
        "SentencePieceExtractor requires",
        "tiktoken",
        "protobuf",
    )
    return any(h.lower() in exc_str.lower() for h in _MISSING_DEP_HINTS)


def _print_load_hint(model_name: str, exc_str: str) -> None:
    """Print a targeted, actionable fix suggestion based on the error message."""
    exc_lower = exc_str.lower()

    if any(h in exc_str for h in ("did not match", "untagged enum", "ModelWrapper")):
        print(
            "\n    Fix: the installed `tokenizers` Rust library is too old to parse\n"
            "    this model's tokenizer.json. Upgrade and retry:\n"
            "        pip install -U tokenizers transformers\n"
        )
    elif "blobfile is not installed" in exc_lower:
        print(
            "\n    Fix: TikToken requires blobfile as a backend dependency.\n"
            "    Install it with:\n"
            "        pip install blobfile\n"
            "    If the model also uses SentencePiece, install both:\n"
            "        pip install sentencepiece blobfile tiktoken\n"
        )
    elif "sentencepiece" in exc_lower or "sentencepieceextractor" in exc_lower:
        print(
            "\n    Fix: this model uses a SentencePiece vocabulary (.model file)\n"
            "    but the library is not installed. Fix with:\n"
            "        pip install sentencepiece\n"
            "    You may need to restart your Python process afterwards.\n"
        )
    elif "tiktoken" in exc_lower:
        print(
            "\n    Fix: this model requires the tiktoken library:\n"
            "        pip install tiktoken\n"
        )
    elif "protobuf" in exc_lower:
        print(
            "\n    Fix: this model requires protobuf:\n"
            "        pip install protobuf\n"
        )
    elif "401" in exc_str or "authorization" in exc_lower or "credentials" in exc_lower:
        print(
            "\n    Fix: this model requires authentication.\n"
            "        huggingface-cli login\n"
            "    then re-run the script.\n"
        )
    elif "404" in exc_str or "not found" in exc_lower:
        print(
            f"\n    Fix: model '{model_name}' was not found on HuggingFace Hub.\n"
            f"    Check the model ID at https://huggingface.co/{model_name}\n"
        )
    else:
        print(
            "\n    Tip: try installing all common tokenizer backends:\n"
            "        pip install -U tokenizers transformers sentencepiece "
            "blobfile tiktoken protobuf\n"
            "    and check your internet connection.\n"
        )


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def bar(rate: float, max_rate: float, width: int = 30) -> str:
    filled = int(round(rate / max_rate * width)) if max_rate > 0 else 0
    return "█" * filled + "░" * (width - filled)


def print_summary(results: List[ModelResult]) -> None:
    valid = [r for r in results if r.error is None]
    print("\n" + "=" * 70)
    print("  FERTILITY RATE COMPARISON")
    print("=" * 70)

    if not valid:
        print("  No successful results to display.")
        return

    max_rate = max(r.fertility_rate for r in valid)

    if HAS_TABULATE:
        rows = []
        for r in valid:
            rows.append([
                r.model_name,
                f"{r.total_words:,}",
                f"{r.total_tokens:,}",
                f"{r.fertility_rate:.4f}",
                bar(r.fertility_rate, max_rate),
            ])
        print(tabulate(
            rows,
            headers=["Model", "Words", "Tokens", "Fertility", ""],
            tablefmt="simple",
        ))
    else:
        header = f"  {'Model':<45} {'Words':>8} {'Tokens':>9} {'Rate':>8}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for r in valid:
            print(
                f"  {r.model_name:<45} {r.total_words:>8,} "
                f"{r.total_tokens:>9,} {r.fertility_rate:>8.4f}  "
                f"{bar(r.fertility_rate, max_rate)}"
            )

    if len(valid) > 1:
        best = min(valid, key=lambda r: r.fertility_rate)
        worst = max(valid, key=lambda r: r.fertility_rate)
        print(f"\n  Most compact  : {best.model_name} ({best.fertility_rate:.4f})")
        print(f"  Most expansive: {worst.model_name} ({worst.fertility_rate:.4f})")

    for r in results:
        if r.error:
            print(f"\n  [ERROR] {r.model_name}: {r.error}")

    print("=" * 70 + "\n")


def print_per_sentence(result: ModelResult) -> None:
    if not result.per_sentence:
        return
    print(f"\n  Per-sentence breakdown — {result.model_name}")
    print("  " + "-" * 60)
    for i, s in enumerate(result.per_sentence, 1):
        preview = (s.sentence[:60] + "…") if len(s.sentence) > 63 else s.sentence
        print(
            f"  [{i:>3}] rate={s.fertility_rate:.3f}  "
            f"words={s.word_count:<4} tokens={s.token_count:<5}  \"{preview}\""
        )


def save_csv(results: List[ModelResult], path: str, per_sentence: bool) -> None:
    p = Path(path)
    with p.open("w", newline="", encoding="utf-8") as f:
        if per_sentence:
            writer = csv.writer(f)
            writer.writerow([
                "model", "sentence_index", "sentence",
                "word_count", "token_count", "fertility_rate",
            ])
            for r in results:
                if r.error:
                    continue
                for i, s in enumerate(r.per_sentence, 1):
                    writer.writerow([
                        r.model_name, i, s.sentence,
                        s.word_count, s.token_count, s.fertility_rate,
                    ])
        else:
            writer = csv.writer(f)
            writer.writerow([
                "model", "total_words", "total_tokens",
                "fertility_rate", "error",
            ])
            for r in results:
                writer.writerow([
                    r.model_name, r.total_words, r.total_tokens,
                    f"{r.fertility_rate:.6f}" if not r.error else "",
                    r.error or "",
                ])
    print(f"  Results saved to: {p.resolve()}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute tokenizer fertility rate for HuggingFace models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to the input text file (UTF-8).",
    )
    parser.add_argument(
        "--models", "-m", nargs="+", required=True,
        help="One or more HuggingFace model IDs (e.g. gpt2 bert-base-uncased).",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Optional path to save results as a CSV file.",
    )
    parser.add_argument(
        "--per-sentence", action="store_true",
        help="Also compute and display per-sentence fertility rates.",
    )
    parser.add_argument(
        "--encoding", default="utf-8",
        help="Text file encoding (default: utf-8).",
    )
    parser.add_argument(
        "--chinese", action="store_true",
        help="Use jieba tokenizer for Chinese word segmentation instead of whitespace split.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Read corpus
    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"Input file not found: {input_path}")
    text = input_path.read_text(encoding=args.encoding)
    if not text.strip():
        sys.exit("Input file is empty.")

    is_chinese = args.chinese

    print(f"Is chinese: {is_chinese}")
    if not is_chinese:
        # check the filename
        if args.input.endswith('.zh'):
            args.chinese = True
    print(f"Is chinese: {args.chinese}")

    word_count = count_words(text, is_chinese=args.chinese)
    print(f"\nCorpus: {input_path.name}  |  {word_count:,} words  |  {len(text):,} characters")
    print(f"Mode   : {'Chinese (jieba)' if args.chinese else 'Whitespace split'}")
    print(f"Models : {', '.join(args.models)}\n")

    # Process each model
    results: List[ModelResult] = []
    for model_name in args.models:
        tokenizer = load_tokenizer(model_name)
        if tokenizer is None:
            results.append(ModelResult(
                model_name=model_name,
                total_words=word_count,
                total_tokens=0,
                fertility_rate=0.0,
                error="Failed to load tokenizer",
            ))
            continue

        total_words, total_tokens, fertility, sentence_stats = compute_fertility(
            text, tokenizer, per_sentence=args.per_sentence, is_chinese=args.chinese
        )
        results.append(ModelResult(
            model_name=model_name,
            total_words=total_words,
            total_tokens=total_tokens,
            fertility_rate=fertility,
            per_sentence=sentence_stats,
        ))

    # Display
    print_summary(results)

    if args.per_sentence:
        for r in results:
            if not r.error:
                print_per_sentence(r)

    # Save
    if args.output:
        save_csv(results, args.output, per_sentence=args.per_sentence)


if __name__ == "__main__":
    main()
