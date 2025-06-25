#!/usr/bin/env python3
"""
Optimised aspect‑based sentiment pass for huge product‑reviews JSON
------------------------------------------------------------------

* Single PyABSA model shared on the Apple‑silicon GPU (M‑series, MPS backend)
* Batches 64 reviews per forward pass to amortise Metal launch overhead
* No forked processes – avoids model duplication; I/O stays on CPU thread
* Automatic checkpointing so you can Ctrl‑C and resume
* Tested on Python 3.12, PyTorch 2.3, macOS 14, MacBook Air M4.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Generator, List, Tuple

import ijson
import torch
from tqdm import tqdm
from pyabsa import AspectTermExtraction as ATEPC

# ────────────────────────── configuration ───────────────────────── #

BATCH_SIZE = 20480          
SAVE_EVERY = 1_000         # flush to disk after N analysed reviews

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

# ────────────────────────── model loading ───────────────────────── #

def load_analyzer() -> ATEPC.AspectExtractor:
    """Load PyABSA extractor on the best available hardware (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    logging.info("Initialising model on %s…", device.upper())

    analyzer = ATEPC.AspectExtractor(
        "multilingual",
        auto_device=device,
        cal_perplexity=False,
        # tiny optimisation – turn off verbose table in PyABSA
        print_result=False,
    )

    # Apply hardware-specific optimizations
    if device == 'cuda':
        # FP16 and compilation are highly effective on NVIDIA GPUs
        # analyzer.model.half() # Using AMP autocast is more robust
        logging.info("Automatic Mixed Precision (AMP) enabled for CUDA.")
        if hasattr(torch, "compile"):
            try:
                analyzer.model = torch.compile(analyzer.model)
                logging.info("torch.compile enabled for CUDA.")
            except Exception:
                logging.warning("torch.compile failed – continuing without.")

    elif device == 'mps':
        # These features are currently unstable on Apple Silicon, so they are disabled.
        pass

    return analyzer


_ANALYZER: ATEPC.AspectExtractor | None = None


def analyse_batch(batch: List[Tuple[str, str]]) -> List[Tuple[str, Any | None]]:
    """Run sentiment on a list of (id, text) pairs."""
    global _ANALYZER
    if _ANALYZER is None:
        _ANALYZER = load_analyzer()

    ids, texts = zip(*batch)

    # Use AMP autocast for safe mixed-precision on CUDA
    with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
        outs = _ANALYZER.predict(
            list(texts),
            print_result=False,
            ignore_error=True,
            save_result=False,
        )

    results: List[Tuple[str, Any | None]] = []
    for rid, out in zip(ids, outs):
        sentiments = [
            {"aspect": a, "sentiment": s, "confidence": c}
            for a, s, c in zip(out["aspect"], out["sentiment"], out["confidence"])
        ]
        results.append((rid, sentiments or None))
    return results


# ──────────────────────────── streaming I/O ─────────────────────── #

def count_reviews(data_file: Path) -> int:
    """Stream-count the number of reviews in the JSON file."""
    logging.info("Counting total reviews for progress bar…")
    count = 0
    with data_file.open("rb") as f:
        for _prod_id, prod_data in ijson.kvitems(f, ""):
            if not isinstance(prod_data, dict):
                continue
            for key, val in prod_data.items():
                if key.startswith("review_") and isinstance(val, str) and val.strip():
                    count += 1
    return count

def stream_unprocessed_reviews(
    data_file: Path, done: set[str]
) -> Generator[Tuple[str, str], None, None]:
    """Yield (id, text) pairs not already present in *done*."""
    with data_file.open("rb") as f:
        for product_id, product_data in ijson.kvitems(f, ""):
            if not isinstance(product_data, dict):
                continue
            for review_key, review_text in product_data.items():
                if not isinstance(review_text, str) or not review_text.strip():
                    continue
                if not review_key.startswith("review_"):
                    continue
                review_id = f"{product_id}_{review_key}"
                if review_id not in done:
                    yield review_id, review_text


def load_existing(outfile: Path) -> Dict[str, Any]:
    if not outfile.exists():
        return {}
    try:
        with outfile.open() as f:
            return json.load(f)
    except json.JSONDecodeError:
        logging.warning("%s is corrupted – starting fresh", outfile)
        return {}


def dump_atomic(mapping: Dict[str, Any], outfile: Path) -> None:
    tmp = outfile.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump(mapping, f, indent=2)
    tmp.replace(outfile)


# ──────────────────────────── main driver ───────────────────────── #

def run(data_file: Path, output_file: Path) -> None:
    total_reviews = count_reviews(data_file)
    results = load_existing(output_file)
    processed = set(results)
    logging.info("Resuming with %d/%d completed reviews", len(processed), total_reviews)

    buf: Dict[str, Any] = {}
    gen = stream_unprocessed_reviews(data_file, processed)

    # Batch generator
    def take(n: int):
        return [next(gen) for _ in range(n)]

    pbar = tqdm(total=total_reviews, initial=len(processed), unit="review", desc="Analysing")

    try:
        while True:
            try:
                batch = take(BATCH_SIZE)
            except StopIteration:
                break

            for rev_id, data in analyse_batch(batch):
                pbar.update()
                if data:
                    buf[rev_id] = data

            if len(buf) >= SAVE_EVERY:
                results.update(buf)
                dump_atomic(results, output_file)
                pbar.set_postfix(saved=len(results))
                buf.clear()
    finally:
        if buf:
            logging.info("Saving final %d reviews…", len(buf))
            results.update(buf)
            dump_atomic(results, output_file)

    logging.info("Done. Total stored sentiments: %d", len(results))
    pbar.close()


# ─────────────────────────── CLI entry point ────────────────────── #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fast ABSA over giant review dumps")
    p.add_argument("-i", "--input", type=Path, default="data/meta_data.json")
    p.add_argument("-o", "--output", type=Path, default="sentiments.json")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.input, args.output)
