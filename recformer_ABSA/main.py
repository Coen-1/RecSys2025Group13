#!/usr/bin/env python3
"""
High-Performance ABSA Sentiment Analysis for Large JSON Reviews
---------------------------------------------------------------

* Optimised for NVIDIA A100 and Apple Silicon (M-series)
* Batch-processed inference with sub-batching for max GPU utilisation
* Multiprocessing queue for concurrent data loading
* Automatic checkpointing for safe stop/resume
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
from multiprocessing import Process, Queue
from pyabsa import AspectTermExtraction as ATEPC

# ────────────────────────── configuration ───────────────────────── #

BATCH_SIZE = 20480
SUB_BATCH_SIZE = 16384
SAVE_EVERY = 1000
STOP_TOKEN = "__STOP__"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

# ────────────────────────── model loading ───────────────────────── #

_ANALYZER: ATEPC.AspectExtractor | None = None

def load_analyzer() -> ATEPC.AspectExtractor:
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
        print_result=False,
    )

    if device == "cuda":
        torch.set_float32_matmul_precision("high")
        if hasattr(torch, "compile"):
            try:
                analyzer.model = torch.compile(analyzer.model)
                logging.info("torch.compile enabled for CUDA.")
            except Exception as e:
                logging.warning(f"torch.compile failed: {e}")

    return analyzer


def analyse_batch(batch: List[Tuple[str, str]]) -> List[Tuple[str, Any | None]]:
    global _ANALYZER
    if _ANALYZER is None:
        _ANALYZER = load_analyzer()

    ids, texts = zip(*batch)
    results: List[Tuple[str, Any | None]] = []

    with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
        for i in range(0, len(texts), SUB_BATCH_SIZE):
            sub_ids = ids[i:i+SUB_BATCH_SIZE]
            sub_texts = texts[i:i+SUB_BATCH_SIZE]
            outs = _ANALYZER.predict(
                list(sub_texts),
                print_result=False,
                ignore_error=True,
                save_result=False,
            )
            for rid, out in zip(sub_ids, outs):
                sentiments = [
                    {"aspect": a, "sentiment": s, "confidence": c}
                    for a, s, c in zip(out["aspect"], out["sentiment"], out["confidence"])
                ]
                results.append((rid, sentiments or None))

    return results

# ────────────────────────── review streaming ────────────────────── #

def stream_unprocessed_reviews(
    data_file: Path, done: set[str], queue: Queue
) -> None:
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
                    queue.put((review_id, review_text))
    queue.put(STOP_TOKEN)

# ──────────────────────────── checkpointing ─────────────────────── #

def count_reviews(data_file: Path) -> int:
    logging.info("Counting total reviews for progress bar…")
    count = 0
    with data_file.open("rb") as f:
        for _prod_id, prod_data in ijson.kvitems(f, ""):
            if isinstance(prod_data, dict):
                for key, val in prod_data.items():
                    if key.startswith("review_") and isinstance(val, str) and val.strip():
                        count += 1
    return count

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

# ────────────────────────────── runner ──────────────────────────── #

def run(data_file: Path, output_file: Path) -> None:
    total_reviews = count_reviews(data_file)
    results = load_existing(output_file)
    processed = set(results)
    logging.info("Resuming with %d/%d completed reviews", len(processed), total_reviews)

    buf: Dict[str, Any] = {}
    queue: Queue = Queue(maxsize=5000)
    producer = Process(target=stream_unprocessed_reviews, args=(data_file, processed, queue))
    producer.start()

    pbar = tqdm(total=total_reviews, initial=len(processed), unit="review", desc="Analysing")

    try:
        batch: List[Tuple[str, str]] = []
        while True:
            item = queue.get()
            if item == STOP_TOKEN:
                break
            batch.append(item)
            if len(batch) >= BATCH_SIZE:
                for rev_id, data in analyse_batch(batch):
                    pbar.update()
                    if data:
                        buf[rev_id] = data
                if len(buf) >= SAVE_EVERY:
                    results.update(buf)
                    dump_atomic(results, output_file)
                    pbar.set_postfix(saved=len(results))
                    buf.clear()
                batch.clear()
    finally:
        if batch:
            for rev_id, data in analyse_batch(batch):
                pbar.update()
                if data:
                    buf[rev_id] = data
        if buf:
            logging.info("Saving final %d reviews…", len(buf))
            results.update(buf)
            dump_atomic(results, output_file)
        pbar.close()
        producer.join()
    logging.info("Done. Total stored sentiments: %d", len(results))

# ────────────────────────────── CLI ────────────────────────────── #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fast ABSA over large review JSON")
    p.add_argument("-i", "--input", type=Path, default="data/meta_data.json")
    p.add_argument("-o", "--output", type=Path, default="sentiments.json")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run(args.input, args.output)
