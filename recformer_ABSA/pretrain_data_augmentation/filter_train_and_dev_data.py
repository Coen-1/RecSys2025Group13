'''
in train.json and dev.json keep only the items in the sequences that are present in the augmented_data/meta_data.json
keep the original structure of the nested list.

'''

import json
from pathlib import Path
import logging
import ijson
from tqdm import tqdm
import random

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

AUGMENTED_METADATA_FILE = Path("data/augmented_data/meta_data.json")
TRAIN_FILE = Path("data/train.json")
DEV_FILE = Path("data/dev.json")

OUTPUT_DIR = Path("data/augmented_data")
FILTERED_TRAIN_FILE = OUTPUT_DIR / "train.json"
FILTERED_DEV_FILE = OUTPUT_DIR / "dev.json"

DEV_SPLIT_PERCENT = 0.1

def filter_and_split_files():
    """
    Filters train.json and dev.json, then splits the combined result
    into new training and development sets.
    """
    OUTPUT_DIR.mkdir(exist_ok=True)

    logging.info("Loading product IDs from %s...", AUGMENTED_METADATA_FILE)
    if not AUGMENTED_METADATA_FILE.exists():
        logging.error(f"Metadata file not found: {AUGMENTED_METADATA_FILE}")
        return

    with AUGMENTED_METADATA_FILE.open("rb") as f:
        augmented_product_ids = {key for key, _ in ijson.kvitems(f, "")}
    logging.info("Loaded %d product IDs.", len(augmented_product_ids))

    all_filtered_sequences = []
    for source_file in [TRAIN_FILE, DEV_FILE]:
        if not source_file.exists():
            logging.warning(f"Source file not found, skipping: {source_file}")
            continue

        logging.info("Filtering %s...", source_file)
        with source_file.open("r") as f:
            original_sequences = json.load(f)

        for sequence in tqdm(original_sequences, desc=f"Filtering {source_file.name}"):
            if not isinstance(sequence, list):
                continue

            filtered_sequence = [
                pid for pid in sequence if pid in augmented_product_ids
            ]
            
            if filtered_sequence:
                all_filtered_sequences.append(filtered_sequence)

    logging.info("Found a total of %d non-empty sequences.", len(all_filtered_sequences))

    logging.info("Shuffling and splitting data into training and development sets...")
    random.shuffle(all_filtered_sequences)

    split_index = int(len(all_filtered_sequences) * DEV_SPLIT_PERCENT)
    
    dev_set = all_filtered_sequences[:split_index]
    train_set = all_filtered_sequences[split_index:]

    logging.info(
        "New training set size: %d sequences. New development set size: %d sequences.",
        len(train_set),
        len(dev_set)
    )

    logging.info("Saving new training set to %s...", FILTERED_TRAIN_FILE)
    with FILTERED_TRAIN_FILE.open("w") as f:
        json.dump(train_set, f)

    logging.info("Saving new development set to %s...", FILTERED_DEV_FILE)
    with FILTERED_DEV_FILE.open("w") as f:
        json.dump(dev_set, f)

    logging.info("Data filtering and splitting complete.")

if __name__ == "__main__":
    filter_and_split_files()