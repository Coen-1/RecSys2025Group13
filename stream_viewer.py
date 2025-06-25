import ijson
import json
from pathlib import Path
import logging
import argparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def stream_and_print_metadata(metadata_file: Path):
    """
    Streams and prints the content of the metadata file.
    """
    if not metadata_file.exists():
        logging.error("Metadata file not found at %s", metadata_file)
        return

    logging.info("Streaming and printing metadata from %s...", metadata_file)

    try:
        with metadata_file.open("rb") as f:
            for product_id, product_data in ijson.kvitems(f, ""):
                print(f"Product ID: {product_id}")
                print(json.dumps(product_data, indent=2))
                print("-" * 40)
    except (IOError, ijson.JSONError) as e:
        logging.error("An error occurred during streaming: %s", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream and view a JSON file.")
    parser.add_argument(
        "file_path",
        type=str,
        nargs="?",
        default="data/meta_data.json",
        help="Path to the JSON file to stream. Defaults to data/meta_data.json.",
    )
    args = parser.parse_args()

    metadata_file = Path(args.file_path)
    stream_and_print_metadata(metadata_file) 