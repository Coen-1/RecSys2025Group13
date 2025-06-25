import ijson
import json
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

METADATA_FILE = Path("data/meta_data.json")

def stream_and_print_metadata():
    """
    Streams and prints the content of the metadata file.
    """
    if not METADATA_FILE.exists():
        logging.error("Metadata file not found at %s", METADATA_FILE)
        return

    logging.info("Streaming and printing metadata from %s...", METADATA_FILE)

    try:
        with METADATA_FILE.open("rb") as f:
            for product_id, product_data in ijson.kvitems(f, ""):
                print(f"Product ID: {product_id}")
                print(json.dumps(product_data, indent=2))
                print("-" * 40)
    except (IOError, ijson.JSONError) as e:
        logging.error("An error occurred during streaming: %s", e)

if __name__ == "__main__":
    stream_and_print_metadata() 