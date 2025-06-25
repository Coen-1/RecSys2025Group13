'''
create a new metadata file that take the original file and adds strings with aspect sentiment
'''

import json
import ijson
from pathlib import Path
import logging
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


ORIGINAL_METADATA_FILE = Path("data/meta_data.json")
SENTIMENTS_FILE = Path("sentiments.json")
NEW_METADATA_FILE = Path("data/augmented_data/meta_data.json")


def count_products(data_file: Path) -> int:
    """Stream-count the number of products in the JSON file."""
    logging.info("Counting total products for progress bar...")
    with data_file.open("rb") as f:
        return sum(1 for _ in ijson.kvitems(f, ""))

def combine_data():
    """
    Combines metadata with sentiment analysis results by appending sentiments
    to an existing text field and filtering for products with sentiments.
    """
    if not SENTIMENTS_FILE.exists():
        logging.error("Sentiments file not found at %s", SENTIMENTS_FILE)
        return

    if not ORIGINAL_METADATA_FILE.exists():
        logging.error("Original metadata file not found at %s", ORIGINAL_METADATA_FILE)
        return

    logging.info("Loading sentiments from %s...", SENTIMENTS_FILE)
    try:
        with SENTIMENTS_FILE.open("r") as f:
            sentiments = json.load(f)
        logging.info("Loaded %d sentiment records.", len(sentiments))
    except (json.JSONDecodeError, IOError) as e:
        logging.error("Error loading sentiments file: %s", e)
        return

    products_with_sentiments = {rid.split('_review_')[0] for rid in sentiments.keys()}
    logging.info("Found %d products with sentiment labels.", len(products_with_sentiments))

    total_products = count_products(ORIGINAL_METADATA_FILE)

    logging.info("Streaming metadata from %s and combining...", ORIGINAL_METADATA_FILE)
    NEW_METADATA_FILE.parent.mkdir(exist_ok=True)

    try:
        with ORIGINAL_METADATA_FILE.open("rb") as infile, NEW_METADATA_FILE.open("w") as outfile:
            outfile.write("{\n")
            first_product = True

            product_iterator = ijson.kvitems(infile, "")
            for product_id, product_data in tqdm(product_iterator, total=total_products, desc="Processing products"):
                if not isinstance(product_data, dict):
                    continue

                if product_id not in products_with_sentiments:
                    continue

                product_sentiments = []
                for review_key in product_data:
                    if not review_key.startswith("review_"):
                        continue

                    review_id = f"{product_id}_{review_key}"
                    if review_id in sentiments:
                        for s in sentiments[review_id]:
                            product_sentiments.append(f"{s['aspect']}: {s['sentiment']}")
                
                if not product_sentiments:
                    continue
                
                sentiment_string = ", ".join(product_sentiments)

                output_data = {}
                if product_data.get("title"):
                    output_data["title"] = str(product_data.get("title")) + " | " + sentiment_string
                else:
                    output_data["title"] = sentiment_string

                if product_data.get("brand"):
                    output_data["brand"] = product_data.get("brand")

                if product_data.get("category"):
                    output_data["category"] = product_data.get("category")

                if not first_product:
                    outfile.write(",\n")

                outfile.write(f'  "{product_id}": ')
                json.dump(output_data, outfile)
                
                first_product = False
            
            outfile.write("\n}")
        
        logging.info("Successfully created %s", NEW_METADATA_FILE)

    except (IOError, ijson.JSONError) as e:
        logging.error("An error occurred during processing: %s", e)


def main():
    combine_data()

if __name__ == "__main__":
    main()

'''
steam ORIGINAL_METADATA_FILE and show output



'''
