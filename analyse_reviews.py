import ijson
from pathlib import Path
import logging
import argparse
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def count_products_for_tqdm(data_file: Path) -> int:
    """Stream-count the number of products in the JSON file for tqdm progress bar."""
    logging.info("Counting total products for progress bar...")
    try:
        with data_file.open("rb") as f:
            return sum(1 for _ in ijson.kvitems(f, ""))
    except (IOError, ijson.JSONError) as e:
        logging.error("Could not count products: %s", e)
        return 0


def analyse_reviews(metadata_file: Path):
    """
    Streams a metadata file to count products, reviews, and analyze review coverage.
    """
    if not metadata_file.exists():
        logging.error("Metadata file not found at %s", metadata_file)
        return

    total_products = 0
    total_reviews = 0
    products_with_reviews = 0
    products_without_reviews = 0

    product_count_for_progress = count_products_for_tqdm(metadata_file)
    if product_count_for_progress == 0:
        logging.warning("No products found to analyze.")
        return

    logging.info("Streaming and analyzing metadata from %s...", metadata_file)

    try:
        with metadata_file.open("rb") as f:
            product_iterator = ijson.kvitems(f, "")
            for _, product_data in tqdm(
                product_iterator, total=product_count_for_progress, desc="Analyzing products"
            ):
                total_products += 1

                if not isinstance(product_data, dict):
                    products_without_reviews += 1
                    continue

                review_count_for_product = 0
                for key in product_data.keys():
                    if str(key).startswith("review_"):
                        review_count_for_product += 1

                total_reviews += review_count_for_product

                if review_count_for_product > 0:
                    products_with_reviews += 1
                else:
                    products_without_reviews += 1

        print("\n--- Analysis Complete ---")
        print(f"Total products: {total_products}")
        print(f"Total reviews: {total_reviews}")
        print(f"Products with reviews: {products_with_reviews}")
        print(f"Products without reviews: {products_without_reviews}")
        print("-------------------------\n")

    except (IOError, ijson.JSONError) as e:
        logging.error("An error occurred during streaming: %s", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze a large JSON metadata file for product and review counts."
    )
    parser.add_argument(
        "file_path",
        type=str,
        help="Path to the JSON metadata file to analyze.",
    )
    args = parser.parse_args()

    metadata_file = Path(args.file_path)
    analyse_reviews(metadata_file)



