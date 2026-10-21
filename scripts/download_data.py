"""Download the public Olist Brazilian e-commerce CSVs into data/raw/.

Source: Olist via Kaggle (CC BY-NC-SA 4.0), mirrored on Hugging Face.
Files that already exist locally are skipped, so this is safe to re-run.
The files are treated purely as data: they are only ever read as CSV by DuckDB.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

BASE_URL = "https://huggingface.co/datasets/aviahYadler/Olist_Ecommerce_Dataset/resolve/main"

FILES: list[str] = [
    "olist_customers_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_order_payments_dataset.csv",
    "olist_order_reviews_dataset.csv",
    "olist_orders_dataset.csv",
    "olist_products_dataset.csv",
    "olist_sellers_dataset.csv",
    "product_category_name_translation.csv",
]

DEFAULT_DEST = Path(__file__).resolve().parent.parent / "data" / "raw"


def download(dest: Path, force: bool = False) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = dest / name
        if target.exists() and target.stat().st_size > 0 and not force:
            print(f"  skip   {name} (already present, {target.stat().st_size / 1e6:.1f} MB)")
            continue
        url = f"{BASE_URL}/{name}"
        tmp = target.with_suffix(".part")
        print(f"  fetch  {name} ...", end="", flush=True)
        with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as fh:
            shutil.copyfileobj(resp, fh)
        tmp.rename(target)
        print(f" {target.stat().st_size / 1e6:.1f} MB")
    print(f"Done. Files in {dest}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()
    download(args.dest, args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
