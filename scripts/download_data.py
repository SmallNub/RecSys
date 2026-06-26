import argparse
import gzip
import os
import shutil
import urllib.request
from huggingface_hub import hf_hub_download

HF_ACCESS_TOKEN = os.getenv("HF_ACCESS_TOKEN")
HF_REPO_2023 = "McAuley-Lab/Amazon-Reviews-2023"

# McAuley lab 2014 dataset hosted on SNAP Stanford
AMAZON_2014_BASE = "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--base_dir",
    default="datasets",
    help="Base directory to download into",
)
parser.add_argument(
    "--categories",
    nargs="+",
    default=["Arts_Crafts_and_Sewing"],
    help="Categories to download",
)
parser.add_argument(
    "--year",
    type=int,
    default=2023,
    choices=[2023, 2014],
    help="Dataset year: 2023 (HuggingFace) or 2014 (McAuley lab direct download)",
)
args = parser.parse_args()


# ---------------------------------------------------------------------------
# Amazon 2023
# ---------------------------------------------------------------------------


def category_files_2023(category):
    return [
        f"raw/meta_categories/meta_{category}.jsonl",
        # f"raw/review_categories/{category}.jsonl",
        f"benchmark/5core/last_out/{category}.train.csv",
        f"benchmark/5core/last_out/{category}.valid.csv",
        f"benchmark/5core/last_out/{category}.test.csv",
    ]


def download_2023(categories, local_dir):
    files = [f for cat in categories for f in category_files_2023(cat)]
    print(f"Downloading {len(files)} files for: {categories}")
    print(f"Destination: {local_dir}")
    for filename in files:
        print(f"  -> {filename}")
        hf_hub_download(
            repo_id=HF_REPO_2023,
            repo_type="dataset",
            filename=filename,
            token=HF_ACCESS_TOKEN,
            local_dir=local_dir,
        )


# ---------------------------------------------------------------------------
# Amazon 2014
# ---------------------------------------------------------------------------


def category_files_2014(category):
    # 5-core ratings and metadata; .gz files are decompressed after download
    return [
        f"reviews_{category}_5.json.gz",
        f"meta_{category}.json.gz",
    ]


def download_2014(categories, local_dir):
    os.makedirs(local_dir, exist_ok=True)
    for category in categories:
        for gz_name in category_files_2014(category):
            gz_path = os.path.join(local_dir, gz_name)
            json_path = gz_path[:-3]  # strip .gz

            if os.path.exists(json_path):
                print(f"  Already exists, skipping: {json_path}")
                continue

            url = f"{AMAZON_2014_BASE}/{gz_name}"
            print(f"  -> {url}")
            urllib.request.urlretrieve(url, gz_path)

            print(f"  Decompressing {gz_name} ...")
            # Decompress streaming gzip file chunk-by-chunk to save RAM
            with gzip.open(gz_path, "rb") as f_in, open(json_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            os.remove(gz_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if args.year == 2014:
    local_dir = os.path.join(args.base_dir, "amazon-2014")
    print(f"Downloading Amazon 2014 -> {local_dir}")
    download_2014(args.categories, local_dir)
else:
    local_dir = os.path.join(args.base_dir, "amazon-2023")
    download_2023(args.categories, local_dir)

print("Done.")
