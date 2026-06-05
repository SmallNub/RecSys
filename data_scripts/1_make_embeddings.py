import os
import pickle
import tempfile
from functools import partial

import fsspec
import hydra
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.utils.tools import patch_fsspec


def preprocess(row, eos):
    lines = []

    if row["title"]:
        lines.append(f"Title: {row['title']}")

    if row["store"]:
        lines.append(f"Store: {row['store']}")

    if row["average_rating"]:
        lines.append(
            f"Rating: {row['average_rating']}/5 ({row['rating_number']} ratings)"
        )

    if row["price"] and not np.isnan(row["price"]):
        lines.append(f"Price: {row['price']}$")

    if len(row["categories"]):
        lines.append(f"Category: {' > '.join(row['categories'][0])}")

    if "features" in row and len(row["features"]):
        lines.append(f"Features: {' '.join(row['features'])}")

    return "\n".join(lines) + eos


def get_model(fs, model_folder):
    if fs.exists(model_folder) and not os.path.exists(model_folder):
        print("Model folder exist but not locally, downloading from ", model_folder)
        tmpdir = tempfile.mkdtemp()
        print("Downloading to ", tmpdir)
        files_to_download = fs.ls(model_folder)
        print("Files to download:", files_to_download)
        [fs.get(f, tmpdir, recursive=True) for f in tqdm(files_to_download)]
        print("Download complete:", os.listdir(tmpdir))

        return tmpdir

    # Probably a model_id that will be loaded by SentenceTransformer
    return model_folder


@hydra.main(
    config_path="../configs", config_name="1_make_embeddings", version_base="1.2"
)
def main(config):
    # Could loop over multiple categories here ?
    embed(config)


def embed(config):
    patch_fsspec()
    fs = fsspec.filesystem(config.paths.protocol)

    # Writing directly to embeddings as it's already filtered
    output_file = config.paths.embeddings_tplt.format(
        emb_method=config.emb_name, category=config.category
    )
    print("Outputs will be written to : ", output_file)

    # Try removing the embeddings, we don't want to have multiple versions in the same folder
    if config.force:
        try:
            fs.rm(output_file, recursive=True)
        except (FileNotFoundError, OSError):
            pass
    else:
        if fs.exists(output_file):
            print(f"File already exists for {config.category}, skipping...")
            return

    print(f"Processing {config.category}...")

    print("Loading model.")

    model_dir = get_model(fs, config.model_folder)
    model = SentenceTransformer(model_dir, trust_remote_code=True).eval()

    print("Model loaded.")

    with fs.open(
        config.paths.unique_items_tplt.format(category=config.category), "rb"
    ) as f:
        items = pickle.load(f)

    pq_path = config.paths.meta_tplt.format(category=config.category)
    print("Input parquet path", pq_path)
    input_df = pd.read_parquet(pq_path, filesystem=fs)
    input_df = input_df[input_df["parent_asin"].isin(set(items))]

    print(
        f"Found {len(input_df)} items in the parquet file ({len(items)} in the pickle)."
    )

    sentences = input_df.apply(
        partial(preprocess, eos=model.tokenizer.eos_token), axis=1
    ).values.tolist()
    print("OK. Start processing...")

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    target_devices = (
        [f"cuda:{i}" for i in range(len(visible.split(",")))]
        if visible else ["cpu"]
    )
    print(f"Using devices: {target_devices}")

    chunk_size = config.batch_size * 256
    pool = model.start_multi_process_pool(target_devices=target_devices)
    chunks = [sentences[i:i + chunk_size] for i in range(0, len(sentences), chunk_size)]
    embeddings = np.concatenate([
        model.encode_multi_process(
            chunk, pool=pool, normalize_embeddings=True, batch_size=config.batch_size
        )
        for chunk in tqdm(chunks, desc="Embedding", unit="chunk")
    ])
    model.stop_multi_process_pool(pool)

    out_df = pd.DataFrame(
        {
            "product_id": input_df["parent_asin"],
            "embedding": list(embeddings),
            "categories": input_df["categories"],
        }
    )

    fs.makedirs(os.path.dirname(output_file), exist_ok=True)
    out_df.to_parquet(output_file, filesystem=fs, index=False)


if __name__ == "__main__":
    main()
