import pickle
import random
import fsspec
import hydra
import pandas as pd
from torch.utils.data import Dataset

from src.models import SpecialTokens
from src.utils.tools import patch_fsspec


def get_quantized(fs, path):
    df = pd.read_parquet(path, filesystem=fs)
    df = df.set_index("product_id")
    sorted_cols = sorted([col for col in df.columns if col.startswith("L")])
    df = df[sorted_cols]
    return df


def get_items_map(fs, path):
    with fs.open(path, "rb") as f:
        items = pickle.load(f)

    m = {}
    m["item_to_id"] = {item: i + len(SpecialTokens) for i, item in enumerate(items)}
    m["id_to_item"] = {id: item for item, id in m["item_to_id"].items()}
    return m


COL_ORDER = ["user_id", "timeline", "rating", "timestamp"]


class TrainDataset(Dataset):
    def __init__(self, df, pp_cfg, total_len):
        self.df = df
        self.pp_cls = hydra.utils.get_class(pp_cfg.pop("_cls_"))
        self.pp = self.pp_cls(**pp_cfg)
        self.total_len = total_len
        self.df_len = len(df)
        self.values = df.values

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        random_idx = random.randint(0, self.df_len - 1)
        row_vals = self.values[random_idx]
        row = {k: row_vals[i] for i, k in enumerate(COL_ORDER)}
        return self.pp(row)


class EvalDataset(Dataset):
    def __init__(self, df, pp_cfg):
        self.df = df
        self.pp_cls = hydra.utils.get_class(pp_cfg.pop("_cls_"))
        self.pp = self.pp_cls(**pp_cfg)
        self.records = self.df.to_dict("records")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.pp(self.records[idx])


def make_datasets(
    emb_id,
    quant_id,
    category,
    prepro_cfg,
    total_len,
    num_workers,  # Legacy argument
    paths,
    which=["train", "valid"],
):
    patch_fsspec()
    fs = fsspec.filesystem(paths.protocol)

    items_path = paths.unique_items_tplt.format(category=category)
    items_ref = get_items_map(fs, items_path)

    if quant_id is not None:
        quantizer_path = paths.semantic_ids_tplt.format(
            emb_method=emb_id, category=category, quant_method=quant_id
        )
        quantizer_ref = get_quantized(fs, quantizer_path)
    else:
        quantizer_ref = None

    datasets = {}

    for split in which:
        pp_cfg = prepro_cfg.copy()
        pp_cfg.update(
            {
                "quantizer_ref": quantizer_ref,
                "items_ref": items_ref,
                "split": split,
            }
        )

        path = paths.timelines_tplt.format(category=category, split=split)

        if split == "train":
            assert (
                total_len is not None
            ), "total_len must be specified for the training dataset."
            df = pd.read_parquet(path, filesystem=fs).reset_index()[COL_ORDER]
            df = df[df.timeline.apply(len) > 1]
            datasets["train"] = TrainDataset(df, pp_cfg, total_len)
        else:
            df = pd.read_parquet(path, filesystem=fs).reset_index()[COL_ORDER]
            df = df[df.timeline.apply(len) > 1]
            datasets[split] = EvalDataset(df, pp_cfg)

    return datasets
