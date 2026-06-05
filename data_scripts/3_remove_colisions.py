import collections
import copy
from collections import defaultdict
from pathlib import Path

import fsspec
import hydra
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.models.cosette import COSETTE
from src.utils.tools import patch_fsspec


def partition_list(lst, partition_sizes):
    if sum(partition_sizes) != len(lst):
        return "Error: Sum of partition sizes does not match list length"

    partitions = []
    start = 0
    for size in partition_sizes:
        partitions.append(lst[start : start + size])
        start += size

    return partitions


def find_latest_quant_method(base_dir: str, category: str):
    base = Path(base_dir)
    runs = [d for d in base.iterdir() if d.is_dir() and category in d.name]

    if not runs:
        raise ValueError(f"No runs found in {base_dir} for category {category}")

    latest_run = max(runs, key=lambda d: d.name)
    return latest_run.name


def check_collision(all_indices_tup):
    tot_item = len(all_indices_tup)
    tot_indice = len(set(all_indices_tup))
    return tot_item == tot_indice


def get_indices_count(all_indices_tup):
    indices_count = collections.defaultdict(int)
    for index in all_indices_tup:
        indices_count[index] += 1
    return indices_count


def get_collision_item(all_indices_tup):
    index2id = defaultdict(list)
    for i, index in enumerate(all_indices_tup):
        index2id[index].append(i)

    collision_item_groups = []
    for index in index2id:
        if len(index2id[index]) > 1:
            collision_item_groups.append(index2id[index])

    return collision_item_groups


@hydra.main(
    config_path="../configs", config_name="3_remove_collisions", version_base="1.2"
)
def main(config):
    patch_fsspec()
    fs = fsspec.filesystem(config.paths.protocol)

    quant_method = find_latest_quant_method(config.paths.ckpt_dir, config.data.category)
    print(f"Using quant method: {quant_method}")

    embeddings_path = config.paths.embeddings_tplt.format(
        emb_method=config.data.emb_method, category=config.data.category
    )
    quantized_path = config.paths.semantic_ids_tplt.format(
        emb_method=config.data.emb_method,
        quant_method=quant_method,
        category=config.data.category,
    )
    model_path = config.paths.semantic_model_tplt.format(
        emb_method=config.data.emb_method,
        quant_method=quant_method,
        category=config.data.category,
    )

    # Embeddings_df
    embs_df = pd.read_parquet(embeddings_path, filesystem=fs)
    embs_block = np.stack(embs_df["embedding"].values)

    with fs.open(model_path, "rb") as f:
        ckpt = torch.load(f, map_location=torch.device("cpu"), weights_only=False)

        model_cfg = ckpt.pop("config")
        state_dict = ckpt.pop("state_dict", None)

        model = COSETTE(
            embs_block=None,  # Provided at inference time
            # Dimensions
            in_dim=embs_block.shape[-1],
            layers=model_cfg.model.layers,
            # Quantization
            n_centroids_list=model_cfg.centroids.n_centroids_list,
            dropout_prob=model_cfg.optim.dropout_prob,
            loss_weights={
                "quantization": 1,
                "reconstruction": model_cfg.loss.reconstruction_weight,
                "contrastive": model_cfg.loss.contrastive_weight,
            },
            tau=model_cfg.loss.tau,
            bias=model_cfg.loss.bias,
            freeze_tau=model_cfg.loss.freeze_tau,
            freeze_bias=model_cfg.loss.freeze_bias,
            # Cluster assignment
            kmeans_init=model_cfg.centroids.kmeans_init,
            kmeans_iters=model_cfg.centroids.kmeans_iters,
            sk_epsilons=model_cfg.centroids.sk_epsilons,
            sk_iters=model_cfg.centroids.sk_iters,
        )

        # We didn't save the embeddings block
        model.load_state_dict(state_dict)

        model.to("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()

    all_indices = []
    all_indices_tup = []
    all_distances = []
    all_indices_tup_set = set()

    indices, all_distances = model.get_indices(
        embeddings=torch.from_numpy(embs_block).cuda(), use_sk=False
    )

    indices = indices.cpu().numpy()  # (N, L)
    all_distances = all_distances.cpu().numpy()  # (N, L, K)

    for index in indices:  # (N, L) -> (1, L)
        code = []
        for i, ind in enumerate(index):
            code.append(int(ind))

        all_indices.append(code)
        all_indices_tup.append(tuple(code))
        all_indices_tup_set.add(tuple(code))

    sort_distances_index = np.argsort(all_distances, axis=2)

    item_min_dis = defaultdict(list)

    for item, distances in tqdm(enumerate(all_distances), desc="cal distances"):
        for dis in distances:
            item_min_dis[item].append(np.min(dis))

    collision_item_groups = get_collision_item(all_indices_tup)
    all_collision_items = set()
    for collision_items in collision_item_groups:
        for item in collision_items:
            all_collision_items.add(item)

    print("collision items num: ", len(all_collision_items))

    tt = 0
    level = len(model_cfg.centroids.n_centroids_list) - 1
    max_num = model_cfg.centroids.n_centroids_list[0]

    while True:
        tot_item = len(all_indices_tup)
        tot_indice = len(set(all_indices_tup))
        print("Collision Rate", (tot_item - tot_indice) / tot_item)

        if check_collision(all_indices_tup):
            print("Exiting after", tt, "iterations.")
            break

        collision_item_groups = get_collision_item(all_indices_tup)

        print("len(collision_item_groups)", len(collision_item_groups))

        for collision_items in tqdm(collision_item_groups, desc="solve collision"):
            min_distances = []
            for i, item in enumerate(collision_items):
                min_distances.append(item_min_dis[item][level])

            min_index = np.argsort(np.array(min_distances))

            for i, m_index in enumerate(min_index):
                if i == 0:
                    continue

                item = collision_items[m_index]

                ori_code = copy.deepcopy(all_indices[item])

                num = i
                while tuple(ori_code) in all_indices_tup_set and num < max_num:
                    ori_code[level] = sort_distances_index[item][level][num]
                    num += 1

                for i in range(1, max_num):
                    if tuple(ori_code) in all_indices_tup_set:
                        ori_code = copy.deepcopy(all_indices[item])
                        ori_code[level - 1] = sort_distances_index[item][level - 1][i]

                    num = 0
                    while tuple(ori_code) in all_indices_tup_set and num < max_num:
                        ori_code[level] = sort_distances_index[item][level][num]
                        num += 1

                    if tuple(ori_code) not in all_indices_tup_set:
                        break

                all_indices[item] = ori_code
                all_indices_tup[item] = tuple(ori_code)
                all_indices_tup_set.add(tuple(ori_code))

        tt += 1

    print("All indices number: ", len(all_indices))
    print(len(set([str(code) for code in all_indices])))
    print("Max number of conflicts: ", max(get_indices_count(all_indices_tup).values()))

    tot_item = len(all_indices_tup)
    tot_indice = len(set(all_indices_tup))
    print("Collision Rate", (tot_item - tot_indice) / tot_item)

    codes_holder = np.zeros((embs_block.shape[0], indices.shape[1]), dtype=np.int32)
    for i, code in enumerate(all_indices):
        codes_holder[i] = code

    series = []
    for i in range(codes_holder.shape[1]):
        series.append(pd.Series(codes_holder[:, i], name=f"L{i}"))
    df = pd.concat(series, axis=1)

    df["product_id"] = embs_df["product_id"].values

    new_quant_method = quant_method + "-col"
    quantized_path = config.paths.semantic_ids_tplt.format(
        emb_method=config.data.emb_method,
        quant_method=new_quant_method,
        category=config.data.category,
    )
    df.to_parquet(quantized_path, filesystem=fs)
    print("Quantized data saved to", quantized_path)


if __name__ == "__main__":
    main()
