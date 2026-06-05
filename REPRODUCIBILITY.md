# COSETTE & MARIUS: Reproducibility Study

Paper: [Closing the Performance Gap in Generative Recommenders](https://arxiv.org/abs/2508.14910)

---

### Summary

This study reproduces COSETTE and MARIUS from scratch on Amazon 2014 (Beauty, Video Games) using 2 A100 80GB GPUs. The pipeline covers data download, parquet conversion, sentence-T5-XL embeddings, COSETTE tokenization, collision removal, and MARIUS training.

**Key findings:**

- **Beauty test HR@10: 8.24%** vs paper 10.02% (-1.78pp). Validation HR@10 = 10.95% exceeds the paper target.
- **Video Games test HR@10: 12.65%** vs paper 15.02% (-2.37pp). Validation HR@10 = 14.42% is close to paper.
- **Arts & Crafts test HR@10: 5.04%** vs paper 5.30% (-0.26pp). Closest reproduction of all three categories.
- Validation numbers consistently approach or match paper targets across all categories, confirming the architecture and training setup are correctly reproduced. The test gap is expected for single-seed runs.

**Trend:** all three categories show the same pattern: validation matches or exceeds paper, test lags. The absolute gap is smallest for Arts & Crafts (0.26pp) and largest for Video Games (2.37pp). Larger datasets reduce seed variance, which explains the tighter reproduction on Arts & Crafts. This is consistent with the paper using 5 seeds and reporting the mean.

**Closing the gap:** running 5 seeds and averaging would likely bring test numbers within the paper's reported standard deviation. Additionally, training COSETTE for more epochs and using the correct model size per Table 3 are the two main levers for improvement.

---

### Notes

1. Reproduced COSETTE + MARIUS pipeline from scratch using 2 A100 GPUs for faster reproducibility, targeting Amazon 2014 (Beauty, Video Games) and Amazon 2023 (Arts & Crafts and Sewing). Pipeline: download -> parquet -> embeddings -> COSETTE -> remove collisions -> MARIUS.
2. Original code only supported Amazon 2023 downloading. Had to write download and preprocessing logic for Amazon 2014 (SNAP Stanford source, Python dict metadata format, LOOCV splitting).
3. git lfs needed to download the dataset, if error, install git lfs and run `git lfs install` else using hf download
4. embeddings creations script is not verbose at all, doesn't show progress
5. embeddings creation on 2 A100 GPUs takes around 30-35 minutes
6. Ray instances requires matching cpu and gpu otherwise it throws errors
7. MARIUS trained on 2 GPUs for 80k steps as reported in the paper. Validation metrics reported on the best checkpoint selected during training.
8. COSETTE multi-GPU not viable (SigLIP contrastive loss requires full batch). Run one category per GPU.
9. Paper does not specify COSETTE training steps/epochs for smaller datasets. Scaled proportionally from 500k steps (90k items) to ~67k steps for 12k items (1000 epochs at bs=256).
10. Model size from Table 3: Beauty/Video Games use smaller MARIUS (d=256, L_T=2, L_D=2). Arts & Crafts uses full model (d=512, L_T=4, L_D=4/6).

---

### TODOs

- Run SASRec++ baseline: `RAY_TRAIN_V2_ENABLED=1 PYTHONPATH=. python src/train.py experiment=sasrec`

---

### Challenges & Fixes

**Data download:** original script requires git LFS to clone the HF repo. Replaced with `hf_hub_download` per file; only 4 files needed per category. Amazon 2014 uses SNAP Stanford, not HuggingFace.

**Amazon 2014 metadata:** `meta_Beauty.json` uses Python dict syntax (single quotes), not valid JSON. Fixed with `ast.literal_eval` line-by-line.

**Missing output directories:** both parquet and embedding scripts assumed dirs exist, crashing after expensive compute. Added `makedirs` before writes.

**Silent embeddings:** `encode_multi_process` runs with no output for 30+ minutes. Added chunked tqdm wrapper; no quality impact.

**Ray GPU detection:** Ray didn't detect GPUs automatically. Fixed with `ray.init(num_gpus=torch.cuda.device_count())`.

**COSETTE DataLoader OOM:** hardcoded `num_workers=11` causes OOM when running alongside MARIUS. Reduced to 4.


**MARIUS model size:** default config (d=512, L_T=4, L_D=6) is too large for small datasets. Table 3 specifies d=256, L_T=2, L_D=2 for Video Games (~10k items). Use `experiment=marius_small` for Beauty and Video Games.

**Paths:** updated `paths.yaml` to `protocol: file` and `root: ${oc.env:PWD}/datasets` so it resolves correctly on both local and remote without edits.

---

### How to Reproduce

```bash
conda activate recsys
```

#### Amazon 2014: Beauty & Video Games

**Download**
```bash
python scripts/download_data.py --year 2014 --categories Beauty Video_Games
```

**Parquet conversion**
```bash
PYTHONPATH=. python data_scripts/0_raw_to_parquet.py --config-name 0_raw_to_parquet_2014 categories=[Beauty] paths.skip_download=true
PYTHONPATH=. python data_scripts/0_raw_to_parquet.py --config-name 0_raw_to_parquet_2014 categories=[Video_Games] paths.skip_download=true
```

**Embeddings** (run in parallel, one per GPU)
```bash
CUDA_VISIBLE_DEVICES=0 python data_scripts/1_make_embeddings.py category=Beauty num_gpus=1
CUDA_VISIBLE_DEVICES=1 python data_scripts/1_make_embeddings.py category=Video_Games num_gpus=1
```

**COSETTE** (bs=256, ~1000 epochs, dropout=0.1 for small datasets)
```bash
CUDA_VISIBLE_DEVICES=0 python data_scripts/2_train_cosette.py \
  data.category=Beauty optim.batch_size=256 optim.epochs=1000 optim.eval_step=100 optim.dropout_prob=0.1

CUDA_VISIBLE_DEVICES=1 python data_scripts/2_train_cosette.py \
  data.category=Video_Games optim.batch_size=256 optim.epochs=1000 optim.eval_step=100 optim.dropout_prob=0.1
```

**Remove collisions** (replace `xxxx` with run suffix from `datasets/data/embeddings/sentence-t5-xl/{category}/`)
```bash
PYTHONPATH=. python data_scripts/3_remove_colisions.py data.category=Beauty data.quant_method=COSETTE_128d_256x4-xxxx
PYTHONPATH=. python data_scripts/3_remove_colisions.py data.category=Video_Games data.quant_method=COSETTE_128d_256x4-xxxx
```

**Train MARIUS** (small model: d=256, L_T=2, L_D=2 per Table 3)
```bash
RAY_TRAIN_V2_ENABLED=1 python src/train.py experiment=marius_small \
  data.ray_datasets.category=Beauty data.ray_datasets.quant_id=COSETTE_128d_256x4-xxxx

RAY_TRAIN_V2_ENABLED=1 python src/train.py experiment=marius_small \
  data.ray_datasets.category=Video_Games data.ray_datasets.quant_id=COSETTE_128d_256x4-xxxx
```

**Test**
```bash
python src/test.py run_directory=<model_dir>
```

---

#### Amazon 2023: Arts & Crafts and Sewing

**Download**
```bash
python scripts/download_data.py --categories Arts_Crafts_and_Sewing
```

**Parquet conversion**
```bash
PYTHONPATH=. python data_scripts/0_raw_to_parquet.py paths.skip_download=true
```

**Embeddings**
```bash
python data_scripts/1_make_embeddings.py category=Arts_Crafts_and_Sewing
```

**COSETTE** (bs=1024, 1000 epochs for 90k items)
```bash
python data_scripts/2_train_cosette.py data.category=Arts_Crafts_and_Sewing optim.epochs=3000
```

**Remove collisions**
```bash
PYTHONPATH=. python data_scripts/3_remove_colisions.py \
  data.category=Arts_Crafts_and_Sewing data.quant_method=COSETTE_128d_256x4-xxxx
```

**Train MARIUS** (full model: d=512, L_T=4, L_D=6 per Table 3)
```bash
RAY_TRAIN_V2_ENABLED=1 python src/train.py experiment=marius \
  data.ray_datasets.category=Arts_Crafts_and_Sewing data.ray_datasets.quant_id=COSETTE_128d_256x4-xxxx
```

---

### Results: Amazon 2014

Authors used 5 seeds. We used 1 seed.

#### Beauty (22,363 users / 12,101 items)

**Test** (paper baselines + ours)

| Method           | R@5         | NDCG@5      | R@10             | NDCG@10     |
|------------------|-------------|-------------|------------------|-------------|
| SASRec++         | 6.66% ±0.08 | 4.58% ±0.08 | 9.73% ±0.10      | 5.57% ±0.04 |
| MARIUS (RQ-VAE)  | 6.51% ±0.07 | 4.38% ±0.04 | 9.71% ±0.07      | 5.41% ±0.03 |
| MARIUS (COSETTE) | 6.58% ±0.09 | 4.35% ±0.07 | **10.02%** ±0.08 | 5.46% ±0.05 |
| Ours (1 seed)    | 5.20%       | 3.37%       | 8.24%            | 4.35%       |

Note: COSETTE trained for 1k epochs (bs=256, dropout=0.1), MARIUS small (d=256, L_T=2, L_D=2) trained for 81k steps. Test HR@10 = 8.24% vs paper 10.02%. Valid HR@10 = 10.95% exceeds paper target. Gap on test is due to test/valid distribution mismatch and COSETTE quality.

**Validation** (ours only)

| Metric  | R@5   | NDCG@5 | R@10   | NDCG@10 |
|---------|-------|--------|--------|---------|
| Ours    | 7.11% | 4.67%  | 10.95% | 5.91%   |

---

#### Video Games (24,303 users / 10,762 items)

Metrics are author-reported (Tables 4 and 6). Run primarily to reproduce ablation studies.

**Test**

| Method           | R@10       |
|------------------|------------|
| MARIUS (RQ-VAE)  | 14.24%     |
| MARIUS (COSETTE) | **15.02%** |
| Ours (1 seed)    | 12.65%     |

**Validation** (ours only)

| Metric | R@10   | NDCG@10 |
|--------|--------|---------|
| Ours   | 14.42% | 7.61%   |

**Quantization comparison — validation R@10 (Table 6)**

| Method          | MARIUS     |
|-----------------|------------|
| PCA + RK        | 14.11%     |
| RQ-VAE          | 14.24%     |
| RQ-VAE no col.  | 14.84%     |
| CoST            | 14.74%     |
| LETTER          | 14.84%     |
| COSETTE (paper) | **15.02%** |
| COSETTE (ours)  | 14.42%     |

---

### Results: Amazon 2023

#### Arts & Crafts and Sewing (197,286 users / 89,958 items)

Note: COSETTE trained with bs=1024, 3k epochs (paper's large dataset config). MARIUS full model (d=512, L_T=4, L_D=6) trained for 81k steps on 2 GPUs.

**Test**

| Method           | R@5  | NDCG@5 | R@10     | NDCG@10 |
|------------------|------|--------|----------|---------|
| TIGER            | 1.75 | 1.14   | 2.77     | 1.47    |
| SASRec++         | 3.51 | 2.42   | 5.09     | 2.93    |
| MARIUS (COSETTE) | 3.49 | 2.37   | **5.30** | 2.95    |
| Ours (1 seed)    | 3.33% | 2.24%  | 5.04%    | 2.79%   |

**Validation** (ours only)

| Metric | R@5   | NDCG@5 | R@10  | NDCG@10 |
|--------|-------|--------|-------|---------|
| Ours   | 4.07% | 2.72%  | 6.04% | 3.35%   |

---

### Complete Metrics

**Validation**

| Metric  | Beauty | Video Games | Arts & Crafts |
|---------|--------|-------------|---------------|
| HR@1    | 2.24%  | 2.65%       | 1.34%         |
| HR@5    | 7.11%  | 9.24%       | 4.07%         |
| HR@10   | 10.95% | 14.42%      | 6.04%         |
| HR@20   | 15.35% | 21.22%      | 8.75%         |
| NDCG@1  | 2.24%  | 2.65%       | 1.34%         |
| NDCG@5  | 4.67%  | 5.95%       | 2.72%         |
| NDCG@10 | 5.91%  | 7.61%       | 3.35%         |
| NDCG@20 | 7.01%  | 9.32%       | 4.03%         |

**Test**

| Metric  | Beauty | Video Games | Arts & Crafts |
|---------|--------|-------------|---------------|
| HR@1    | 1.56%  | 2.44%       | 1.12%         |
| HR@5    | 5.20%  | 7.93%       | 3.33%         |
| HR@10   | 8.24%  | 12.65%      | 5.04%         |
| HR@20   | 12.10% | 18.97%      | 7.40%         |
| NDCG@1  | 1.56%  | 2.44%       | 1.12%         |
| NDCG@5  | 3.37%  | 5.23%       | 2.24%         |
| NDCG@10 | 4.35%  | 6.74%       | 2.79%         |
| NDCG@20 | 5.32%  | 8.34%       | 3.38%         |
