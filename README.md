<div align="center">

# Autoregressive Retrieval in the Ball
**A Reproduction and Hyperbolic Manifold Extension of MARIUS and COSETTE**

<strong>Nordin el Assassi</strong>
·
<strong>Elliott Callenbach</strong>
·
<strong>Steven Dong</strong>
·
<strong>Sakr Ismail</strong>
·
<strong>Simen Veenman</strong>

University of Amsterdam (UvA)
<br><br>

<a href="https://github.com/SmallNub/RecSys">
    <img alt="GitHub Repo" src="https://img.shields.io/badge/GitHub-Repository-blue.svg">
</a>
<a href="LICENSE">
    <img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg">
</a>
<br><br>
</div>

---

## Abstract

Generative sequential recommenders have recently emerged as a promising alternative to traditional ID-based models by eliminating massive embedding tables and complex approximate nearest neighbor searches. However, closing the accuracy and computational efficiency gap remains a critical challenge. This paper presents a reproducibility study of a framework consisting of two models designed to bridge this performance divide: COSETTE, which utilizes contrastive semantic tokenization, and MARIUS, which leverages an audio-inspired multi-scale attention architecture. We evaluate the scalability of the models using both small and large-scale Amazon datasets (2014 and 2023). In addition to standard ranking metrics, we analyze structural characteristics by measuring popularity bias and semantic variety. Our findings show that, while both models are reproducible on Amazon 2023, there were difficulties in reproducing results on Amazon 2014. Our evaluation shows that COSETTE and MARIUS mitigate popularity bias with a small trade-off in diversity. Finally, we introduce HyperCOS, an architectural extension that shifts residual vector quantization into a hyperbolic manifold to better capture hierarchical item spaces. Our empirical findings show that while HyperCOS successfully reduces semantic ID collision rates and maintains optimization stability, the resulting hyperbolic tokenization yields no statistically significant impact on downstream recommendation metrics. Code and configuration files are publicly available to facilitate future benchmarking.

## 1. Introduction & Methodology

This project aims to verify and reproduce the findings of Lepage et al. [1], who proposed COSETTE and MARIUS. Beyond reproducing the core accuracy claims, our work extends the original evaluation in several key directions:

- **Reproducibility**: Validation on both legacy Amazon 2014 and large-scale Amazon 2023 datasets.
- **Fairness and Popularity Bias**: Evaluation using non-ranking metrics like Gini Index and Intra-List Diversity (ILD).
- **HyperCOS**: An architectural extension replacing Euclidean operations in COSETTE's latent space with a hyperbolic manifold (Poincaré ball model) to better capture hierarchical item relations.
- **MARIUS Variations**: Architectural explorations including Feature Fusion, SwiGLU/RoPE/GELU integration, Listwise/Uniformity loss functions, and Knowledge Distillation.


## 2. Repository Structure

```text
.
├── configs/            # Hydra configuration files (model parameters, paths, etc.)
├── data_scripts/       # Data preprocessing: raw -> parquet, embedding generation, quantization
├── datasets/           # Root directory for raw and processed datasets
├── scripts/            # SLURM-compatible bash pipelines for automated end-to-end runs
├── src/                # Core PyTorch/Lightning source code for models (MARIUS, SASRec)
├── LICENSE             # Apache 2.0 License
└── README.md           
```

## 3. Usage

### 3.1. Hardware Requirements
All experiments in this project were conducted on the Snellius national supercomputer, utilizing NVIDIA H100 GPUs. Our provided bash scripts in `scripts/` are configured for SLURM workload managers with the corresponding `#SBATCH` directives. You can adapt these scripts to run locally or on other GPU clusters by modifying or removing the SLURM headers.

### 3.2. Environment Setup
Create a new environment with Python 3.9 and install dependencies:
```sh
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
```

### 3.3. Quick Start (Minimal Example)
If you want to bypass the automated SLURM pipelines and test a small configuration locally, you can use the direct Python commands. Each step relies on Hydra configurations found in `configs/`:

```sh
# 1. Download and process a small subset (e.g., Beauty category)
python data_scripts/0_raw_to_parquet.py --config-name 0_raw_to_parquet_2014 data.category=Beauty

# 2. Generate item embeddings
python data_scripts/1_make_embeddings.py category=Beauty num_gpus=1

# 3. Train the model manually
python src/train.py experiment=marius_small data.datasets.category=Beauty

# 4. Evaluate (Replace 'xxx' with the generated output folder name)
python src/test.py run_directory=xxx
```
> Evaluation metrics (Recall@K, NDCG@K, Gini, ILD) will be printed to `stdout` and saved within the respective run directory inside `outputs/`.

### 3.4. Full Data Processing Pipelines
To automate processing, we provide SLURM bash scripts.

```sh
# Process the 2014 dataset splits:
bash scripts/process_data_2014.sh

# Process the 2023 dataset splits:
bash scripts/process_data_2023.sh
```

### 3.5. Training and Evaluation Pipelines
For evaluation, our SLURM pipelines automatically handle training, inference, and evaluation across multiple seeds and configurations. Check the configs for the exact hyperparameters used:

```sh
# Standard MARIUS and COSETTE baseline pipeline
bash scripts/train_pipeline_cos_marius.sh

# Extended HyperCOS pipeline
bash scripts/train_pipeline_hypercos_marius.sh

# SASRec++ baseline pipeline
bash scripts/train_pipeline_sasrec.sh
```

## 4. Acknowledgements & References

This repository heavily relies on the foundational work by Lepage et al.:
> [1] Lepage, S., Mary, J., Picard, D.: *Closing the Performance Gap in Generative Recommenders with Collaborative Tokenization and Efficient Modeling*. arXiv preprint arXiv:2508.14910 (2025).

We gratefully acknowledge the use of the [Snellius](https://www.surf.nl/en/services/compute/snellius-the-national-supercomputer) national supercomputer, provided by SURF, for all experiments conducted in this work.

## 5. License
This project is licensed under the [Apache License 2.0](LICENSE).
