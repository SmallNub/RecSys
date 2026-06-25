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

Generative sequential recommenders have recently emerged as a promising alternative to traditional ID-based models by eliminating massive embedding tables and complex approximate nearest neighbor searches. However, closing the accuracy and computational efficiency gap remains a critical challenge. This repository presents a comprehensive reproducibility study of **COSETTE** and **MARIUS**, two frameworks designed to bridge this performance divide through contrastive semantic tokenization and audio-inspired multi-scale attention architectures. 

Moving beyond standard ranking metrics, we evaluate the scalability of these models on the large-scale Amazon 2023 dataset and assess structural characteristics using non-ranking metrics for popularity bias and semantic variety. Finally, we introduce **HyperCOS**, an architectural extension that shifts residual vector quantization into a hyperbolic manifold to better capture hierarchical item spaces. Our empirical findings show that while HyperCOS successfully reduces semantic ID collision rates and maintains optimization stability, the resulting hyperbolic tokenization yields no statistically significant impact on downstream recommendation metrics.

## 1. Introduction

This project aims to verify and reproduce the findings of Lepage et al. [1], who proposed COSETTE and MARIUS to close the accuracy and computational gap between generative and traditional ID-based recommenders. Beyond reproducing the core accuracy claims, our work extends the original evaluation in several key directions:
- **Fairness and Diversity**: We evaluate how MARIUS and COSETTE perform on non-ranking metrics such as popularity bias (Gini Index, Entropy) and Intra-List Diversity (ILD).
- **Scalability**: We rigorously test the framework on both the legacy Amazon 2014 and the large-scale Amazon 2023 datasets.
- **Hyperbolic Extension (HyperCOS)**: We modify the latent space of COSETTE to use hyperbolic geometry, which is theoretically better suited for capturing hierarchical relations in item data.

## 2. Methodology & Extensions

### 2.1. Reproducibility of COSETTE and MARIUS
We reimplemented the COSETTE (Collaborative and Semantic Tokenization) and MARIUS (Multi-scale Attention as Recommendation Index with fUSion) architectures, along with the SASRec++ baseline. While the results on the modern, large-scale Amazon 2023 dataset are highly reproducible, we found that backward compatibility with legacy 2014 configurations presented challenges due to undocumented parameter shifts in the original codebase.

### 2.2. Fairness and Popularity Bias
Generative recommenders are often prone to popularity bias. Our evaluation demonstrates that MARIUS effectively mitigates exposure inequality, yielding a lower Gini Index, at a marginal cost to intra-list diversity compared to SASRec++.

### 2.3. HyperCOS
Residual quantization naturally induces a hierarchical item space. To better capture this, **HyperCOS** replaces the Euclidean operations in COSETTE's latent space with operations in a hyperbolic manifold (the Poincaré ball model). This exponential spatial capacity helps drive distinct items to receive unique, collision-free semantic IDs. While HyperCOS successfully reduces the collision rate of COSETTE, our experiments indicate it has no statistically significant effect on the downstream performance of MARIUS.

### 2.4. MARIUS Architectural Explorations
In addition to the core reproductions, our repository includes several experimental variations of the MARIUS architecture (available under `src/models/` and executable via the `scripts/train_marius_*.sh` scripts):
- **Feature Fusion (`marius_fusion`)**: Injects the unquantized semantic ID as an additional input for MARIUS.
- **Advanced Architecture (`marius_advanced`)**: Integrates modern transformer components including RoPE, SwiGLU blocks, and GELU activations.
- **Advanced Loss Functions**: Implements listwise and uniformity losses to better align optimization with target ranking metrics.
- **Knowledge Distillation (`marius_student` & `marius_teacher`)**: Uses a student-teacher paradigm to help the high-capacity advanced models learn from the robust baseline representation space.

## 3. Usage

### 3.1. Environment

Create a new environment with Python 3.9 and install dependencies:

```sh
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
```

### 3.2. Data Processing

We provide automated SLURM-compatible bash scripts in the `scripts/` directory to download and process the datasets. These scripts wrap the core Python data generation modules found in `data_scripts/` (e.g., `0_raw_to_parquet.py`, `1_make_embeddings.py`).

```sh
# Process the 2014 dataset splits:
bash scripts/process_data_2014.sh

# Process the 2023 dataset splits:
bash scripts/process_data_2023.sh
```

### 3.3. Training and Evaluation Pipelines

For robust evaluation and reproducibility, we provide comprehensive end-to-end pipelines that handle training, quantization, and evaluation across multiple seeds and configurations.

- **SASRec++ Pipeline**:
  ```sh
  bash scripts/train_pipeline_sasrec.sh
  ```

- **COSETTE and MARIUS Pipeline**:
  ```sh
  bash scripts/train_pipeline_cos_marius.sh
  ```

- **HyperCOS Extension Pipeline**:
  ```sh
  bash scripts/train_pipeline_hypercos_marius.sh
  ```

Alternatively, you can run individual steps manually. Each Python script can be configured via its associated Hydra configuration file in the `configs/` directory:
```sh
# Manually train a specific model (e.g. marius, sasrec)
python src/train.py experiment=marius 

# Evaluate a specific model run
python src/test.py run_directory=xxx
```

## 4. Acknowledgements & References

This repository is built upon the original work by Lepage et al.:

> [1] Lepage, S., Mary, J., Picard, D.: *Closing the Performance Gap in Generative Recommenders with Collaborative Tokenization and Efficient Modeling*. arXiv preprint arXiv:2508.14910 (2025).

## 5. License

This project is licensed under the [Apache License 2.0](LICENSE) - see the [LICENSE](LICENSE) file for details.