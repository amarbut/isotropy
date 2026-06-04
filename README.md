# Reliable Measures of Spread in High Dimensional Latent Spaces

Code for the ICML 2023 paper:

> **Reliable Measures of Spread in High Dimensional Latent Spaces**  
> Anna C. Marbut, Katy McKinney-Bock, Travis J. Wheeler  
> *Proceedings of the 40th International Conference on Machine Learning (ICML 2023)*  
> [PDF](https://amarbut.github.io/files/ICML_Reliable_Measures_of_Data_Spread.pdf)

## Overview

Measuring the spread of data in high-dimensional embedding spaces is fundamental to evaluating and comparing text representations, yet existing isotropy measures haven't been rigorously evaluated as estimators. This paper:

1. Identifies failure modes in two widely-used isotropy measures: Average Cosine Similarity (ACS) and I(V)
2. Introduces **EEE (Eigenvalue Early Enrichment)** — captures the concentration of variance in leading principal components
3. Introduces **VRM (Vasicek Ratio MSE)** — a nonparametric measure derived from order statistics

We show that EEE and VRM are more reliable estimators of actual data spread across a range of synthetic and real embedding spaces.

## Repository Structure

```
alternate_isotropy_measures.py   # Core implementations of all measures (ACS, I(V), EEE, VRM, others)
latent_space_models.py           # Generates synthetic data distributions
word2vec_measures.py             # Applies measures to word2vec embeddings
isotropy_explore.py              # Interactive analysis scripts (designed for Spyder / VS Code interactive)
isotropy_results.py              # Results aggregation and visualization scripts
```


## Key Dependencies

```
numpy
scipy
scikit-learn
faiss
matplotlib
seaborn
```

## Citation

```bibtex
@inproceedings{marbut2023reliable,
  title     = {Reliable Measures of Spread in High Dimensional Latent Spaces},
  author    = {Marbut, Anna C. and McKinney-Bock, Katy and Wheeler, Travis J.},
  booktitle = {Proceedings of the 40th International Conference on Machine Learning},
  year      = {2023}
}
```
