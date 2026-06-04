"""
Measures of spread in high-dimensional embedding spaces.

This module implements the full suite of measures evaluated in:

    Marbut, McKinney-Bock & Wheeler (2023). "Reliable Measures of Spread in
    High Dimensional Latent Spaces." ICML 2023.

The file is organized in two sections:
  1. Examined measures (ACS, I(V)): widely-used baselines shown to be unreliable.
  2. Alternative measures (EEE/AUC_eigensum, VRM/vasicek_entropy, and others)
     evaluated as more reliable estimators of data spread.

The aggregate function `isotropy_measures()` computes all measures for a given
embedding matrix.
"""

from sklearn.decomposition import PCA
from sklearn.metrics import auc
from scipy.stats import entropy, differential_entropy, multivariate_normal
from scipy.special import gamma
from scipy.spatial import KDTree
import numpy as np
from math import e
import argparse
import matplotlib.pyplot as plt
import faiss
from sympy import EulerGamma


# ---------------------------------------------------------------------------
# Examined measures (prior work)
# These are the measures evaluated and found to be unreliable in the paper.
# ---------------------------------------------------------------------------

def avg_cos(embeddings, num_sample=None, seed=None):
    """Average Cosine Similarity (ACS) — Arora et al. (2016); Mu & Viswanath (2018).

    Computes the mean pairwise cosine similarity across the embedding space.
    Higher values indicate less isotropic (more anisotropic) representations.

    Parameters
    ----------
    embeddings : array-like, shape (N, d)
    num_sample : int, optional
        If provided, compute over a random subsample of this size.
    seed : int, optional
        Random seed for subsampling.

    Returns
    -------
    float
        Mean pairwise cosine similarity, rounded to 4 decimal places.
    """
    embeddings = np.array(embeddings)
    if num_sample is not None:
        if seed is not None:
            np.random.seed(seed)
        idx = np.random.choice(len(embeddings), min(num_sample, len(embeddings)), replace=False)
        embeddings = embeddings[idx]
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-16)
    cos_sim = normalized @ normalized.T
    n = len(embeddings)
    return round(float((np.sum(cos_sim) - n) / (n * (n - 1))), 4)


def Iw(embeddings, n_samples=1000, seed=None):
    """I(V) partition function isotropy measure — Mu & Viswanath (2018).

    Computes the ratio of minimum to maximum partition function value Z(c, V)
    over uniformly sampled unit vectors c. A ratio close to 1 indicates
    isotropy; a ratio near 0 indicates extreme anisotropy.

    Parameters
    ----------
    embeddings : array-like, shape (N, d)
    n_samples : int
        Number of unit vectors c to sample. Default 1000.
    seed : int, optional

    Returns
    -------
    float
        min Z(c) / max Z(c), rounded to 4 decimal places.
    """
    embeddings = np.array(embeddings)
    if seed is not None:
        np.random.seed(seed)
    d = embeddings.shape[1]
    c_samples = np.random.normal(0, 1, (n_samples, d))
    c_samples /= np.linalg.norm(c_samples, axis=1, keepdims=True) + 1e-16
    Z = np.exp(embeddings @ c_samples.T).mean(axis=0)
    return round(float(np.min(Z) / (np.max(Z) + 1e-16)), 4)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def normal_compare(dimension, size):
    """Sample a standard normal reference distribution of matching shape.

    Parameters
    ----------
    dimension : int
    size : int

    Returns
    -------
    tuple : (distribution, sample, total_explained_variance)
    """
    x = multivariate_normal([0] * dimension, np.identity(dimension))
    y = x.rvs(size=size)
    pc = PCA()
    v = sum(pc.fit(y).explained_variance_)
    return x, y, v


# ---------------------------------------------------------------------------
# Alternative measures
# ---------------------------------------------------------------------------

def discrete_entropy_ratio(embeddings, k):
    """Mean normalized entropy across PCA-projected dimensions, binned into k bins.

    Parameters
    ----------
    embeddings : array-like, shape (N, d)
    k : int
        Number of histogram bins per dimension.

    Returns
    -------
    list of float
        Normalized entropy value per dimension (0 = maximally concentrated,
        1 = maximally uniform).
    """
    pc = PCA()
    px_embeddings = pc.fit_transform(embeddings)
    min_v = np.min(px_embeddings)
    max_v = np.max(px_embeddings)

    dims = px_embeddings.T
    dim_freq = []
    for dim in dims:
        hist, _ = np.histogram(dim, bins=k, range=(min_v, max_v), density=False)
        dim_freq.append(hist / len(px_embeddings))

    max_ent = entropy([1 / k] * k)
    return [entropy(dim) / max_ent for dim in dim_freq]


def PC_ratio(embeddings):
    """Ratio of smallest to largest PCA eigenvalue.

    A ratio near 1 indicates uniform variance across dimensions (isotropic);
    a ratio near 0 indicates extreme concentration of variance in few components.

    Parameters
    ----------
    embeddings : array-like, shape (N, d)

    Returns
    -------
    float
    """
    pc = PCA()
    pc.fit(embeddings)
    r = pc.explained_variance_[-1] / pc.explained_variance_[0]
    return round(r, 4)


def AUC_eigensum(embeddings, plot=False):
    """Eigenvalue Early Enrichment (EEE) — normalized AUC of cumulative eigenvalue sum.

    Measures how quickly variance accumulates across PCA components relative
    to a uniform baseline. A value near 0 indicates early enrichment (variance
    concentrated in few components); a value near 1 indicates uniform spread.

    This is the primary alternative measure introduced in the paper.

    Parameters
    ----------
    embeddings : array-like, shape (N, d)
    plot : bool
        If True, plot the cumulative eigenvalue sum vs. uniform baseline.

    Returns
    -------
    float
        Normalized AUC in [0, 1].
    """
    pc = PCA()
    pc.fit(embeddings)
    num_pc = pc.n_components_
    eigensum = np.cumsum(pc.explained_variance_)
    ref = np.cumsum([eigensum[-1] / num_pc] * num_pc)
    AUC_sum = eigensum - ref
    AUC_val = auc(range(num_pc), AUC_sum)
    total_poss = (eigensum[-1] * num_pc) / 2

    if plot:
        plt.plot(range(num_pc), eigensum, label='Cumulative Sum of Eigenvalues')
        plt.plot(range(num_pc), ref)
        plt.legend()
        plt.show()

    return round(AUC_val / total_poss, 4)


def vasicek_entropy(embeddings):
    """Vasicek Ratio MSE (VRM) — MSE of per-component Vasicek entropy ratios from 1.

    Estimates differential entropy per PCA component using the Vasicek estimator,
    then computes the mean squared deviation of the ratio (observed/normal) from 1.
    A value near 0 indicates the embedding space matches a standard normal in entropy;
    higher values indicate deviation.

    This is the second primary alternative measure introduced in the paper.

    Parameters
    ----------
    embeddings : array-like, shape (N, d)

    Returns
    -------
    float
    """
    N = len(embeddings)
    d = len(embeddings[0])
    pc = PCA()
    tx = pc.fit_transform(embeddings)

    comp, sample, ev = normal_compare(d, N)
    scale = ev / sum(pc.explained_variance_)
    tx *= np.sqrt(scale)

    m = np.log(np.sqrt(2 * np.pi * e))  # theoretical max Vasicek entropy for std normal
    ent = differential_entropy(tx)
    r = np.exp(ent) / np.exp(m)

    return round(np.sum((1 - r) ** 2) / d, 4)


def koleo_entropy(embeddings, k, plot=False):
    """KoLeo entropy ratio relative to a standard normal reference.

    Uses the Beirlant k-nearest-neighbor entropy estimator, normalized by the
    expected entropy of a standard normal with matching variance.

    Parameters
    ----------
    embeddings : array-like, shape (N, d)
    k : int
        Number of nearest neighbors.
    plot : bool

    Returns
    -------
    float
        Ratio of observed KoLeo entropy to normal reference entropy.
    """
    N = len(embeddings)
    m = len(embeddings[0])

    tx = embeddings * np.sqrt(m / sum(np.var(embeddings, axis=0)))

    index = faiss.IndexFlatL2(m)
    index.add(np.array(tx).astype('float32'))
    D, I = index.search(np.array(tx).astype('float32'), k + 1)
    rho = np.sqrt(np.array([D[i][-1] for i in range(N)]))
    H_obs = (1 / N) * np.sum(np.log((N * rho) + 1e-16)) + np.log(2) + EulerGamma.evalf()

    comp, sample, ev = normal_compare(m, N)
    index_n = faiss.IndexFlatL2(m)
    index_n.add(np.array(sample).astype('float32'))
    D_n, I_n = index_n.search(np.array(sample).astype('float32'), k + 1)
    rho_n = np.sqrt(np.array([D_n[i][-1] for i in range(N)]))
    H_n = (1 / N) * np.sum(np.log(N * rho_n)) + np.log(2) + EulerGamma.evalf()

    if plot:
        plt.hist(rho, bins=30, label='Observed Nearest Neighbors', histtype='step')
        plt.hist(rho_n, bins=30, label='Normal Nearest Neighbors', histtype='step')
        plt.legend()
        plt.show()

    return round(H_obs / H_n, 4)


def knn_overlap(embeddings, plot=False):
    """K-nearest-neighbor overlap between first- and k-th-neighbor distance distributions.

    Measures the overlap between the distribution of 1-NN and k-NN distances,
    where k = min(5d, 100). Low overlap indicates strong local clustering.

    Parameters
    ----------
    embeddings : array-like, shape (N, d)
    plot : bool

    Returns
    -------
    float
        Fraction of k-NN distribution mass overlapping with the 1-NN distribution.
    """
    N = len(embeddings)
    m = len(embeddings[0])
    k = min(m * 5, 100)

    index = faiss.IndexFlatL2(m)
    index.add(np.array(embeddings).astype('float32'))
    D, I = index.search(np.array(embeddings).astype('float32'), k + 1)

    rho_1 = np.sqrt(np.array([D[i][1] for i in range(N)]))
    rho_k = np.sqrt(np.array([D[i][-1] for i in range(N)]))

    rho_min = np.min([np.min(rho_1), np.min(rho_k)])
    rho_max = np.max([np.max(rho_1), np.max(rho_k)])

    if plot:
        plt.hist(rho_1, bins=30, range=(rho_min, rho_max), label='First Nearest Neighbors', histtype='step')
        plt.hist(rho_k, bins=30, range=(rho_min, rho_max), label=f'{k}th Nearest Neighbors', histtype='step')
        plt.legend()
        plt.show()

    rho_1_bin = np.histogram(rho_1, bins=30, range=(rho_min, rho_max))
    rho_k_bin = np.histogram(rho_k, bins=30, range=(rho_min, rho_max))
    rho_difference = rho_k_bin[0] - rho_1_bin[0]
    rho_ovl = rho_k_bin[0] - [i if i > 0 else 0 for i in rho_difference]

    return round(np.sum(rho_ovl) / np.sum(rho_k_bin[0]), 4)


def KL_divergence(embeddings):
    """KL divergence from the PCA-projected embedding distribution to a standard normal.

    Uses the closed-form KL divergence between two Gaussians, with the observed
    distribution normalized to match the total variance of a standard normal.

    Parameters
    ----------
    embeddings : array-like, shape (N, d)

    Returns
    -------
    float
    """
    N = len(embeddings)
    d = len(embeddings[0])
    pc = PCA()
    tx = pc.fit_transform(embeddings)

    comp, sample, ev = normal_compare(d, N)
    scale = ev / sum(pc.explained_variance_)
    tx *= np.sqrt(scale)
    m_obs = np.mean(tx, axis=0)
    s_obs = np.cov(tx.T)

    kl = 0.5 * ((m_obs @ m_obs) + np.trace(s_obs) - d - np.log(np.linalg.det(s_obs) + 1e-16))
    return round(kl, 4)


def kl_divergence_discrete(embeddings):
    """Mean squared per-dimension KL divergence using discrete histogram binning.

    Parameters
    ----------
    embeddings : array-like, shape (N, d)

    Returns
    -------
    float
    """
    N = len(embeddings)
    d = len(embeddings[0])
    pc = PCA()
    tx = pc.fit_transform(embeddings)

    comp, sample, ev = normal_compare(d, N)
    scale = ev / sum(pc.explained_variance_)
    tx *= np.sqrt(scale)

    norm_min = np.min(sample)
    norm_max = np.max(sample)

    norm_bin = np.array([np.histogram(sample.T[i], bins=30, range=(norm_min, norm_max))[0] + 1e-16 for i in range(d)])
    norm_p = norm_bin / np.array([np.sum(norm_bin, axis=1)]).T

    tx_bin = np.array([np.histogram(tx.T[i], bins=30, range=(norm_min, norm_max))[0] + 1e-16 for i in range(d)])
    tx_p = tx_bin / np.array([np.sum(tx_bin, axis=1)]).T

    kl = np.array([entropy(tx_p[i], norm_p[i]) for i in range(d)])
    return round(np.sum(kl ** 2) / d, 4)


def kl_divergence_empirical(embeddings, k):
    """Empirical KL divergence using k-nearest-neighbor density estimation (Wang et al. 2009).

    Parameters
    ----------
    embeddings : array-like, shape (N, d)
    k : int
        Number of nearest neighbors for density estimation.

    Returns
    -------
    float
    """
    N = len(embeddings)
    d = len(embeddings[0])
    pc = PCA()
    tx = pc.fit_transform(embeddings)

    comp, sample, ev = normal_compare(d, N)
    scale = ev / sum(pc.explained_variance_)
    tx *= np.sqrt(scale)
    tx = tx.copy(order='C')

    index = faiss.IndexFlatL2(d)
    index.add(np.array(tx).astype('float32'))
    D, I = index.search(np.array(tx).astype('float32'), k + 1)
    r_k = np.sqrt(np.array([D[i][-1] for i in range(N)]))

    comp, sample, ev = normal_compare(d, N)
    index_n = faiss.IndexFlatL2(d)
    index_n.add(np.array(sample).astype('float32'))
    D_n, I_n = index_n.search(np.array(tx).astype('float32'), k)
    s_k = np.sqrt(np.array([D_n[i][-1] for i in range(N)]))

    return round((d / N) * np.sum(np.max([[0] * N, np.log((s_k + 1e-16) / (r_k + 1e-16))], axis=0)), 4)


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def isotropy_measures(embeddings, cos_samples, seed=11):
    """Compute all measures for a given embedding matrix.

    Parameters
    ----------
    embeddings : array-like, shape (N, d)
    cos_samples : int
        Number of vectors to subsample for ACS computation.
    seed : int
        Random seed. Default 11.

    Returns
    -------
    tuple : (cos, iw, pcr, auc, vas, koleo, ovl, kl, kl_disc, kl_emp)
        One value per measure, in the order: ACS, I(V), PC_ratio, EEE,
        VRM, KoLeo, KNN_overlap, KL_gaussian, KL_discrete, KL_empirical.
    """
    np.random.seed(seed)
    cos = avg_cos(embeddings, num_sample=cos_samples)
    iw = Iw(embeddings)
    pcr = PC_ratio(embeddings)
    auc_val = AUC_eigensum(embeddings)
    vas = vasicek_entropy(embeddings)
    koleo = koleo_entropy(embeddings, 1)
    ovl = knn_overlap(embeddings)
    kl = KL_divergence(embeddings)
    kl_disc = kl_divergence_discrete(embeddings)
    kl_emp = kl_divergence_empirical(embeddings, 1)

    return cos, iw, pcr, auc_val, vas, koleo, ovl, kl, kl_disc, kl_emp
