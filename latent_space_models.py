"""
Synthetic latent space generators for evaluating isotropy measures.

Provides `cluster()` and `sphere()` for generating controlled synthetic
embedding spaces used in the simulation experiments of:

    Marbut, McKinney-Bock & Wheeler (2023). "Reliable Measures of Spread in
    High Dimensional Latent Spaces." ICML 2023.
"""

import numpy as np


def cluster(dim, shift=False, shuffle=False, dist=1.):
    """Generate synthetic data with symmetric clusters along each axis.

    For each dimension, places a pair of clusters on opposite sides of the
    origin. The resulting space has 2*dim clusters total.

    Parameters
    ----------
    dim : int
        Dimensionality of the embedding space. Produces 250*dim total points.
    shift : bool
        If False, clusters are exactly mirrored across the origin.
        If True, adds uniform noise to the mirrored cluster location.
    shuffle : bool
        If False, 125 points in each cluster of each pair.
        If True, distributes 250 points randomly between the two clusters.
    dist : float
        Upper bound for sampling cluster center distances from the origin.

    Returns
    -------
    np.ndarray, shape (250 * dim, dim)
    """
    mus = np.random.uniform(-dist, dist, size=(dim, dim))
    var = np.min((1 / dim, 0.2))
    data = []
    for i in range(dim):
        mu = mus[i]
        if shift:
            mu1 = mu + np.random.uniform(-dist, 0, size=dim)
            mu2 = (mu * -1) + np.random.uniform(0, dist, size=dim)
        else:
            mu1 = mu
            mu2 = mu * -1

        if shuffle:
            k = np.random.choice(250)
            clust1 = np.random.normal(mu1, var, size=(k, dim))
            clust2 = np.random.normal(mu2, var, size=(250 - k, dim))
        else:
            clust1 = np.random.normal(mu1, var, size=(125, dim))
            clust2 = np.random.normal(mu2, var, size=(125, dim))

        data.extend(clust1)
        data.extend(clust2)

    return np.array(data)


def sphere(dim, radius=1, fill=True, rings=1, cone=False):
    """Generate synthetic data uniformly distributed on (or within) a hypersphere.

    Uses the normal-vector normalization method, which produces a uniform
    distribution on the sphere surface.

    Parameters
    ----------
    dim : int
        Dimensionality. Produces 250*dim total points.
    radius : float
        Sphere radius.
    fill : bool
        If True, sample uniformly within the sphere volume.
        If False, sample only from the surface (or `rings` concentric shells).
    rings : int
        Number of concentric shells when fill=False.
    cone : bool
        If True, restrict samples to a cone with half-angle arctan(1/sqrt(dim))
        along the first axis.

    Returns
    -------
    np.ndarray, shape (250 * dim, dim)
    """
    if fill:
        r = np.array([np.random.uniform(0, radius, size=dim * 250) ** (1.0 / dim)])
    else:
        r_vals = np.array([])
        n = int((dim * 250) / rings)
        for i in range(rings):
            r_vals = np.concatenate((r_vals, [(i + 1) * (radius / rings)] * n))
        r = np.array([r_vals])

    if cone:
        w = 1 / np.sqrt(dim)
        r2 = r * np.tan(w)
        r3 = np.array([np.random.uniform(0, r2[0], size=(dim * 250)) ** (1.0 / (dim - 1))])
        u = np.random.normal(0, 1, (dim * 250, dim - 1))
        norm = np.array([np.linalg.norm(u, axis=1)])
        sp = r3.T * u / norm.T
        return np.concatenate((r.T, sp), axis=1)
    else:
        u = np.random.normal(0, 1, (dim * 250, dim))
        norm = np.array([np.linalg.norm(u, axis=1)])
        return r.T * u / norm.T
