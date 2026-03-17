"""
2D Gaussian mixture model for sampler benchmarking, specified with NumPyro.

The target distribution is a mixture of K isotropic Gaussians in 2D:
    p(x) = sum_k w_k * N(x | mu_k, sigma_k^2 * I)

A NumPyro model is used to define the joint, and numpyro.infer.util.log_density
is used to extract a BlackJAX-compatible log-density function.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer.util import log_density

OUTPUT_DIR = Path(__file__).parent.parent / "output"


# Default mixture: 3 well-separated modes of equal weight
DEFAULT_MEANS = jnp.array([
    [-5.0,  0.0],
    [ 5.0,  0.0],
    [ 0.0,  5.0],
])
DEFAULT_SCALES = jnp.array([1.0, 1.0, 1.0])
DEFAULT_WEIGHTS = jnp.array([1/3, 1/3, 1/3])


def gaussian_mixture(means=DEFAULT_MEANS, scales=DEFAULT_SCALES, weights=DEFAULT_WEIGHTS):
    """
    NumPyro model for a 2D Gaussian mixture.

    Samples a 2D position x from:
        k   ~ Categorical(weights)
        x   ~ MultivariateNormal(means[k], scales[k]^2 * I)
    """
    mixing = dist.Categorical(probs=weights)
    cov_matrices = jax.vmap(lambda s: s ** 2 * jnp.eye(2))(scales)  # (K, 2, 2)
    components = dist.MultivariateNormal(means, cov_matrices)
    mixture = dist.MixtureSameFamily(mixing, components)
    numpyro.sample("x", mixture)


def make_log_density(means=DEFAULT_MEANS, scales=DEFAULT_SCALES, weights=DEFAULT_WEIGHTS):
    """
    Returns a BlackJAX-compatible log-density function derived from the NumPyro model.

    Args:
        means:   (K, 2) array of component means
        scales:  (K,)   array of isotropic standard deviations per component
        weights: (K,)   array of mixture weights (must sum to 1)

    Returns:
        log_density_fn(x): scalar log p(x) for a 2D position vector x,
                           suitable for use as blackjax logdensity_fn
    """
    def log_density_fn(x):
        ld, _ = log_density(
            gaussian_mixture,
            model_args=(),
            model_kwargs={"means": means, "scales": scales, "weights": weights},
            params={"x": x},
        )
        return ld

    return log_density_fn


def plot_model(
    means=DEFAULT_MEANS,
    scales=DEFAULT_SCALES,
    weights=DEFAULT_WEIGHTS,
    grid_range=(-8, 8),
    resolution=200,
    filename="gaussian_mixture.png",
):
    """
    Saves a 3D surface plot of the 2D Gaussian mixture density to OUTPUT_DIR.

    Args:
        grid_range:  (min, max) extent of both axes
        resolution:  number of grid points per axis
        filename:    output filename inside gaussian_mix/output/
    """
    log_density_fn = make_log_density(means, scales, weights)
    vmap_log_density = jax.vmap(log_density_fn)

    lo, hi = grid_range
    xs = np.linspace(lo, hi, resolution)
    ys = np.linspace(lo, hi, resolution)
    XX, YY = np.meshgrid(xs, ys)
    grid = jnp.array(np.stack([XX.ravel(), YY.ravel()], axis=-1))  # (N, 2)

    log_p = np.array(vmap_log_density(grid)).reshape(resolution, resolution)
    Z = np.exp(log_p)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(XX, YY, Z, cmap="viridis", linewidth=0, antialiased=True)
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.set_zlabel("p(x)")
    ax.set_title("2D Gaussian Mixture")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / filename
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {out_path}")
