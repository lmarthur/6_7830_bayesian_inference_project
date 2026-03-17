"""
Unit tests for gaussian_mix/src/model.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import jax.numpy as jnp
import numpy as np
import pytest

from model import (
    DEFAULT_MEANS,
    DEFAULT_SCALES,
    DEFAULT_WEIGHTS,
    make_log_density,
)


def test_default_weights_sum_to_one():
    assert np.isclose(DEFAULT_WEIGHTS.sum(), 1.0)


def test_default_means_shape():
    assert DEFAULT_MEANS.shape == (3, 2)


def test_default_scales_positive():
    assert np.all(DEFAULT_SCALES > 0)


def test_log_density_returns_scalar():
    log_density_fn = make_log_density()
    x = jnp.array([0.0, 0.0])
    result = log_density_fn(x)
    assert result.shape == ()


def test_log_density_higher_at_modes():
    """Log density at each mode center should exceed density at the origin."""
    log_density_fn = make_log_density()
    log_p_origin = log_density_fn(jnp.array([0.0, 0.0]))
    for mean in DEFAULT_MEANS:
        log_p_mode = log_density_fn(mean)
        assert log_p_mode > log_p_origin, (
            f"Expected higher density at mode {mean} than at origin"
        )


def test_log_density_decreases_far_from_modes():
    """Log density should be very low far from all modes."""
    log_density_fn = make_log_density()
    log_p_far = log_density_fn(jnp.array([100.0, 100.0]))
    log_p_near = log_density_fn(DEFAULT_MEANS[0])
    assert log_p_near > log_p_far


def test_custom_weights():
    """A single-component mixture should concentrate mass at that mean."""
    means = jnp.array([[2.0, 2.0], [-2.0, -2.0]])
    scales = jnp.array([0.5, 0.5])
    weights = jnp.array([1.0, 0.0])
    log_density_fn = make_log_density(means=means, scales=scales, weights=weights)
    log_p_mode0 = log_density_fn(means[0])
    log_p_mode1 = log_density_fn(means[1])
    assert log_p_mode0 > log_p_mode1


def test_equal_density_at_symmetric_modes():
    """With equal weights, all mode centers should have the same log density."""
    log_density_fn = make_log_density()
    log_ps = [float(log_density_fn(mean)) for mean in DEFAULT_MEANS]
    assert np.allclose(log_ps, log_ps[0], atol=1e-5), (
        f"Expected equal density at all modes, got {log_ps}"
    )
