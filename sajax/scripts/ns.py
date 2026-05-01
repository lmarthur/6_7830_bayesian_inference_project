"""
Run Nested Sampling (JAXNS) on the SAJAX planet+activity model and save outputs.
"""

import json
import sys
import time
import logging
import warnings
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    import arviz as az
import jaxns
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro.distributions as dist
import tensorflow_probability.substrates.jax as tfp

from model import (
    make_log_likelihood,
    plot_model,
    plot_bestfit_lightcurve,
    OUTPUT_DIR,
    PARAM_NAMES,
    GROUND_TRUTH,
    OBS_LIGHT_CURVE,
    PRIOR_DISTRIBUTIONS,
)

tfpd = tfp.distributions


def _numpyro_to_tfp(d):
    if isinstance(d, dist.Uniform):
        return tfpd.Uniform(low=d.low, high=d.high)
    elif isinstance(d, dist.LogNormal):
        return tfpd.LogNormal(loc=d.loc, scale=d.scale)
    elif isinstance(d, dist.Normal):
        return tfpd.Normal(loc=d.loc, scale=d.scale)
    elif isinstance(d, dist.Beta):
        return tfpd.Beta(
            concentration1=jnp.float64(d.concentration1),
            concentration0=jnp.float64(d.concentration0),
        )
    raise TypeError(f"No TFP equivalent known for {type(d)}")

NS_OUTPUT_DIR = OUTPUT_DIR / "ns"

MAX_SAMPLES = 1e5
NUM_POSTERIOR_DRAWS = 5000
NUM_LIVE_POINTS = 100


def main(seed=0, save_outputs=True):
    rng_key = jax.random.PRNGKey(seed)
    resample_key, run_key = jax.random.split(rng_key)
    _print = print if save_outputs else lambda *a, **kw: None

    log_likelihood_fn = make_log_likelihood(OBS_LIGHT_CURVE)
    if save_outputs:
        plot_model(filename="sajax_ground_truth.png")

    # --- Define JAXNS prior and likelihood ---
    def prior_model():
        samples = {}
        for name, d in PRIOR_DISTRIBUTIONS.items():
            samples[name] = yield jaxns.Prior(_numpyro_to_tfp(d), name=name)
        return samples

    model = jaxns.Model(prior_model=prior_model, log_likelihood=log_likelihood_fn)
    if save_outputs:
        model.sanity_check(jax.random.PRNGKey(1), S=100)

    # --- Run nested sampler ---
    ns = jaxns.NestedSampler(model=model, max_samples=MAX_SAMPLES, num_live_points=NUM_LIVE_POINTS, verbose=True)

    _print("Running nested sampling...")
    t0 = time.perf_counter()
    termination_reason, state = jax.jit(ns)(run_key)
    results = ns.to_results(termination_reason=termination_reason, state=state)
    wall_time_s = time.perf_counter() - t0

    _print(f"\nTermination reason: {termination_reason}")

    # --- Resample to uniform posterior draws ---
    uniform_samples = jaxns.resample(
        key=resample_key,
        samples=results.samples,
        log_weights=results.log_dp_mean,
        S=NUM_POSTERIOR_DRAWS,
        replace=True,
    )

    # --- Diagnostics ---
    jaxns_ess = float(results.ESS)
    total_likelihood_evals = int(results.total_num_likelihood_evaluations)
    ess_per_likelihood_eval = jaxns_ess / total_likelihood_evals
    log_z = float(results.log_Z_mean)
    log_z_uncert = float(results.log_Z_uncert)

    # uniform_samples keys match PRIOR_DISTRIBUTIONS (and thus PARAM_NAMES)
    posterior_dict = {name: np.array(uniform_samples[name])[None, :] for name in PARAM_NAMES}
    _az_log = logging.getLogger("arviz")
    _az_prev = _az_log.level
    if not save_outputs:
        _az_log.setLevel(logging.ERROR)
    idata = az.from_dict(posterior=posterior_dict)
    summary = az.summary(idata)
    _az_log.setLevel(_az_prev)

    gt_array = np.array([GROUND_TRUTH[p] for p in PARAM_NAMES])
    posterior_means = np.array([np.array(uniform_samples[name]).mean() for name in PARAM_NAMES])
    param_bias = posterior_means - gt_array

    _print("\n=== Diagnostics ===")
    _print(f"  log Z (evidence):              {log_z:.3f} ± {log_z_uncert:.3f}")
    _print(f"  JAXNS ESS (Kish estimate):     {jaxns_ess:.1f}")
    _print(f"  Total likelihood evaluations:  {total_likelihood_evals}")
    _print(f"  Likelihood evals / NS sample:  {results.total_num_likelihood_evaluations / max(1, int(results.total_num_samples)):.1f}")
    _print(f"  ESS per likelihood eval:       {ess_per_likelihood_eval:.4f}")
    _print(f"  Wall-clock time:               {wall_time_s:.2f}s")
    _print()
    _print("  Parameter recovery (posterior mean vs ground truth):")
    for name, pm, gt, bias in zip(PARAM_NAMES, posterior_means, gt_array, param_bias):
        _print(f"    {name:20s}  mean={pm:8.4f}  truth={gt:8.4f}  bias={bias:+.4f}")
    _print()
    _print("  ArviZ summary (ESS, MCSE — R-hat is trivially 1.0 for a single chain):")
    _print(summary.to_string())

    # --- Results ---
    diagnostics = {
        "sampler": "NestedSampling_JAXNS",
        "wall_time_s": float(wall_time_s),
        "num_posterior_draws": NUM_POSTERIOR_DRAWS,
        "num_live_points": NUM_LIVE_POINTS,
        "log_Z_mean": log_z,
        "log_Z_uncert": log_z_uncert,
        "total_likelihood_evals": total_likelihood_evals,
        "total_ns_samples": int(results.total_num_samples),
        "likelihood_evals_per_ns_sample": float(
            results.total_num_likelihood_evaluations / max(1, int(results.total_num_samples))
        ),
        "jaxns_ess_kish": jaxns_ess,
        "ess_per_likelihood_eval": ess_per_likelihood_eval,
        "posterior_means": {name: float(pm) for name, pm in zip(PARAM_NAMES, posterior_means)},
        "ground_truth": {k: float(v) for k, v in GROUND_TRUTH.items()},
        "param_bias": {name: float(b) for name, b in zip(PARAM_NAMES, param_bias)},
        "arviz_summary": json.loads(summary.to_json()),
    }

    if save_outputs:
        NS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        idata.to_netcdf(str(NS_OUTPUT_DIR / "sajax_idata.nc"))
        diag_path = NS_OUTPUT_DIR / "diagnostics.json"
        with open(diag_path, "w") as f:
            json.dump(diagnostics, f, indent=2)
        _print(f"\nSaved idata to {NS_OUTPUT_DIR / 'sajax_idata.nc'}")
        _print(f"Saved diagnostics to {diag_path}")

    if not save_outputs:
        return diagnostics

    # --- Plots ---

    # NS shrinkage curve
    fig, ax = plt.subplots(figsize=(10, 4))
    log_L_dead = np.array(results.log_L_samples[: int(results.total_num_samples)])
    ax.plot(log_L_dead, lw=0.6, color="steelblue", alpha=0.8)
    ax.set_xlabel("dead point index")
    ax.set_ylabel("log L")
    ax.set_title("NS shrinkage: log-likelihood of dead points")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    shrinkage_path = NS_OUTPUT_DIR / "shrinkage.png"
    fig.savefig(shrinkage_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _print(f"Saved shrinkage plot to {shrinkage_path}")

    # Corner plot — all parameters
    az.rcParams["plot.max_subplots"] = len(PARAM_NAMES) ** 2
    az.plot_pair(
        idata,
        var_names=PARAM_NAMES,
        kind="kde",
        marginals=True,
        figsize=(24, 24),
    )
    corner_path = NS_OUTPUT_DIR / "corner_all.png"
    plt.savefig(corner_path, dpi=120, bbox_inches="tight")
    plt.close()
    _print(f"Saved full corner plot to {corner_path}")

    # Best-fit light curve using posterior mean.
    # Augment uniform_samples with derived quantities for plot_bestfit_lightcurve.
    ecc_h = np.array(uniform_samples["ecc_h"])
    ecc_k = np.array(uniform_samples["ecc_k"])
    ldc_q1 = np.array(uniform_samples["ldc_q1"])
    ldc_q2 = np.array(uniform_samples["ldc_q2"])
    constrained_with_derived = {
        **{name: np.array(uniform_samples[name]) for name in PRIOR_DISTRIBUTIONS},
        "eccentricity": ecc_h**2 + ecc_k**2,
        "arg_periapsis": np.arctan2(ecc_k, ecc_h),
        "ldc_u1": 2 * np.sqrt(ldc_q1) * ldc_q2,
        "ldc_u2": np.sqrt(ldc_q1) * (1 - 2 * ldc_q2),
    }
    plot_bestfit_lightcurve(constrained_with_derived, NS_OUTPUT_DIR)

    return diagnostics


if __name__ == "__main__":
    main()
