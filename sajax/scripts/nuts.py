"""
Run NUTS on the SAJAX planet+activity model and save outputs.
"""

import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    import arviz as az
import blackjax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from model import (
    make_log_density,
    plot_model,
    _call_sajax,
    OUTPUT_DIR,
    PARAM_NAMES,
    GROUND_TRUTH,
    TIMES,
    OBS_LIGHT_CURVE,
    LAT_MIN, LAT_MAX, LONG_MIN, LONG_MAX, SIZE_MIN, SIZE_MAX,
    FLUX_MIN, FLUX_MAX, P_ROT_MIN, P_ROT_MAX, LDC_U1_MIN, LDC_U1_MAX,
    LDC_U2_MIN, LDC_U2_MAX, PLANET_RADIUS_MIN, PLANET_RADIUS_MAX,
    SEMI_MAJOR_MIN, SEMI_MAJOR_MAX, INCLINATION_MIN, INCLINATION_MAX,
    ECCENTRICITY_MIN, ECCENTRICITY_MAX, ARG_PERIAPSIS_MIN, ARG_PERIAPSIS_MAX,
    P_ORB_MIN, P_ORB_MAX,
)

NUTS_OUTPUT_DIR = OUTPUT_DIR / "nuts"

NDIM = len(PARAM_NAMES)
NUM_WARMUP = 500
NUM_SAMPLES = 1000
NUM_CHAINS = 8

PRIOR_MINS = np.array([
    LAT_MIN, LONG_MIN, SIZE_MIN, FLUX_MIN,
    LAT_MIN, LONG_MIN, SIZE_MIN, FLUX_MIN,
    P_ROT_MIN,
    PLANET_RADIUS_MIN, SEMI_MAJOR_MIN, INCLINATION_MIN,
    ECCENTRICITY_MIN, ARG_PERIAPSIS_MIN, P_ORB_MIN,
    LDC_U1_MIN, LDC_U2_MIN,
])
PRIOR_MAXS = np.array([
    LAT_MAX, LONG_MAX, SIZE_MAX, FLUX_MAX,
    LAT_MAX, LONG_MAX, SIZE_MAX, FLUX_MAX,
    P_ROT_MAX,
    PLANET_RADIUS_MAX, SEMI_MAJOR_MAX, INCLINATION_MAX,
    ECCENTRICITY_MAX, ARG_PERIAPSIS_MAX, P_ORB_MAX,
    LDC_U1_MAX, LDC_U2_MAX,
])


def get_initial_positions(key: jax.Array, num_chains: int) -> jnp.ndarray:
    key, base_key, k_prot, k_rp, k_sma, k_ecc = jax.random.split(key, 6)

    center = np.array([GROUND_TRUTH[p] for p in PARAM_NAMES])
    width = (PRIOR_MAXS - PRIOR_MINS) * 0.10
    low = np.maximum(PRIOR_MINS, center - width)
    high = np.minimum(PRIOR_MAXS, center + width)
    positions = jax.random.uniform(base_key, shape=(num_chains, NDIM), minval=low, maxval=high)

    positions = positions.at[:, 8].set(
        jnp.exp(jax.random.normal(k_prot, (num_chains,)) * 0.1 + jnp.log(center[8])))
    positions = positions.at[:, 9].set(
        jnp.exp(jax.random.normal(k_rp,   (num_chains,)) * 0.1 + jnp.log(center[9])))
    positions = positions.at[:, 10].set(
        jnp.exp(jax.random.normal(k_sma,  (num_chains,)) * 0.1 + jnp.log(5.0)))
    positions = positions.at[:, 12].set(
        jax.random.uniform(k_ecc, (num_chains,), minval=0.0, maxval=0.1))

    return positions


def inference_loop(rng_key, kernel, initial_state, num_samples):
    @jax.jit
    def one_step(state, rng_key):
        state, info = kernel(rng_key, state)
        return state, (state, info)

    keys = jax.random.split(rng_key, num_samples)
    _, (states, infos) = jax.lax.scan(one_step, initial_state, keys)
    return states, infos


def main(seed: int = 0, save_outputs: bool = True):
    init_key, warmup_key, sample_key = jax.random.split(jax.random.PRNGKey(seed), 3)
    _print = print if save_outputs else lambda *a, **kw: None

    # --- Model ---
    log_density_fn = make_log_density()
    if save_outputs:
        plot_model(filename="sajax_ground_truth.png")

    t0 = time.perf_counter()

    # --- Warmup: adapt one chain, share parameters across all chains ---
    _print(f"Running warmup ({NUM_WARMUP} steps)...")
    warmup_start = get_initial_positions(init_key, 1)[0]
    warmup = blackjax.window_adaptation(blackjax.nuts, log_density_fn)
    (_, parameters), _ = warmup.run(warmup_key, warmup_start, num_steps=NUM_WARMUP)
    _print(f"  Adapted step size: {parameters['step_size']:.4f}")

    # --- Initialize chains ---
    init_key2, sample_key = jax.random.split(sample_key)
    initial_positions = get_initial_positions(init_key2, NUM_CHAINS)

    kernel = blackjax.nuts(log_density_fn, **parameters)
    init_fn = jax.vmap(kernel.init)
    initial_states = init_fn(initial_positions)

    # --- Run chains in parallel ---
    _print(f"Sampling ({NUM_SAMPLES} steps, {NUM_CHAINS} chains, {NDIM} params)...")
    chain_sample_keys = jax.random.split(sample_key, NUM_CHAINS)

    @jax.vmap
    def run_chain(rng_key, initial_state):
        return inference_loop(rng_key, kernel.step, initial_state, NUM_SAMPLES)

    all_states, all_infos = run_chain(chain_sample_keys, initial_states)

    # all_states.position: (NUM_CHAINS, NUM_SAMPLES, NDIM)
    samples = np.array(all_states.position)

    # --- Diagnostics ---
    num_integration_steps = np.array(all_infos.num_integration_steps)
    total_grad_evals = int(num_integration_steps.sum())
    acceptance = float(np.mean(np.array(all_infos.acceptance_rate)))

    posterior_dict = {PARAM_NAMES[i]: samples[:, :, i] for i in range(NDIM)}
    idata = az.from_dict(
        posterior=posterior_dict,
        sample_stats={
            "acceptance_rate": np.array(all_infos.acceptance_rate),
            "n_steps": num_integration_steps,
        },
    )
    summary = az.summary(idata)
    total_bulk_ess = summary["ess_bulk"].sum()
    ess_per_grad = total_bulk_ess / total_grad_evals

    max_steps = num_integration_steps.max()
    saturation_frac = float(np.mean(num_integration_steps == max_steps))

    gt_array = np.array([GROUND_TRUTH[p] for p in PARAM_NAMES])
    posterior_means = samples.mean(axis=(0, 1))
    param_bias = posterior_means - gt_array

    _print("\n=== Diagnostics ===")
    _print(f"  Mean acceptance rate:       {acceptance:.3f}")
    _print(f"  Total gradient evaluations: {total_grad_evals}")
    _print(f"  Mean tree depth (steps):    {num_integration_steps.mean():.1f}  (max observed: {int(max_steps)})")
    _print(f"  Tree depth saturation:      {saturation_frac:.1%}")
    _print()
    _print("  Parameter recovery (posterior mean vs ground truth):")
    for name, pm, gt, bias in zip(PARAM_NAMES, posterior_means, gt_array, param_bias):
        _print(f"    {name:20s}  mean={pm:8.4f}  truth={gt:8.4f}  bias={bias:+.4f}")
    _print()
    _print("  ArviZ summary (R-hat, ESS, MCSE):")
    _print(summary.to_string())
    _print()
    _print(f"  Total bulk ESS: {total_bulk_ess:.1f}")
    _print(f"  Bulk ESS per gradient eval: {ess_per_grad:.4f}")

    wall_time_s = time.perf_counter() - t0
    _print(f"\n  Wall-clock time: {wall_time_s:.2f}s")

    # --- Results ---
    diagnostics = {
        "sampler": "NUTS",
        "num_chains": NUM_CHAINS,
        "num_warmup": NUM_WARMUP,
        "num_samples": NUM_SAMPLES,
        "ndim": NDIM,
        "adapted_step_size": float(parameters["step_size"]),
        "mean_acceptance_rate": float(acceptance),
        "total_grad_evals": total_grad_evals,
        "mean_integration_steps": float(num_integration_steps.mean()),
        "max_integration_steps": int(max_steps),
        "tree_depth_saturation": saturation_frac,
        "wall_time_s": float(wall_time_s),
        "total_bulk_ess": float(total_bulk_ess),
        "bulk_ess_per_grad_eval": float(ess_per_grad),
        "posterior_means": {name: float(pm) for name, pm in zip(PARAM_NAMES, posterior_means)},
        "ground_truth": {k: float(v) for k, v in GROUND_TRUTH.items()},
        "param_bias": {name: float(b) for name, b in zip(PARAM_NAMES, param_bias)},
        "arviz_summary": json.loads(summary.to_json()),
    }

    if save_outputs:
        NUTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        idata.to_netcdf(str(NUTS_OUTPUT_DIR / "sajax_idata.nc"))
        diag_path = NUTS_OUTPUT_DIR / "diagnostics.json"
        with open(diag_path, "w") as f:
            json.dump(diagnostics, f, indent=2)
        _print(f"\nSaved idata to {NUTS_OUTPUT_DIR / 'sajax_idata.nc'}")
        _print(f"Saved diagnostics to {diag_path}")

    if not save_outputs:
        return diagnostics

    # --- Plots ---

    # Trace plots for first 6 parameters
    az.plot_trace(idata, var_names=PARAM_NAMES[:6], figsize=(14, 10))
    plt.tight_layout()
    trace_path = NUTS_OUTPUT_DIR / "traces_subset.png"
    plt.savefig(trace_path, dpi=150, bbox_inches="tight")
    plt.close()
    _print(f"Saved trace plot to {trace_path}")

    # Corner plot — all parameters
    az.rcParams["plot.max_subplots"] = len(PARAM_NAMES) ** 2
    az.plot_pair(
        idata,
        var_names=PARAM_NAMES,
        kind="kde",
        marginals=True,
        figsize=(24, 24),
    )
    corner_path = NUTS_OUTPUT_DIR / "corner_all.png"
    plt.savefig(corner_path, dpi=120, bbox_inches="tight")
    plt.close()
    _print(f"Saved full corner plot to {corner_path}")

    # Best-fit light curve using posterior mean
    mean_dict = {name: float(posterior_means[i]) for i, name in enumerate(PARAM_NAMES)}

    lc_bestfit = np.array(
        _call_sajax(
            TIMES,
            np.array([mean_dict["spot_lat"], mean_dict["fac_lat"]]),
            np.array([mean_dict["spot_long"], mean_dict["fac_long"]]),
            np.array([mean_dict["spot_size"], mean_dict["fac_size"]]),
            np.stack([np.array([mean_dict["spot_flux"]]), np.array([mean_dict["fac_flux"]])]),
            mean_dict["p_rot"],
            mean_dict["planet_radius"],
            mean_dict["semimajor_axis"],
            np.deg2rad(mean_dict["inclination"]),
            mean_dict["eccentricity"],
            mean_dict["arg_periapsis"],
            mean_dict["P_orb"],
            mean_dict["LDC_u1"],
            mean_dict["LDC_u2"],
        )["lc"]
    )

    lc_true = np.array(
        _call_sajax(
            TIMES,
            np.array([GROUND_TRUTH["spot_lat"], GROUND_TRUTH["fac_lat"]]),
            np.array([GROUND_TRUTH["spot_long"], GROUND_TRUTH["fac_long"]]),
            np.array([GROUND_TRUTH["spot_size"], GROUND_TRUTH["fac_size"]]),
            np.stack([np.array([GROUND_TRUTH["spot_flux"]]), np.array([GROUND_TRUTH["fac_flux"]])]),
            GROUND_TRUTH["p_rot"],
            GROUND_TRUTH["planet_radius"],
            GROUND_TRUTH["semimajor_axis"],
            np.deg2rad(GROUND_TRUTH["inclination"]),
            GROUND_TRUTH["eccentricity"],
            GROUND_TRUTH["arg_periapsis"],
            GROUND_TRUTH["P_orb"],
            GROUND_TRUTH["LDC_u1"],
            GROUND_TRUTH["LDC_u2"],
        )["lc"]
    )

    fig, (ax_lc, ax_res) = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                                         gridspec_kw={"height_ratios": [3, 1]})

    ax_lc.scatter(TIMES, OBS_LIGHT_CURVE, s=4, color="orange", alpha=0.6,
                  label="Observations", zorder=1)
    ax_lc.plot(TIMES, lc_true, lw=2, color="steelblue", label="True", zorder=2)
    ax_lc.plot(TIMES, lc_bestfit, lw=2, color="crimson", linestyle="--",
               label="Posterior mean fit", zorder=3)
    ax_lc.set_ylabel("Normalised flux")
    ax_lc.legend(frameon=False)
    ax_lc.spines["top"].set_visible(False)
    ax_lc.spines["right"].set_visible(False)

    residuals_ppm = (OBS_LIGHT_CURVE - lc_bestfit) * 1e6
    ax_res.scatter(TIMES, residuals_ppm, s=4, color="orange", alpha=0.6)
    ax_res.axhline(0, color="crimson", lw=1, linestyle="--")
    ax_res.set_xlabel("Time [days]")
    ax_res.set_ylabel("Residuals [ppm]")
    ax_res.spines["top"].set_visible(False)
    ax_res.spines["right"].set_visible(False)

    fig.tight_layout()
    lc_path = NUTS_OUTPUT_DIR / "bestfit_lightcurve.png"
    fig.savefig(lc_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _print(f"Saved best-fit light curve to {lc_path}")

    return diagnostics


if __name__ == "__main__":
    main()
