"""
lc_fit.py

Stage 3 of the analysis pipeline: fit an FPCA (functional PCA) template to
each simulated light curve from Stage 2 (lc_gen), separately per band, using
mpfit with Gaussian priors on the two FPCA scores (a1, a2). Priors and their
widths come from a per-band linear mapping vs. redshift (mapping_coeff.csv),
derived from the PLAsTiCC dataset.

Generalized to any band list (cfg.bands).
Add a band by adding it to cfg.bands  in config.py, as long as mapping_coeff.csv 
has a row for that band's letter.

For each SN light curve CSV in cfg.lc_dir:
  1. Skip if redshift < cfg.min_redshift, or already processed (resume log).
  2. Split points by band; a band needs more than cfg.min_points_per_band
     points to be attempted.
  3. Fit each attempted band with mpfit (best of cfg.num_attempts tries).
  4. Keep a band's fit only if it passes the quality cuts in FittingConfig.
  5. Append one output row per (SN, band) to the results table.

Every successfully fitted SN is also plotted (all fitted bands in one panel,
data + FPCA model, bands offset) and saved to cfg.plot_dir/{cosmo_id}_{sn_id}.png.

Progress is logged to cfg.log_file so a run can be resumed if interrupted —
already-logged "{cosmo_id}_{sn_id}.csv" filenames are skipped on a rerun.

Usage
-----
    python lc_fit.py
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpfit import mpfit
from numpy import sqrt, square
from scipy.interpolate import interp1d

from config import FittingConfig, config as default_config


# Plot style per band (offset in mag, so bands don't overlap).
BAND_STYLE = {
    "lsstg": {"color": "#000000", "marker": "^", "ms": 6, "offset": -2.0},
    "lsstr": {"color": "#E69F00", "marker": "h", "ms": 6, "offset": -1.0},
    "lssti": {"color": "#56B4E9", "marker": "P", "ms": 6, "offset":  0.0},
    "lsstz": {"color": "#009E73", "marker": "o", "ms": 5, "offset":  1.0},
    "lssty": {"color": "#CC79A7", "marker": "*", "ms": 8, "offset":  2.0},
}


# --------------------------------------------------------------------------
# 1. Load FPCA eigenvectors & per-band prior mapping
# --------------------------------------------------------------------------

def load_eigenvectors(path: str):
    """Load the FPCA eigenvector table and return phase-interpolators for
    the mean curve (vec0) and the first two principal components (vec1, vec2).
    """
    eigen_val = np.genfromtxt(path)
    phase = eigen_val[1:, 0]
    vec0, vec1, vec2 = eigen_val[1:, 1], eigen_val[1:, 2], eigen_val[1:, 3]

    imod0 = interp1d(phase, vec0, fill_value="extrapolate")
    imod1 = interp1d(phase, vec1, fill_value="extrapolate")
    imod2 = interp1d(phase, vec2, fill_value="extrapolate")
    return imod0, imod1, imod2


def load_mapping_coeff(path: str) -> pd.DataFrame:
    """Load the per-band linear (a1, a2) prior mapping vs. redshift, indexed
    by band letter (e.g. 'i', 'z', 'g', ...).
    """
    mapping_coeff = pd.read_csv(path)
    return mapping_coeff.set_index("Filter")


def band_priors(band_letter: str, redshift: float, mapping_coeff: pd.DataFrame):
    """Priors (b1, b2) and their std devs for one band, used as the Gaussian
    prior center/width on FPCA scores (a1, a2) in the fit.

    b1 is quadratic in redshift (m1*z^2 + c1*z + c0) if this band's row in
    mapping_coeff has a real (non-null) 'c0' value; otherwise linear
    (m1*z + c1). This is data-driven, not hardcoded per band letter — e.g.
    g/r rows can carry a c0 value to get the quadratic form, while i/z rows
    leave c0 blank/absent and stay linear. b2 is always linear (m2*z + c2).
    """
    row = mapping_coeff.loc[band_letter]
    has_quadratic_b1 = "c0" in row.index and pd.notna(row["c0"])

    if has_quadratic_b1:
        b1 = row["m1"] * redshift**2 + row["c1"] * redshift + row["c0"]
        sigma_b1_sq = (
            square(row["std_m1"]) * redshift**4
            + square(row["std_c1"]) * redshift**2
            + square(row["std_c0"])
        )
    else:
        b1 = redshift * row["m1"] + row["c1"]
        sigma_b1_sq = square(row["std_m1"] * redshift) + square(row["std_c1"])

    b2 = redshift * row["m2"] + row["c2"]
    sigma_b2_sq = square(row["std_m2"] * redshift) + square(row["std_c2"])

    return b1, b2, sqrt(sigma_b1_sq), sqrt(sigma_b2_sq)


# --------------------------------------------------------------------------
# 2. FPCA model + mpfit machinery
# --------------------------------------------------------------------------

def n_points_in_window(times: np.ndarray, t_peak: float, redshift: float, cfg: FittingConfig) -> int:
    """Count points whose rest-frame phase falls inside [phase_min, phase_max]."""
    phase = (times - t_peak) / (1 + redshift)
    return int(np.sum((phase >= -10) & (phase <= 40)))


def fpca_model(t: float, theta, arr_pass, redshift: float, imods, cfg: FittingConfig):
    """Evaluate the FPCA template (mean + a1*PC1 + a2*PC2) at one time point.

    Returns (nan, nan) if the rest-frame phase falls outside the fit window.
    """
    imod0, imod1, imod2 = imods
    t_peak, mag_offset, a1, a2 = theta
    phase = (t - t_peak) / (1 + redshift)

    if not (-10 <= phase <= 40):
        return np.nan, np.nan

    vec0, vec1, vec2 = float(imod0(phase)), float(imod1(phase)), float(imod2(phase))
    model_mag = mag_offset + vec0 + a1 * vec1 + a2 * vec2
    model_var = np.square(arr_pass[0] * vec1) + np.square(arr_pass[1] * vec2)
    return model_mag, model_var


def _mpfit_residuals(p, fjac=None, x=None, y=None, err=None, arr_pass=None,
                      ini_params=None, redshift=None, imods=None, cfg=None):
    """mpfit residual function: data residuals + Gaussian priors on (a1, a2)."""
    dev = np.zeros(len(x), dtype=np.float64)
    for k in range(len(x)):
        model_mag, model_var = fpca_model(x[k], p, arr_pass, redshift, imods, cfg)
        if not np.isnan(model_mag):
            dev[k] = (y[k] - model_mag) / sqrt(np.square(err[k]) + model_var)

    prior_dev = np.zeros(2, dtype=np.float64)
    prior_dev[0] = (p[2] - ini_params[2]) / (sqrt(2) * arr_pass[0])
    prior_dev[1] = (p[3] - ini_params[3]) / (sqrt(2) * arr_pass[1])

    return [0, np.concatenate((dev, prior_dev))]


def fit_one_band(times, mags, errors, ini_params, t_bounds, arr_pass, redshift, imods, cfg: FittingConfig):
    """Run mpfit once for one band's light curve. Returns (params, errors, chi2, dof)."""
    n_pts = n_points_in_window(times, ini_params[0], redshift, cfg)
    if n_pts < cfg.min_points_per_band:
        return np.full(4, 99.0), np.full(4, -99.0), 1.0e5, 1


    parinfo = [{"value": v, "fixed": 0, "limited": [0, 0], "limits": [0.0, 0.0]} for v in ini_params]
    parinfo[0]["limited"] = [1, 1]
    parinfo[0]["limits"] = [t_bounds[0], t_bounds[1]]
    parinfo[2]["limited"] = [1, 1]
    parinfo[2]["limits"] = [-10, 10]
    parinfo[3]["limited"] = [1, 1]
    parinfo[3]["limits"] = [-10, 10]

    fa = {
        "x": times, "y": mags, "err": errors, "arr_pass": arr_pass,
        "ini_params": ini_params, "redshift": redshift, "imods": imods, "cfg": cfg,
    }
    m = mpfit(_mpfit_residuals, parinfo=parinfo, quiet=True, functkw=fa)
    chi2 = m.fnorm if (m.status > 0 and m.perror is not None) else 100000.0
    perror = m.perror if m.perror is not None else np.zeros(4)
    return m.params, perror, chi2, n_pts - 2


# --------------------------------------------------------------------------
# 3. Per-SN fitting: split by band, fit each, apply quality cuts
# --------------------------------------------------------------------------

def split_by_band(lc: pd.DataFrame, cfg: FittingConfig) -> dict:
    """Group a light curve's points by band, keeping only bands with more
    than cfg.min_points_per_band points, sorted with the most-observed
    band first.
    """
    per_band = {}
    for band in cfg.bands:
        rows = lc[lc["filter"] == band]
        if len(rows) >= cfg.min_points_per_band:
            per_band[band] = {
                "time": rows["time"].to_numpy(),
                "mag": rows["mag_noisy"].to_numpy(),
                "err": rows["error"].to_numpy(),
            }
    # Sort bands by point count, descending (most-sampled band fit first).
    return dict(sorted(per_band.items(), key=lambda kv: -len(kv[1]["time"])))


def passes_quality_cuts(errors: np.ndarray, chi2: float, dof: int, cfg: FittingConfig) -> bool:
    if np.isnan(errors).any():
        return False
    return (
        errors[0] < cfg.max_err_pk_time
        and errors[1] < cfg.max_err_pk_mag
        and errors[2] < cfg.max_err_a1
        and errors[3] < cfg.max_err_a2
        and dof > 0
        and chi2 / dof < cfg.max_reduced_chi2
        and chi2 > -0.001
    )


def per_sn_fit_rng(cosmo_id: int, sn_id: int, cfg: FittingConfig) -> np.random.Generator:
    """Reproducible RNG for this SN's per-attempt jittering, independent of
    lc_gen's noise RNG and of any other SN's -- same seeding pattern as
    lc_gen.per_sn_rng, so reruns/subsets/reordering give identical fits."""
    seed_seq = np.random.SeedSequence([cfg.base_seed, int(cosmo_id), int(sn_id)])
    return np.random.default_rng(seed_seq)


def fit_all_bands(per_band: dict, redshift: float, cosmo_id: int, sn_id: int,
                   imods, mapping_coeff: pd.DataFrame, cfg: FittingConfig) -> dict:
    """Fit every band in per_band, keeping only fits that pass quality cuts.
    Returns {band: {"params", "errors", "chi2", "dof"}} for surviving bands.
    """
    rng = per_sn_fit_rng(cosmo_id, sn_id, cfg)

    t_peak_guess = per_band[next(iter(per_band))]["time"][
        np.argmin(per_band[next(iter(per_band))]["mag"])
    ]

    results = {}
    fitted_peak_time = None  # set once the first (best-sampled) band's fit succeeds

    for i, (band, data) in enumerate(per_band.items()):
        
        b1, b2, sigma_b1, sigma_b2 = band_priors(band, redshift, mapping_coeff)
        arr_pass = np.array([sigma_b1, sigma_b2])

        # First band anchors on its own brightest-point guess; every later
        # band anchors on the first band's FITTED peak time instead, falling
        # back to the raw guess only if the first band's fit failed.
        t_anchor = t_peak_guess if i == 0 else fitted_peak_time
        t_bounds = (
            t_anchor - 7 * (1 + redshift),
            t_anchor + 7 * (1 + redshift),
        )

        best_chi2 = np.inf
        best_result = None
        for _ in range(cfg.num_attempts):
            ini_params = np.array([
                t_anchor + rng.uniform(-cfg.jitter_time_days, cfg.jitter_time_days),
                np.min(data["mag"]) + rng.uniform(-cfg.jitter_mag, cfg.jitter_mag),
                b1 + rng.uniform(-cfg.jitter_a1, cfg.jitter_a1),
                b2 + rng.uniform(-cfg.jitter_a2, cfg.jitter_a2),
            ])
            params, errors, chi2, dof = fit_one_band(
                data["time"], data["mag"], data["err"], ini_params, t_bounds, arr_pass, redshift, imods, cfg
            )
            if chi2 < best_chi2:
                best_chi2, best_result = chi2, (params, errors, chi2, dof)

        params, errors, chi2, dof = best_result
        band_ok = passes_quality_cuts(errors, chi2, dof, cfg)

        if i == 0:
            if not band_ok:
                return {}  # best-sampled band failed -> abandon this SN entirely
            fitted_peak_time = params[0]

        if band_ok:
            results[band] = {"params": params, "errors": errors, "chi2": chi2, "dof": dof}

    return results


def fit_one_sn(lc: pd.DataFrame, row: pd.Series, cosmo_id: int, sn_id: int,
                imods, mapping_coeff: pd.DataFrame, cfg: FittingConfig):
    """Fit every requested band for one SN's light curve.

    Returns a list of output rows (one per surviving band), or None if the
    SN doesn't meet the require_all_bands criterion.
    """
    redshift = row["redshift"]
    if redshift < cfg.min_redshift:
        return None

    per_band = split_by_band(lc, cfg)
    if len(per_band) < 1:
        return None

    results = fit_all_bands(per_band, redshift, cosmo_id, sn_id, imods, mapping_coeff, cfg)

    if cfg.require_all_bands:
        if not all(band in results for band in cfg.bands):
            return None
    elif not results:
        return None

    #plot_sn_fit(lc, results, redshift, cosmo_id, sn_id, imods, cfg)

    out_rows = []
    for band, res in results.items():
        p, e = res["params"], res["errors"]
        out_rows.append({
            "cosmo_id": cosmo_id, "sn_id": sn_id,
            "pk_time": p[0], "pk_mag": p[1], "a1": p[2], "a2": p[3],
            "std_time": e[0], "std_mag": e[1], "std_a1": e[2], "std_a2": e[3],
            "chi_2": res["chi2"], "dof": res["dof"],
            "redshift": redshift,
            "omega_m": row.get("omega_m"), "omega_lambda": row.get("omega_lambda"),
            "filter": band,
        })
    return out_rows


# --------------------------------------------------------------------------
# 4. Plot
# --------------------------------------------------------------------------

def plot_sn_fit(lc: pd.DataFrame, results: dict, redshift: float, cosmo_id: int, sn_id: int,
                imods, cfg: FittingConfig) -> None:
    """All fitted bands in one panel: data (with errors) + FPCA model, each band offset."""
    imod0, imod1, imod2 = imods
    phase = np.linspace(-10, 40, 300)
    v0, v1, v2 = imod0(phase), imod1(phase), imod2(phase)

    bands = list(results)
    t_ref = results[bands[0]]["params"][0]          # peak time of the first (best-sampled) band

    fig, ax = plt.subplots(figsize=(7, 5))
    for band in bands:
        st = BAND_STYLE.get(band, {"color": "gray", "marker": "o", "ms": 5, "offset": 0.0})
        off = st["offset"]
        name = band.replace("lsst", "")
        label = f"{name}{off:+.1f}" if off else name

        t_peak, mag_o, a1, a2 = results[band]["params"]
        model_t = phase * (1 + redshift) + t_peak - t_ref
        model_m = mag_o + v0 + a1 * v1 + a2 * v2 + off
        ax.plot(model_t, model_m, color=st["color"], linestyle="solid")

        data = lc[lc["filter"] == band]
        ax.errorbar(data["time"] - t_ref, data["mag_noisy"] + off, yerr=data["error"],
                    color=st["color"], fmt=st["marker"], markersize=st["ms"],
                    elinewidth=1.5, label=label)

    ax.set_xlim(-10 * (1 + redshift) - 7, 40 * (1 + redshift) + 7)
    ax.invert_yaxis()
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Magnitude")
    ax.set_title(f"ID = {cosmo_id}_{sn_id}    z = {redshift:.2f}")
    ax.legend()

    
    fig.savefig(os.path.join(cfg.plot_dir, f"{cosmo_id}_{sn_id}.png"))
    plt.close(fig)


# --------------------------------------------------------------------------
# 5. Resume-tracking log helpers
# --------------------------------------------------------------------------

def load_processed_set(log_file: str) -> set:
    if not os.path.isfile(log_file):
        return set()
    with open(log_file) as f:
        return {line.strip() for line in f if line.strip()}


def append_to_log(log_file: str, filename: str) -> None:
    with open(log_file, "a") as f:
        f.write(filename + "\n")


# --------------------------------------------------------------------------
# 6. Main loop
# --------------------------------------------------------------------------

def run(cfg: FittingConfig) -> None:
    warnings.filterwarnings("ignore")

    imods = load_eigenvectors(cfg.eigen_val_path)
    mapping_coeff = load_mapping_coeff(cfg.mapping_coeff_path)

    meta = pd.read_csv(cfg.meta_csv)
    meta_index = meta.set_index(["cosmo_id", "sn_id"])

    processed = load_processed_set(cfg.log_file)
    n_fit_per_cosmo: dict[int, int] = {}

    lc_dir = Path(cfg.lc_dir)
    lc_files = sorted(
        lc_dir.glob("*_*.csv"),
        key=lambda p: tuple(int(x) for x in p.stem.split("_")),
    )
    print(f"Found {len(lc_files)} light curve files in {lc_dir}")

    all_rows = []
    n_saved_sn = 0
    n_seen = 0

    from tqdm import tqdm
    pbar = tqdm(total=cfg.max_total_fits, desc="fitting light curves")
    for lc_path in lc_files:
        filename = lc_path.name
        if filename in processed:
            continue

        cosmo_id, sn_id = (int(x) for x in lc_path.stem.split("_"))

        

        if n_fit_per_cosmo.get(cosmo_id, 0) >= cfg.max_sn_per_cosmo:
            continue

        if cfg.max_total_fits is not None and n_saved_sn >= cfg.max_total_fits:
            break

        try:
            row = meta_index.loc[(cosmo_id, sn_id)]
        except KeyError:
            append_to_log(cfg.log_file, filename)
            continue

        lc = pd.read_csv(lc_path)
        out_rows = fit_one_sn(lc, row, cosmo_id, sn_id, imods, mapping_coeff, cfg)

        append_to_log(cfg.log_file, filename)
        n_seen += 1
       

        if out_rows is None:
            continue

        all_rows.extend(out_rows)
        n_fit_per_cosmo[cosmo_id] = n_fit_per_cosmo.get(cosmo_id, 0) + 1
        n_saved_sn += 1
        pbar.update(1)

    pbar.close()

    full_cosmos = {c for c, n in n_fit_per_cosmo.items() if n == cfg.max_sn_per_cosmo}
    all_rows = [r for r in all_rows if r["cosmo_id"] in full_cosmos]
    n_saved_sn = len(full_cosmos) * cfg.max_sn_per_cosmo


    result_df = pd.DataFrame(all_rows)
    write_header = not os.path.exists(cfg.raw_csv)
    result_df.to_csv(cfg.raw_csv, mode="a" if not write_header else "w", index=False, header=write_header)

    print(f"Done. Fit {n_saved_sn} SNe ({len(result_df)} band-rows) -> {cfg.raw_csv}")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def main() -> None:
    run(default_config.fitting)


if __name__ == "__main__":
    main()
