"""
simulate_lightcurves.py

Stage 2 of the analysis pipeline: simulate LSST-like SALT2 supernova light
curves with realistic photometric noise, from the params table produced by
Stage 1 (sim_gen; see config.py -> SimGenConfig).

Input CSV schema (one row per SN):
    cosmo_id, omega_m, omega_lambda, M0_mu, M0_sigma, alpha, beta,
    sn_id, redshift, x1, c, mb_theory

`cosmo_id` identifies which cosmology (of many) this SN was drawn under.
`sn_id` identifies which SN within that cosmology (e.g. 0 .. n_sn_per_cosmo-1).


For each input row, this script:
  1. Builds a SALT2 model light curve (noiseless) in the requested bands.
  2. Adds LSST single-visit photometric noise (random + systematic terms)
     using the standard m5/gamma noise model.
  3. Applies a simple depth/quality cut on the photometric error.
  4. Writes one CSV per simulated light curve to the output directory,
     named "{cosmo_id}_{sn_id}.csv".

Reproducibility
----------------
Noise is drawn from a per-SN RNG seeded from (cosmo_id, sn_id), not from a
single global seed. This means a given SN's noisy light curve is identical

Configuration
-------------
All settings come from config.py (LightcurveConfig, part of the shared
AnalysisConfig). Edit config.py directly to change defaults.
Usage
-----
    python lc_gen.py
    
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sncosmo

from config import LightcurveConfig, config as default_config


# --------------------------------------------------------------------------
# Core simulation
# --------------------------------------------------------------------------

def photometric_error(mag: np.ndarray, band: str, cfg: LightcurveConfig) -> np.ndarray:
    """LSST single-visi           1t photometric error (mag) for given magnitudes/band."""
    m5_band = cfg.m5[band]
    gamma_band = cfg.gamma[band]
    x = 10 ** (0.4 * (mag - m5_band))
    sigma_rand_sq = (0.04 - gamma_band) * x + gamma_band * x**2
    sigma_sys_sq = cfg.sigma_sys**2
    return np.sqrt(sigma_sys_sq + sigma_rand_sq)


def per_sn_rng(cosmo_id: int, sn_id: int, cfg: LightcurveConfig) -> np.random.Generator:
    """Deterministic RNG stream keyed only on (cosmo_id, sn_id).

    Guarantees the same noisy light curve for a given SN regardless of
    processing order, subsetting, or how many other SNe are run alongside it.
    """
    seed_seq = np.random.SeedSequence([cfg.base_seed, int(cosmo_id), int(sn_id)])
    return np.random.default_rng(seed_seq)


def simulate_one_light_curve(row: pd.Series, cfg: LightcurveConfig) -> pd.DataFrame | None:
    """Simulate one noisy multi-band light curve for a single input SN.

    Returns None if any requested band has no usable point after the
    quality cut (matches original "require every band to have a
    detection" behavior).
    """
    cosmo_id = int(row["cosmo_id"])
    sn_id = int(row["sn_id"])
    redshift = row["redshift"]
    x1 = row["x1"]
    c = row["c"]
    mb_theory = row["mb_theory"]

    rng = per_sn_rng(cosmo_id, sn_id, cfg)

    times = np.arange(
        cfg.phase_min * (1 + redshift),
        cfg.phase_max * (1 + redshift),
        cfg.phase_cadence,
    )

    model = sncosmo.Model(source="salt2-extended")
    model.set(z=redshift, x1=x1, c=c)
    model.set_source_peakmag(mb_theory, "bessellb", "ab")

    per_band_frames = []

    for band in cfg.bands:
        try:
            mags = model.bandmag(band, "ab", times)
        except ValueError:
        # Band's observer-frame wavelength range falls outside the
        # SALT2 model's spectral coverage at this redshift/phase.
        # Skip just this band for this SN, keep going with the rest.
            continue
    
        errors = photometric_error(mags, band, cfg)
        noisy_mags = mags + rng.normal(0.0, errors, size=len(mags))

        mask = np.isfinite(mags) & (errors <= cfg.max_error)
        if not np.any(mask):
            continue

        per_band_frames.append(pd.DataFrame({
            "time": times[mask],
            "mag_noisy": noisy_mags[mask],
            "mag_true": mags[mask],
            "error": errors[mask],
            "filter": band,
        }))

    if not per_band_frames:
            return None

    

    return pd.concat(per_band_frames, ignore_index=True)


def run(cfg: LightcurveConfig) -> None:
    warnings.filterwarnings("ignore")  # sncosmo/astropy emit benign warnings

    print("Reading input file:", cfg.input_csv)
    df = pd.read_csv(cfg.input_csv)
    

    n_cosmologies = df["cosmo_id"].nunique()
    print(f"Loaded {len(df)} SNe across {n_cosmologies} cosmologies")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_total = len(df)
    progress_every = max(1, n_total // 100)
    n_saved = 0

    for step, (_, row) in enumerate(df.iterrows(), start=1):
        if step % progress_every == 0:
            print(f"{100 * step // n_total}%")

        lc = simulate_one_light_curve(row, cfg)
        if lc is None:
            continue

        cosmo_id = int(row["cosmo_id"])
        sn_id = int(row["sn_id"])
        out_path = output_dir / f"{cosmo_id}_{sn_id}.csv"
        lc.to_csv(out_path, index=False)
        n_saved += 1

    print(f"Done. Saved {n_saved}/{n_total} light curves to {output_dir}")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main() -> None:
    run(default_config.lightcurves)


if __name__ == "__main__":
    main()
