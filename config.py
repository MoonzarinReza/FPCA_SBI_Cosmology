"""
config.py

Single configuration for the full analysis pipeline:

  Stage 1 (sim_gen)      : draw cosmologies + per-SN SALT2 params from priors,
                            write params to sim_gen.output_path.
  Stage 2 (lightcurves)   : read that params table, simulate LSST-like noisy
                            light curves, write one CSV per SN to
                            lightcurves.output_dir.
  Stage 3 (fitting)       : fit FPCA templates (mpfit) to Stage 2's light
                            curves per band, write one row per (SN, band)
                            to fitting.raw_csv.
  Stage 4 (processing)    : merge bands -> processing.merged_csv.
  Stage 5 (prepare_features): build SBI inputs -> prepare_features.features_csv.
  Stage 6 (sbi)           : train SNPE on the train/test feature files.


Usage
-----
    from config import config
    config.sim_gen.n_cosmologies
    config.lightcurves.bands
    config.fitting.bands

To edit settings,  edit the field values below directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field




# Parameters inferred by SBI = the true-value columns written by sim_gen.
# Used by processing.py and train_sbi.py so all stages agree.

PARAM_LATEX: dict[str, str] = {
    "omega_m":      r"$\Omega_m$",
    "omega_lambda": r"$\Omega_\Lambda$",
    "omega_k":      r"$\Omega_k$",
    "w0":           r"$w_0$",
    "wa":           r"$w_a$",
    "H0":           r"$H_0$",
}

PARAM_COLS: tuple[str, ...] = ("omega_m", "omega_lambda")
@dataclass
class SimGenConfig:
    """Stage 1: draw cosmologies and per-SN SALT2 params from priors."""

    # --- run size ---
    n_cosmologies: int = 500       # number of (Om, Ol) draws
    n_sn_per_cosmo: int = 20       # number of SNe simulated per cosmology
    seed: int | None = 0         # set for reproducible draws

    # --- fixed cosmology params ---
    H0: float = 67.4              # km/s/Mpc, held fixed

    # --- cosmology priors (Om, Ol drawn independently, uniform) ---
    cosmo_prior: str = "uniform"          # "uniform" or "gaussian"

    # uniform: (low, high)
    omega_m_bounds: tuple[float, float] = (0.0, 1.0)
    omega_l_bounds: tuple[float, float] = (0.0, 1.0)

    # gaussian: truncated normal (mu, sigma, lower, upper)
    omega_m_gauss: tuple[float, float, float, float] = (0.3, 0.1, 0.0, 0.6)
    omega_l_gauss: tuple[float, float, float, float] = (0.7, 0.1, 0.3, 1.0)

    # --- redshift distribution binning ---
    z_min: float = 0.0
    z_max: float = 1.6
    z_bin_width: float = 0.1

    # --- SALT2-like nuisance parameter priors ---
    # alpha, beta: truncated normal (mu, sigma, lower, upper)
    alpha_prior: tuple[float, float, float, float] = (0.167, 0.012, 0.131, 0.203)
    beta_prior: tuple[float, float, float, float] = (3.51, 0.16, 3.03, 3.99)

    # absolute magnitude population: M0_mu ~ Normal, M0_sigma ~ Uniform
    MB_mu_loc: float = -19.3
    MB_mu_scale: float = 0.1
    MB_sigma_bounds: tuple[float, float] = (0.08, 0.12)

    # per-SN stretch/color, and M_B draw around MB_mu/MB_sigma
    x1_prior: tuple[float, float, float, float] = (0.0, 1.0, -3.0, 3.0)
    c_prior: tuple[float, float, float, float] = (0.0, 0.1, -0.3, 0.3)
    MB_abs_bounds: tuple[float, float] = (-19.8, -18.8)

    # --- output: the params table consumed by Stage 2 ---
    output_path: str = ""   # set by AnalysisConfig(split=...)




@dataclass
class LightcurveConfig:
    """Stage 2: simulate LSST-like noisy SALT2 light curves from Stage 1's output."""

    # Defaults to Stage 1's output — override only if diverging on purpose.
    input_csv: str = ""     # set by AnalysisConfig(split=...)
    output_dir: str = ""    # set by AnalysisConfig(split=...)

    # Bands to simulate. Keys must match m5 / gamma below.
    bands: tuple[str, ...] = ( 'lsstr', "lssti", "lsstz")

    # Cap on number of SNe simulated PER COSMOLOGY (not a flat row count) —
    # e.g. 5 means at most 5 SNe from each cosmo_id, however many cosmologies
    # there are. None = process every SN in every cosmology.
  
    
   

    # Phase grid: rest-frame days relative to peak, stretched by (1+z).
    phase_min: float = -15.0
    phase_max: float = 45.0
    phase_cadence: float = 4.0

    # LSST 5-sigma depth (m5) and gamma per band (LSST Science Book values).
    m5: dict = field(default_factory=lambda: {
        "lsstg": 25.0, "lsstr": 24.7, "lssti": 24.0, "lsstz": 23.3, "lssty": 22.1,
    })
    gamma: dict = field(default_factory=lambda: {
        "lsstg": 0.038, "lsstr": 0.039, "lssti": 0.039, "lsstz": 0.040, "lssty": 0.040,
    })

    # Irreducible systematic photometric error floor (mag), added in quadrature.
    sigma_sys: float = 0.005

    # Quality cut: reject points with total photometric error above this (mag).
    max_error: float = 0.36

    # Base seed; combined with (cosmo_id, sn_id) for per-SN reproducibility.
    # Independent of SimGenConfig.seed, which governs param *draws*, not noise.
    base_seed: int = 42


@dataclass
class FittingConfig:
    """Stage 3: fit FPCA templates (via mpfit) to Stage 2's noisy light curves."""

    # Defaults to Stage 2's output — override only if diverging on purpose.
    lc_dir: str = ""        # set by AnalysisConfig(split=...)
    meta_csv: str = ""      # set by AnalysisConfig(split=...)

    # Bands to fit. Each must have an entry in band_letter_map below.
   
    bands: tuple[str, ...] = ( "lsstr", "lssti", "lsstz")

    # Maps a band name to the single-letter key used in mapping_coeff.csv
    # (e.g. LSST's "lssti" -> the "i" row). Extend this if you add bands.
    

    # --- FPCA template + per-band prior mapping inputs ---
    eigen_val_path: str = "eigen_val.txt"
    mapping_coeff_path: str = "mapping_coeff_original.csv"

    # --- fit behavior ---
    num_attempts: int = 1          # mpfit re-tries per band per SN; best (lowest chi2) kept
    min_points_per_band: int = 5   # band needs more than this many points to attempt a fit
   

    time_window_days: float = 7.0

    base_seed: int = 42
    jitter_time_days: float = 0.0
    jitter_mag: float = 0.0
    jitter_a1: float = 0.0
    jitter_a2: float = 0.0

   
    # --- SN-level cuts before fitting ---
    min_redshift: float = 0.05     # skip SNe below this redshift entirely

    # --- per-band fit quality cuts (all must pass for a band's fit to be kept) ---
    max_err_pk_time: float = 10
    max_err_pk_mag: float = 3
    max_err_a1: float = 4.0
    max_err_a2: float = 3.0
    max_reduced_chi2: float = 25.0

    # Require every band in `bands` to pass its quality cuts for the SN to be
    # kept at all. If False, a SN is kept as long as at least one band passes.
    require_all_bands: bool = False

    # --- run size / resume behavior ---
    max_sn_per_cosmo: int  = 3   # cap fits per cosmology; None = no cap
    max_total_fits: int | None = None   # optional global cap across the whole run; None = no cap
   

    # --- output ---
    raw_csv: str = ""       # set by AnalysisConfig(split=...)
    log_file: str = ""      # set by AnalysisConfig(split=...); resume log
    plot_dir: str = ""      # set by AnalysisConfig(split=...); light-curve fit plots
@dataclass
class ProcessingConfig:
    """Stage 4 (processing.py): merge bands into one row per SN."""

    merged_csv: str = ""    # set by AnalysisConfig(split=...); one row per SN


@dataclass
class PrepareFeaturesConfig:
    """Stage 5 (prepare_features.py): zero-filled, NaN-free SBI feature matrix."""

    features_csv: str = ""  # set by AnalysisConfig(split=...)

@dataclass
class SBIConfig:
    """Stage 6 (train_sbi): train an SNPE posterior on Stage 5's feature files."""

    seed: int = 42

    # Number of posterior samples drawn for the held-out test observation.
    n_posterior_samples: int = 100000

    # Uniform prior bounds for (omega_m, omega_lambda), independent of
    # SimGenConfig's draw bounds -- this is the box the posterior is trained
    # and sampled against, not the prior SNe were simulated from.
    omega_m_bounds: tuple[float, float] = (0.0, 1.0)
    omega_l_bounds: tuple[float, float] = (0.0, 1.0)

    # Feature files SBI reads (always both train and test).
    train_features_csv: str = "data/train_features.csv"
    test_features_csv: str = "data/test_features.csv"
    figure_path: str = "figures/posterior_plot.png"

@dataclass
class BiasUncConfig:
    """Stage 7 (bias_unc.py): bias and uncertainty per test set, saved and plotted."""

    # Posterior samples drawn per test set.
    n_posterior_samples: int = 100000

    stat: str = "min"

    # Training seeds: the SBI network is retrained once per seed, and every
    # test set is evaluated for each. Number of seeds = len(seeds).
    seeds: tuple[int, ...] = (0, 20, 42)

    # One row per (test set, seed); new seeds are appended to the same file.
    results_csv: str = "results/bias_unc.csv"
    figure_path: str = "figures/bias_unc.png"


@dataclass
class AnalysisConfig:
    """Top-level config. `split` ("train" or "test") sets all file paths."""

    split: str = "train"
    sim_gen: SimGenConfig = field(default_factory=SimGenConfig)
    lightcurves: LightcurveConfig = field(default_factory=LightcurveConfig)
    fitting: FittingConfig = field(default_factory=FittingConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    prepare_features: PrepareFeaturesConfig = field(default_factory=PrepareFeaturesConfig)
    sbi: SBIConfig = field(default_factory=SBIConfig)
    bias_unc: BiasUncConfig = field(default_factory=BiasUncConfig)

    def __post_init__(self):
        s = self.split
        params = f"data/{s}_params_theory.csv"
        lc_dir = f"data/lc_{s}"

        if s == "train":
            self.sim_gen.cosmo_prior = "uniform"
        else:
            self.sim_gen.cosmo_prior = "gaussian"

        self.sim_gen.output_path    = params
        self.lightcurves.input_csv  = params
        self.lightcurves.output_dir = lc_dir
        self.fitting.lc_dir         = lc_dir
        self.fitting.meta_csv       = params
        self.fitting.raw_csv        = f"data/{s}_fit_raw.csv"
        self.fitting.log_file       = f"data/{s}_fit_raw.log"
        self.fitting.plot_dir       = f"plots/lc_fit_{s}"
        self.processing.merged_csv   = f"data/{s}_merged.csv"
        self.prepare_features.features_csv = f"data/{s}_features.csv"

        os.makedirs("data", exist_ok=True)
        os.makedirs(lc_dir, exist_ok=True)
        os.makedirs("figures", exist_ok=True)
        os.makedirs("results", exist_ok=True)
        os.makedirs(self.fitting.plot_dir, exist_ok=True)


config = AnalysisConfig(split="train")
