"""
bias_unc.py

Stage 7 of the analysis pipeline: retrain the posterior once per seed
(config.bias_unc.seeds), evaluate every test set, and save the bias and
1-sigma uncertainty per test set to a CSV, then plot them.

A "test set" is one simulated data set at one true cosmology (one row of the
test features file). For each seed, every test set gets, per parameter p:

    pred = posterior median
    unc  = 1-sigma uncertainty = (84th - 16th percentile) / 2
    err  = (pred - true) / unc        (signed bias/sigma)

These are then combined across seeds. The results file has ONE ROW PER TEST
SET, with, for each parameter p and each statistic s in (min, med, mean):

    {p}_true          true value
    {p}_err_{s}       bias/sigma of the selected seed
    {p}_unc_{s}       1-sigma of that same seed
    {p}_pred_{s}      posterior median of that same seed

The seed is selected per test set and per parameter:
    min   -> seed with the smallest |err|
    med   -> seed whose |err| is closest to the median |err| over seeds
    mean  -> seed whose |err| is closest to the mean |err| over seeds

Usage (notebook)
----------------
    from bias_unc import run_seeds, summary, plot_bias_unc
    run_seeds()
    summary()          # uses config.bias_unc.stat
    plot_bias_unc()    # uses config.bias_unc.stat
"""

from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from config import config, PARAM_COLS, PARAM_LATEX

PARAM_COLS = list(PARAM_COLS)
STATS = ("min", "med", "mean")


# --------------------------------------------------------------------------
# 1. Test sets
# --------------------------------------------------------------------------

def make_test_sets(test_df: pd.DataFrame, feature_cols: list[str]):
    """Test features file (one row per data set) -> one test set per row.

    Yields (cosmo_id, x_obs, truth), with x_obs of shape 1 x (n_sn * n_features).
    """
    for _, row in test_df.iterrows():
        x = row[feature_cols].to_numpy(dtype=np.float32).reshape(1, -1)
        truth = row[PARAM_COLS].to_numpy(dtype=float)
        yield row["cosmo_id"], torch.tensor(x), truth


# --------------------------------------------------------------------------
# 2. Bias and uncertainty for one seed
# --------------------------------------------------------------------------

def summarize_samples(samples: np.ndarray, truth: np.ndarray) -> dict:
    """pred, unc and err (bias/sigma) for one test set's posterior samples."""
    row = {}
    for i, p in enumerate(PARAM_COLS):
        pred = np.median(samples[:, i])
        p16, p84 = np.percentile(samples[:, i], [16, 84])
        unc = (p84 - p16) / 2
        row[f"{p}_true"] = truth[i]
        row[f"{p}_pred"] = pred
        row[f"{p}_unc"] = unc
        row[f"{p}_err"] = (pred - truth[i]) / unc
    return row


def evaluate_posterior(posterior, test_sets: Iterable, seed: int,
                       n_samples: int = config.bias_unc.n_posterior_samples) -> pd.DataFrame:
    """Sample the posterior for every test set; one row per test set."""
    rows = []
    for cosmo_id, x_obs, truth in test_sets:
        samples = posterior.sample((n_samples,), x=x_obs,
                                   show_progress_bars=False).numpy()
        rows.append({"cosmo_id": cosmo_id, "seed": seed, **summarize_samples(samples, truth)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 3. Combine seeds: min / median / mean bias per test set
# --------------------------------------------------------------------------

def combine_seeds(per_seed: pd.DataFrame) -> pd.DataFrame:
    """One row per test set: for each parameter, the err/unc/pred of the
    seed selected by min, median and mean |err| (see module docstring)."""
    rows = []
    for cosmo_id, group in per_seed.groupby("cosmo_id"):
        row = {"cosmo_id": cosmo_id}
        for p in PARAM_COLS:
            row[f"{p}_true"] = group[f"{p}_true"].iloc[0]
            abs_err = group[f"{p}_err"].abs().to_numpy()
            targets = {"min": abs_err.min(), "med": np.median(abs_err), "mean": abs_err.mean()}
            for s, target in targets.items():
                k = int(np.argmin(np.abs(abs_err - target)))
                row[f"{p}_err_{s}"] = group[f"{p}_err"].iloc[k]
                row[f"{p}_unc_{s}"] = group[f"{p}_unc"].iloc[k]
                row[f"{p}_pred_{s}"] = group[f"{p}_pred"].iloc[k]
        rows.append(row)
    return pd.DataFrame(rows)


def run_seeds(seeds=config.bias_unc.seeds,
              results_csv: str = config.bias_unc.results_csv) -> pd.DataFrame:
    """Retrain once per seed, evaluate every test set, combine the seeds and
    save one row per test set to results_csv."""
    from single_run import set_seeds, load_features, train_posterior

    train_df, feature_cols = load_features(config.sbi.train_features_csv)
    test_df, _ = load_features(config.sbi.test_features_csv)

    per_seed = []
    for n, seed in enumerate(seeds, start=1):
        print(f"--- seed {seed} ({n}/{len(seeds)}) ---")
        set_seeds(seed)
        posterior = train_posterior(train_df, feature_cols)
        per_seed.append(evaluate_posterior(posterior, make_test_sets(test_df, feature_cols), seed))

    results = combine_seeds(pd.concat(per_seed, ignore_index=True))
    results.to_csv(results_csv, index=False)
    print(f"Saved {len(results)} test sets ({len(seeds)} seeds) -> {results_csv}")
    return results


# --------------------------------------------------------------------------
# 4. Summary and plot
# --------------------------------------------------------------------------

def summary(results_csv: str = config.bias_unc.results_csv) -> pd.DataFrame:
    """Median and mean over test sets of bias/sigma and its 1-sigma, for each
    parameter and each statistic (min, med, mean)."""
    df = pd.read_csv(results_csv)
    cols = [f"{p}_{k}_{s}" for s in STATS for p in PARAM_COLS for k in ("err", "unc")]
    table = df[cols].agg(["median", "mean"]).T
    print(f"Bias/sigma and 1-sigma over {len(df)} test sets:")
    print(table.round(3))
    return table



def plot_bias_unc(stat: str = config.bias_unc.stat,
                  results_csv: str = config.bias_unc.results_csv,
                  figure_path: str = config.bias_unc.figure_path):
    """Top row: bias/sigma vs true value. Bottom row: 1-sigma vs true value.
    One column per parameter. stat = "min", "med" or "mean"."""
    df = pd.read_csv(results_csv)

    plt.rcParams.update({
        "font.size": 13, "axes.labelsize": 13,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "axes.linewidth": 1.0, "figure.dpi": 150,
    })

    n = len(PARAM_COLS)
    fig, axes = plt.subplots(2, n, sharex="col", figsize=(5 * n, 7), squeeze=False)

    for j, p in enumerate(PARAM_COLS):
        label = PARAM_LATEX.get(p, p)

        ax = axes[0, j]
        ax.scatter(df[f"{p}_true"], df[f"{p}_err_{stat}"], color="dodgerblue", s=10)
        ax.axhline(0, color="green")
        ax.axhline(1, color="red")
        ax.axhline(-1, color="red")
        ax.set_ylabel(rf"bias/$\sigma$ ({label})")

        ax = axes[1, j]
        ax.scatter(df[f"{p}_true"], df[f"{p}_unc_{stat}"], color="magenta", s=10)
        ax.set_xlabel(label, fontsize=16)
        ax.set_ylabel(rf"1-$\sigma$ ({label})")

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.15, wspace=0.26)
    plt.savefig(figure_path, dpi=300)
    print(f"Saved figure -> {figure_path}")
    plt.show()
