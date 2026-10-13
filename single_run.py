"""
train_sbi.py

Trains an SBI (SNPE) posterior for (omega_m, omega_lambda) from PREPARED
feature CSVs and evaluates it on a held-out test observation.

The feature CSVs must be made beforehand with
processing.build_train_test_features(), which:
  - drops bands with no data anywhere in the TRAINING set,
  - zero-fills bands missing for only some SNe (train and test),
  - uses the same feature columns for train and test.

Each feature CSV contains: ID columns (cosmo_id, sn_id), the feature
columns (redshift + per-band mag/a1/a2), and the true parameter columns.

Train and test are SEPARATE simulation runs (e.g. train drawn from a broad
prior, test drawn at a single cosmology).

Usage
-----
    python train_sbi.py
"""

from __future__ import annotations

import random

import corner
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sbi import utils as sbi_utils
from sbi.inference import SNPE
from config import config, PARAM_COLS, PARAM_LATEX


TRAIN_FEATURES_CSV = config.sbi.train_features_csv
TEST_FEATURES_CSV  = config.sbi.test_features_csv

PARAM_COLS = list(PARAM_COLS)
PARAM_LABELS = [PARAM_LATEX[p] for p in PARAM_COLS]

ID_COLS = ["cosmo_id", "sn_id"]


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_features(features_csv: str) -> tuple[pd.DataFrame, list[str]]:
    """Read a prepared feature CSV.

    Feature columns = every column except the ID and true-parameter
    columns, in the order they appear in the file.
    """
    df = pd.read_csv(features_csv)
    missing_params = [c for c in PARAM_COLS if c not in df.columns]
    if missing_params:
        raise ValueError(f"Parameter columns not found in {features_csv}: {missing_params}")
    feature_cols = [c for c in df.columns if c not in ID_COLS + PARAM_COLS]
    return df, feature_cols


def train_posterior(train_df: pd.DataFrame, feature_cols: list[str]):
    x = torch.tensor(train_df[feature_cols].values, dtype=torch.float32)
    theta = torch.tensor(train_df[PARAM_COLS].values, dtype=torch.float32)
    print(f"x.shape {x.shape}")

    prior = sbi_utils.BoxUniform(
        low=torch.tensor([
            config.sbi.omega_m_bounds[0],
            config.sbi.omega_l_bounds[0],
        ]),
        high=torch.tensor([
            config.sbi.omega_m_bounds[1],
            config.sbi.omega_l_bounds[1],
        ]),
    )

    inference = SNPE(prior=prior, density_estimator="mdn")
    inference = inference.append_simulations(theta, x)
    density_estimator = inference.train()
    return inference.build_posterior(density_estimator)


def plot_posterior(samples: np.ndarray, true_values: list[float]):
    for i, param in enumerate(PARAM_COLS):
        vals = samples[:, i]
        median_val = np.median(vals)
        p16, p84 = np.percentile(vals, [16, 84])
        p2_5, p97_5 = np.percentile(vals, [2.5, 97.5])
        print(f"{param}:")
        print(f"  Median = {median_val:.4f}")
        print(f"  1-sigma (68%) lower = {p16:.4f}, upper = {p84:.4f}")
        print(f"  2-sigma (95%) lower = {p2_5:.4f}, upper = {p97_5:.4f}")

    fig = corner.corner(
        samples,
        labels=PARAM_LABELS,
        show_titles=True,
        title_fmt=".3f",
        color="skyblue",
        title_kwargs={"fontsize": 14},
        label_kwargs={"fontsize": 18},
        levels=[0.68, 0.95],
        hist_kwargs={"linewidth": 3.0},
        smooth=1,
    )

    axes = np.array(fig.axes).reshape(len(PARAM_COLS), len(PARAM_COLS))
    for i, param in enumerate(PARAM_COLS):
        ax = axes[i, i]
        vals = samples[:, i]
        ax.axvline(true_values[i], color="red", linestyle="solid", linewidth=2)
        p16, p84 = np.percentile(vals, [16, 84])
        ax.axvline(p16, color="black", linestyle="dashed", linewidth=2)
        ax.axvline(p84, color="black", linestyle="dashed", linewidth=2)
        p2_5, p97_5 = np.percentile(vals, [2.5, 97.5])
        ax.axvline(p2_5, color="black", linestyle="dotted", linewidth=2.3)
        ax.axvline(p97_5, color="black", linestyle="dotted", linewidth=2.3)

    for ax in fig.get_axes():
        for coll in ax.collections:
            coll.set_linewidth(3)

    plt.tight_layout()
    plt.savefig(config.sbi.figure_path, dpi=300)
    plt.show()

def run(train_features_csv: str = TRAIN_FEATURES_CSV,
        test_features_csv: str = TEST_FEATURES_CSV, test_index: int = 0):
    set_seeds(config.sbi.seed)

    train_df, feature_cols = load_features(train_features_csv)
    test_df, test_feature_cols = load_features(test_features_csv)

    if set(test_feature_cols) != set(feature_cols):
        raise ValueError("Train/test feature columns differ:\n"
        f"  only in train: {sorted(set(feature_cols) - set(test_feature_cols))}\n"
        f"  only in test:  {sorted(set(test_feature_cols) - set(feature_cols))}"
    )

    print(f"{len(train_df)} training SNe, {len(feature_cols)} feature columns: {feature_cols}")

    posterior = train_posterior(train_df, feature_cols)

    test_row = test_df.iloc[[test_index]]
    x_obs = torch.tensor(test_row[feature_cols].values, dtype=torch.float32)
    true_values = [test_row[p].iloc[0] for p in PARAM_COLS]

    samples = posterior.sample((config.sbi.n_posterior_samples,), x=x_obs).numpy()

    print(f"true_values {true_values}")
    plot_posterior(samples, true_values)
   

    a = abs(np.median(samples[:, 0] - true_values[0]))
    b = abs(np.median(samples[:, 1] - true_values[1]))
   

    

    return posterior, samples


if __name__ == "__main__":
    run(TRAIN_FEATURES_CSV, TEST_FEATURES_CSV)
