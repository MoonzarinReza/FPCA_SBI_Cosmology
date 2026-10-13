"""
prepare_features.py

Stage 5 of the analysis pipeline: turn the train and test merged tables
(processing.merged_csv, one row per SN, NaN for missing bands) into
fixed-width, NaN-free feature matrices for SBI.

  - Bands with no data anywhere in the TRAINING set are dropped.
  - Bands missing for only some SNe are zero-filled (train and test).
  - Train and test get exactly the same feature columns.

Output per split (features_csv): one row per data set (cosmo_id). Each
set keeps fitting.max_sn_per_cosmo SNe (first n by sn_id), sorted by
redshift, with columns redshift_0, mag_<band>_0, ..., redshift_<n-1>, ... .
Sets with fewer SNe than that are dropped.

This is a lossy transform (a zero-filled "missing" is no longer
distinguishable from a genuine zero), which is why it is kept separate
from processing.py's NaN-preserving merge.

Needs both merged tables, so run Stages 1-4 for split="train" and
split="test" first.

Usage
-----
    python prepare_features.py
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from config import AnalysisConfig, PARAM_COLS, config as default_config


# Per-band feature templates, matching merge_bands()'s output column names.

BAND_FEATURE_TEMPLATES: list[str] = ["mag_{band}", "{band}_a1", "{band}_a2"]

# SN-level features -- one value per SN, not per band.
BASE_FEATURE_COLS: list[str] = ["redshift"]


def select_available_bands(train_df: pd.DataFrame, bands: Sequence[str]) -> list[str]:
    """Bands to actually use as features, decided from the TRAINING set.

    A band is kept if its mag_{band} column has at least one real
    (non-NaN) value anywhere in train_df; a band that's entirely NaN
    across every training row is dropped entirely -- a fully-empty column
    carries no signal, and zero-filling it would be indistinguishable
    from "every SN measured 0" rather than "this band was never fit."
    """
    available = []
    for band in bands:
        col = f"mag_{band}"
        if col in train_df.columns and train_df[col].notna().any():
            available.append(band)
    return available


def build_feature_columns(bands: Sequence[str]) -> list[str]:
    """The full, ordered list of feature column names for a given band set."""
    cols = list(BASE_FEATURE_COLS)
    for band in bands:
        for template in BAND_FEATURE_TEMPLATES:
            cols.append(template.format(band=band))
    return cols


def prepare_features(df: pd.DataFrame, bands: Sequence[str]) -> tuple[pd.DataFrame, list[str]]:
    """Zero-fills any remaining NaNs (SN-specific missing bands, as opposed
    to a band missing across the whole training set -- see
    select_available_bands) in the given bands' feature columns. 


    Returns (df with those columns zero-filled, the feature column list).
    """
    feature_cols = build_feature_columns(bands)
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Expected feature columns not found in dataframe: {missing_cols}")

    df = df.copy()
    df[feature_cols] = df[feature_cols].fillna(0.0)
    return df, feature_cols


def to_wide(df: pd.DataFrame, feature_cols: Sequence[str], param_cols: Sequence[str],
            n_sn: int) -> tuple[pd.DataFrame, list[str]]:
    """One row per data set (cosmo_id): its n_sn SNe side by side, sorted by
    redshift. Column names follow "{feature}_{i}": redshift_0, mag_x_0, ...

    Returns (wide df, wide feature column list).
    """
    wide_cols = [f"{c}_{i}" for i in range(n_sn) for c in feature_cols]

    rows = []
    for cosmo_id, group in df.groupby("cosmo_id"):
        group = group.sort_values("redshift")
        values = group[list(feature_cols)].to_numpy().ravel()
        row = {"cosmo_id": cosmo_id}
        row.update({p: group[p].iloc[0] for p in param_cols})
        row.update(zip(wide_cols, values))
        rows.append(row)

    return pd.DataFrame(rows), wide_cols


def build_train_test_features(
    train_csv: str,
    test_csv: str,
    bands: Sequence[str],
    param_cols: Sequence[str],
    train_features_csv: str,
    test_features_csv: str,
    n_sn_per_set: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """End-to-end: read a train and a test merged CSV, decide which bands
    to use (from train only, via select_available_bands), zero-fill both
    into matching fixed-width feature matrices (via prepare_features), and
    reshape each into one row per data set (via to_wide), and write each
    to disk.

    Train and test are separate simulation runs, not a split of one file
    -- e.g. train drawn from a broad prior, test drawn at one "true"
    cosmology for evaluation.

    Returns (train_wide, test_wide, wide_cols, available_bands), where
    train_wide/test_wide also carry cosmo_id and param_cols alongside the
    feature columns, for traceability.
    """
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    available_bands = select_available_bands(train_df, bands)
    dropped_bands = [b for b in bands if b not in available_bands]
    print(f"Bands with data in training set: {available_bands}")
    if dropped_bands:
        print(f"Dropping bands with no data anywhere in training set: {dropped_bands}")

    train_df, feature_cols = prepare_features(train_df, available_bands)
    test_df, _ = prepare_features(test_df, available_bands)  # same bands as train, by construction

    train_wide, wide_cols = to_wide(train_df, feature_cols, param_cols, n_sn_per_set)
    test_wide, _ = to_wide(test_df, feature_cols, param_cols, n_sn_per_set)

    train_wide.to_csv(train_features_csv, index=False)
    test_wide.to_csv(test_features_csv, index=False)
    print(f"Wrote feature matrices ({len(train_wide)} train sets, {len(test_wide)} test sets, "
          f"{len(wide_cols)} features each) -> {train_features_csv}, {test_features_csv}")

    return train_wide, test_wide, wide_cols, available_bands


def main():
    train_paths = AnalysisConfig(split="train")
    test_paths = AnalysisConfig(split="test")
    build_train_test_features(
        train_paths.processing.merged_csv,
        test_paths.processing.merged_csv,
        list(default_config.fitting.bands),
        PARAM_COLS,
        train_paths.prepare_features.features_csv,
        test_paths.prepare_features.features_csv,
        default_config.fitting.max_sn_per_cosmo,
    )


if __name__ == "__main__":
    main()
