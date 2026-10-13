"""
processing.py

Stage 4 of the analysis pipeline: merge Stage 3's per-(SN, band) fit results
(fitting.raw_csv) into one row per SN, with each band's fitted values as
its own set of columns. Generalized to any band list (cfg.fitting.bands).
merge_bands()/run() produce that NaN-preserving table -- the
"which band failed for which SN" distinction is real information, kept
intact here for any consumer that wants it.

Turning this table into NaN-free SBI inputs is a separate step: see
prepare_features.py.

Usage
-----
    python processing.py
"""

from __future__ import annotations

import pandas as pd

from config import FittingConfig, PARAM_COLS, config as default_config


# Per-band fields pulled from Stage 3's raw output, and the output column
# name template each maps to (formatted with the band name).
PER_BAND_FIELDS = {
    "pk_mag": "mag_{band}",
    "a1": "{band}_a1",
    "a2": "{band}_a2",
    "std_mag": "err_mag_{band}",
    "std_a1": "err_a1_{band}",
    "std_a2": "err_a2_{band}",
}

# Columns that describe the SN itself (not any particular band) -- taken
# once per SN, since they're identical across that SN's band rows.
SN_LEVEL_COLS = list(PARAM_COLS) + ["redshift"]


def merge_bands(raw: pd.DataFrame, bands: tuple[str, ...]) -> pd.DataFrame:
    """One row per (cosmo_id, sn_id), with each band's fitted values as its
    own set of columns. A band missing for a given SN (e.g. it failed
    quality cuts in Stage 3) leaves that band's columns as NaN for that row.
    """
    raw = raw.copy()
    raw["chi_dof"] = raw["chi_2"] / raw["dof"]

    # SN-level columns: one value per SN, taken from whichever row happens
    # to be first (redshift/omega_m/omega_lambda don't vary by band).
    base = raw.drop_duplicates(subset=["cosmo_id", "sn_id"]).set_index(["cosmo_id", "sn_id"])
    merged = base[SN_LEVEL_COLS].copy()

    for band in bands:
        band_rows = raw[raw["filter"] == band].set_index(["cosmo_id", "sn_id"])
        for src_col, name_template in PER_BAND_FIELDS.items():
            out_col = name_template.format(band=band)
            # .reindex aligns onto merged's full SN index; SNe with no row
            # for this band automatically get NaN in every one of its columns.
            merged[out_col] = band_rows[src_col].reindex(merged.index)
        merged[f"chi_dof_{band}"] = band_rows["chi_dof"].reindex(merged.index)

    return merged.reset_index()


def run(cfg: FittingConfig, raw_csv: str, merged_csv: str) -> pd.DataFrame:
    raw = pd.read_csv(raw_csv)
    merged = merge_bands(raw, cfg.bands)

    merged.to_csv(merged_csv, index=False)
    print(f"Done. Merged {len(merged)} SNe across {len(cfg.bands)} bands -> {merged_csv}")
    return merged


def main():
    run(default_config.fitting, default_config.fitting.raw_csv, default_config.processing.merged_csv)


if __name__ == "__main__":
    main()
