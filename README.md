# FPCA-SBI

Simulation-based inference (SNPE) of cosmological parameters (Ω_m, Ω_Λ) from Type Ia supernova light curves compressed fitted Functional PCA (FPCA) [He et al. 2018].

## Pipeline

| Stage | Script | Output |
|---|---|---|
| 1. Simulate parameters | `simulate.py` | `data/{split}_params_theory.csv` |
| 2. Generate light curves | `lc_gen.py` | `data/lc_{split}/` (one CSV per SN) |
| 3. Fit FPCA templates | `lc_fit.py` | `data/{split}_fit_raw.csv` |
| 4. Merge bands | `processing.py` | `data/{split}_merged.csv` |
| 5. Build features | `prepare_features.py` | `data/{split}_features.csv` |
| 6. Train SBI | `single_run.py` | `figures/posterior_plot.png` |
| 7. Bias & uncertainty | `bias_unc.py` | `results/bias_unc.csv`, `figures/bias_unc.png` |

All settings live in `config.py`. `AnalysisConfig(split="train" | "test")` sets file paths for each split.

## Usage

Run stages 1–4 once for `split="train"` and once for `split="test"`, then stages 5–7. The full workflow is in `notebook.ipynb`.

## Requirements

```
numpy pandas scipy matplotlib torch sbi corner tqdm
```

FPCA template fits use `mpfit`. The fits also need `eigen_val.txt` (FPCA eigenvectors) and `mapping_coeff_original.csv` (per-band priors on the FPCA scores vs. redshift).
