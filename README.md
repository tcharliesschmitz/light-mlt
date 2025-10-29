
[![PyPI version](https://img.shields.io/pypi/v/light-mlt?color=orange)](https://pypi.org/project/light-mlt/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![GitHub](https://img.shields.io/badge/github-light--mlt-black?logo=github)](https://github.com/tcharliesschmitz/light-mlt)



# 🧩 Module: `light_mlt.py`

Lightweight, reproducible preprocessing pipeline for tabular datasets — integrating **categorical encoding via Modular Linear Tokenization (MLT)** and **continuous feature scaling**.  
Designed for **efficient fit/transform workflows** with full reversibility and schema persistence.

> 📘 **Reference**  
> Schmitz, T. (2025). *light-mlt: Modular Linear Tokenization for Scalable Categorical Encoding*.  
> DOI: [10.5281/zenodo.17467914](https://doi.org/10.5281/zenodo.17467914)

---

## 🔖 Key Features

- Deterministic preprocessing for categorical and continuous data  
- Append-only vocabulary management (`cats_map.pkl`)  
- Fully reversible categorical encoding via **MLT**  
- Continuous feature scaling with `StandardScaler`  
- Persistent schema for consistent transformations  
- Minimal dependencies: `numpy`, `pandas`, `scikit-learn`

---

## 📦 Artifacts

Each `fit()` operation generates or updates the following artifacts (default directory: `light_mlt_artifacts/`):

| File | Description |
|------|--------------|
| `schema.pkl` | Schema metadata (columns, types, MLT config) |
| `scaler.pkl` | Trained `StandardScaler` for continuous columns |
| `cats_map.pkl` | Append-only `{label → id}` mapping for categorical features |
| `mlt_params.pkl` | MLT parameters (`p`, `n`, `M`, `Minv`) per column |
| `preprocessed.csv` | Optional transformed dataset export |

---

## ⚙️ Core Function: `fit`

```python
from light_mlt import fit

fit(
    df,
    categorical_cols=["city", "vehicle"],
    continuous_cols=["age", "salary"],
    dir="light_mlt_artifacts/"
)
