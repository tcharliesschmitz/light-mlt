# 🧩 light-mlt

[![PyPI version](https://img.shields.io/pypi/v/light-mlt.svg)](https://pypi.org/project/light-mlt/)
[![Python versions](https://img.shields.io/pypi/pyversions/light-mlt.svg)](https://pypi.org/project/light-mlt/)
[![License](https://img.shields.io/pypi/l/light-mlt.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/github/actions/workflow/status/tcharliesschmitz/light-mlt/tests.yml?branch=main)](https://github.com/tcharliesschmitz/light-mlt/actions)
[![Downloads](https://static.pepy.tech/personalized-badge/light-mlt?period=total&units=international_system&left_color=gray&right_color=blue&left_text=Downloads)](https://pepy.tech/project/light-mlt)

**Lightweight preprocessing and reversible Modular Linear Tokenization (MLT) utilities for categorical and continuous data.**

---

## ✨ Overview

`light-mlt` is a lightweight Python package that implements **Modular Linear Tokenization (MLT)** — a deterministic and fully reversible method for encoding high-cardinality categorical identifiers into compact numerical vectors.

Unlike hashing or one-hot encodings, **MLT guarantees bijective mappings**, offers **explicit control of dimensionality**, and integrates seamlessly with **machine learning pipelines**.

It was developed as part of applied research on scalable tokenization and efficient preprocessing for tabular and recommendation systems.

---

## 🚀 Installation

```bash
pip install light-mlt
