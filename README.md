# NovoFlow

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Release](https://img.shields.io/badge/Release-MassIVE--KB%20Model-brightgreen.svg)](https://github.com/joelinator/JoelResearch/releases/tag/v0.3.0)

**NovoFlow** is a tool for *de novo* peptide sequencing from tandem mass spectrometry (MS/MS) data. It uses continuous-time discrete flow matching (DFM) coupled with dynamic programming knapsack reachability filtering, fragment-ladder scoring, and calibrated False Discovery Rate (FDR) estimation.

> **Note**: This project is inspired by and builds upon research developed during my Master's thesis: [JoelResearch](https://github.com/joelinator/JoelResearch).

By default, NovoFlow automatically fetches the official **MassIVE-KB pretrained checkpoint** (`dfm_massive_kb.ckpt`) from the [JoelResearch GitHub Releases](https://github.com/joelinator/JoelResearch/releases/tag/v0.3.0).

---

## Installation

```bash
git clone https://github.com/joelinator/novoflow.git
cd novoflow
pip install -e .[io]
```

---

## Quickstart

### 1. Command-Line Interface (CLI)

NovoFlow can be run directly from the terminal on `.mgf` or `.mzML` files:

```bash
# Display environment and GPU details
novoflow info

# Sequence an MS/MS run (automatically downloads MassIVE-KB checkpoint on first run)
novoflow predict -i sample.mgf -o results.tsv

# Sequence on GPU with 1% FDR filtering and mzTab export
novoflow predict -i sample.mzML -o results.mzTab --device cuda --filter-fdr
```

### 2. Python API

```python
from novoflow import NovoFlowPipeline
from novoflow.io import read_mgf, export_identifications

# Initialize pipeline (loads official MassIVE-KB model by default)
pipeline = NovoFlowPipeline(device="cuda")

# Read spectra and sequence
spectra = list(read_mgf("sample.mgf"))
results = pipeline.predict_batch(spectra)

# Inspect identifications
for psm in results:
    print(f"Scan: {psm['title']} | Sequence: {psm['sequence']} | q-value: {psm['q_value']:.4f}")

# Export results (TSV, CSV, or mzTab)
export_identifications(results, "results.tsv", format="tsv")
```

---

## What NovoFlow Does

1. **Non-Autoregressive Generation**: Predicts peptide sequences iteratively via continuous-time discrete flow matching, decoding parallel positions simultaneously.
2. **Knapsack Reachability**: Restricts candidate amino acids at every step to combinations that can physically sum to the experimental precursor mass.
3. **Multi-Candidate Beam Exploration**: Explores the top candidate peptide lengths and flow trajectories, scoring candidates against theoretical $b$ and $y$ fragment ion ladders.
4. **Calibrated FDR / PEP Filtering**: Computes Posterior Error Probabilities (PEP) and monotonic $q$-values directly, reporting identifications at target FDR thresholds without requiring database decoys.
5. **Standard File Formats**: Reads `.mgf` and `.mzML` runs and outputs standard TSV, CSV, or HUPO-PSI compliant `mzTab` files.

---

## Pretrained Model

By default, NovoFlow downloads and caches the MassIVE-KB pretrained model:

- **Source**: [joelinator/JoelResearch (Release v0.3.0)](https://github.com/joelinator/JoelResearch/releases/tag/v0.3.0)
- **File**: `dfm_massive_kb.ckpt`
- **Cache Location**: `~/.cache/novoflow/dfm_massive_kb.ckpt`

To use a custom checkpoint instead, pass `--checkpoint /path/to/model.ckpt` in the CLI or `NovoFlowPipeline(checkpoint_path="/path/to/model.ckpt")` in Python.

---

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
