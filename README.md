# Identifying Depression Subtypes Using Gene Expression
### A Multi-Phase Analytical Framework for Classification, Subtyping, and Systems-Level Discovery
**Author:** Vybhav Chaturvedi  
**Roll No.:** 25BM6JP60  

---

## Project Overview

This repository contains a complete analytical workflow for understanding molecular heterogeneity in Major Depressive Disorder (MDD) using postmortem amygdala gene expression (GSE54564 dataset).

The project consists of three sequential phases, each implemented in a dedicated Jupyter Notebook:

1. **Phase 1 – Baseline Supervised Classification**  
2. **Phase 2 – Nested Validation & Advanced ML**  
3. **Phase 3 – Unsupervised Subtype Discovery & Network Biology**

The workflow demonstrates that binary classification fails due to diagnostic heterogeneity, revealing a biologically meaningful **Neuro-Immune-Vascular subtype** within MDD.

---

## Repository Structure

```
C:.
│   01_Phase1_Baseline_Classification.ipynb
│   02_Phase2_Advanced_Classification.ipynb
│   03_Phase3_Unsupervised Analysis.ipynb
│   requirements.txt
│
├───data
│       GSE54564_non-normalized.txt
│       GSE54564_series_matrix.txt
│       NCBI_Depression.bgx
│
├───output
│   ├───Phase_1
│   ├───Phase_2
│   └───Phase_3
│
└───output/cytoscape_session
```

---

## Installation

Requires **Python 3.9+**.

```bash
pip install -r requirements.txt
```

Or using conda:

```bash
conda create -n depression_env python=3.9
conda activate depression_env
pip install -r requirements.txt
```

---

## How to Run the Project

### 1️⃣ Phase 1 — Baseline Classification  
`01_Phase1_Baseline_Classification.ipynb`

- Loads raw data  
- Performs preprocessing  
- Trains 7 ML models  
- Generates EDA + ROC + confusion matrices  
- Saves outputs in `output/Phase_1/`

---

### 2️⃣ Phase 2 — Nested LOO-CV & Advanced ML  
`02_Phase2_Advanced_Classification.ipynb`

- Implements **Nested Leave-One-Out CV**  
- Evaluates 6 advanced pipelines  
- Computes unbiased performance estimates  
- Saves results to `output/Phase_2/`

---

### 3️⃣ Phase 3 — Unsupervised Discovery & Systems Biology  
`03_Phase3_Unsupervised Analysis.ipynb`

- Consensus clustering (500 bootstraps)  
- Subtype identification  
- Cell-type deconvolution  
- WGCNA network construction  
- Cytoscape exports in `output/cytoscape_session/`

---

## Reproducibility

Run all three notebooks **in order**.

---

## Key Findings

- **Classification fails** (AUC ~ 0.49), proving diagnostic heterogeneity  
- **4 molecular subtypes** discovered  
- **Subtype S3 enriched for MDD (86%)**  
- **Inhibitory neuron depletion** (d = –0.77)  
- **Inflammatory module (M2) upregulated**  
- **Neurovascular module (M8) downregulated**  
- **MED8 emerges as a key regulatory hub**

---

## Contact

For replication or questions:  
**Vybhav Chaturvedi**  

- ISI Kolkata (25BM6JP60)
- IIT Kharagpur
- IIM Calcutta
- TIET
