# Explainable Intrusion Detection System (X-IDS)

**ICCN-INE2 Deep Learning Project — Project 5: Explainable IDS**

## Project Overview

This project builds an Intrusion Detection System using deep learning on the NSL-KDD dataset, then applies post-hoc explainability methods (SHAP, LIME) to make decisions interpretable. We evaluate explanation stability and analyze security implications of exposing model explanations.

## Core Research Question

> *Can we make IDS decisions interpretable without compromising detection performance, and are these explanations stable enough to be trusted in security-critical settings?*

## Repository Structure

```
.
├── README.md                          # This file
├── docs/
│   ├── project_plan.md                # Detailed project plan & methodology
│   ├── threat_model.md                # Threat model document
│   └── architecture.md                # Model architecture & design choices
├── data/
│   └── preprocess.py                  # Data loading & preprocessing pipeline
├── models/
│   ├── mlp_baseline.py                # MLP baseline model
│   ├── lstm_model.py                  # LSTM variant
│   └── cnn1d_model.py                 # 1D-CNN variant
├── explainability/
│   ├── shap_analysis.py               # SHAP explanations
│   ├── lime_analysis.py               # LIME explanations
│   └── stability_eval.py             # Explanation stability evaluation
├── experiments/
│   ├── train_baseline.py              # Training script
│   ├── run_explainability.py          # Run all XAI methods
│   └── run_stability.py              # Stability evaluation experiments
├── results/                           # Generated results (figures, metrics)
├── requirements.txt                   # Dependencies
└── reproduce.sh                       # One-command reproducibility script
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Reproduce all experiments
bash reproduce.sh

# Or run step by step:
python data/preprocess.py              # Download & preprocess NSL-KDD
python experiments/train_baseline.py   # Train 3 models (MLP, LSTM, CNN)
python explainability/shap_analysis.py # SHAP + LIME analysis
python explainability/stability_eval.py # Stability evaluation
```

## Dataset

**NSL-KDD** (Network Security Laboratory - KDD) — an improved version of KDD Cup 99.
- Source: [UNB Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/nsl.html)
- HF Hub: [`Mireu-Lab/NSL-KDD`](https://huggingface.co/datasets/Mireu-Lab/NSL-KDD)
- Train: 151,165 records | Test: 34,394 records
- 41 features (3 categorical + 38 numerical)
- Binary classification: Normal vs Anomaly
- 5-class: Normal, DoS, Probe, R2L, U2R

## Models

| Model | Architecture | Parameters |
|-------|-------------|------------|
| MLP | 41→256→128→64→2 with BatchNorm + Dropout | ~50K |
| LSTM | 41-step sequence → 2-layer LSTM(64) → FC(2) | ~35K |
| 1D-CNN | Conv1d(64)→Conv1d(128)→AvgPool→FC(2) | ~45K |

## Explainability Methods

- **SHAP** (SHapley Additive exPlanations): KernelExplainer (model-agnostic)
- **LIME** (Local Interpretable Model-agnostic Explanations): Tabular explainer with perturbation sampling

## Evaluation Metrics

- **Classification**: Precision, Recall, F1-Score (per-class + weighted), PR-AUC, ROC-AUC
- **Explanation Quality**: Faithfulness (feature masking), Sensitivity (SENS_MAX), Stability (PCC across perturbations)

## Reproducibility

- Random seed: 42 (fixed across all experiments)
- Python 3.10+ | PyTorch 2.x | scikit-learn 1.x
- All preprocessing steps documented
- Commands in `reproduce.sh`

## References

1. Tavallaee et al. (2009). *A Detailed Analysis of the KDD CUP 99 Data Set.* IEEE Symposium on CISDA.
2. Lundberg & Lee (2017). *A Unified Approach to Interpreting Model Predictions.* NeurIPS.
3. Ribeiro et al. (2016). *"Why Should I Trust You?": Explaining the Predictions of Any Classifier.* KDD.
4. Huang et al. (2022). *SAFARI: Versatile and Efficient Evaluations for Robustness of Interpretability.* ICCV.

## Author

ICCN-INE2 Student Project
