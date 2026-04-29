# Project Plan — Explainable IDS

## 1. Problem Statement

Intrusion Detection Systems (IDS) powered by deep learning achieve high detection rates but operate as black boxes. In security-critical environments, analysts need to understand **why** a connection is flagged as malicious — not just that it is. This project addresses three key questions:

1. **Can we explain IDS decisions** using post-hoc methods (SHAP, LIME)?
2. **Are these explanations stable** — do similar inputs produce similar explanations?
3. **What are the security risks** of making model decisions interpretable?

## 2. Methodology

### Phase 1: Data Understanding & Preprocessing
- Load NSL-KDD dataset (41 features, binary + 5-class labels)
- Encode 3 categorical features (protocol_type, service, flag) via LabelEncoder
- Normalize all features to [0,1] via MinMaxScaler
- Analyze class distribution and document imbalance (especially U2R: ~52 samples, R2L: ~995)

### Phase 2: Baseline Model Training
- **Primary model**: MLP (256→128→64→num_classes) with BatchNorm and Dropout
- **Comparison models**: LSTM (2-layer, hidden=64) and 1D-CNN (Conv64→Conv128→AvgPool→FC)
- Training: Adam optimizer, lr=1e-3, weight_decay=1e-4, 50 epochs
- Evaluation: Per-class Precision/Recall/F1, Weighted F1, PR-AUC, Confusion Matrix

### Phase 3: Explainability Analysis
- **SHAP**: KernelExplainer (model-agnostic) — compute per-feature attributions for each class
  - Global summary plots (feature importance rankings)
  - Local force plots (individual predictions)
  - Class-specific analysis (which features drive anomaly detection)
- **LIME**: LimeTabularExplainer
  - Per-instance explanations with top-10 features
  - Compare LIME vs SHAP feature rankings

### Phase 4: Explanation Stability Evaluation
- **Perturbation stability (SENS_MAX)**: Add ε-bounded noise (ε=0.01, 0.03, 0.05), measure max attribution shift
- **LIME stochastic stability**: Run LIME 20 times per sample with different seeds, compute pairwise Spearman rank correlation
- **Faithfulness**: Mask top-k features identified by SHAP/LIME, measure prediction drop (higher drop = more faithful)
- **Threshold**: PCC > 0.6 = stable (per SAFARI framework, Huang et al. 2022)

### Phase 5: Security Implications Analysis
- Can an attacker use SHAP output to identify which features to manipulate for evasion?
- Is LIME's stochasticity a security concern (inconsistent analyst decisions)?
- Risk of explanation manipulation attacks (backdoored models with clean explanations)

## 3. Experimental Design (≥3 variations required)

| Experiment | Description | Metric |
|------------|-------------|--------|
| **Baseline** | MLP on binary NSL-KDD | Weighted F1, PR-AUC |
| **Variation 1** | MLP on 5-class NSL-KDD | Per-class F1 |
| **Variation 2** | LSTM on binary NSL-KDD | Weighted F1 (compare to MLP) |
| **Variation 3** | 1D-CNN on binary NSL-KDD | Weighted F1 (compare to MLP) |
| **XAI Comparison** | SHAP vs LIME feature rankings | Rank correlation, faithfulness |
| **Stability** | Explanation stability across ε values | SENS_MAX, PCC |

## 4. Timeline

| Phase | Duration | Status |
|-------|----------|--------|
| Data preprocessing | 1 day | ✅ Done |
| Baseline training | 1 day | 🔄 In Progress |
| Explainability | 2 days | Pending |
| Stability eval | 1 day | Pending |
| Security analysis | 1 day | Pending |
| Report writing | 2 days | Pending |

## 5. Deliverables

1. **Explanation Analysis** — SHAP/LIME visualizations with interpretation
2. **Security Report** — Adversarial risks of exposing explanations
3. **Code + README** — Fully reproducible pipeline
4. **Report** (max 10 pages PDF) — All design choices justified
