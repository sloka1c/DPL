# Model Architecture & Design Choices

## 1. Design Philosophy

The project requires **lightweight models** (max 2h training per experiment). We prioritize:
- Simplicity over complexity — fewer layers, interpretable capacity
- Reproducibility — fixed seeds, deterministic operations
- Fair comparison — same preprocessing, same train/test split across all models

## 2. Preprocessing Pipeline

```
Raw NSL-KDD (41 features + label)
    │
    ├─ Categorical encoding: LabelEncoder for protocol_type (3), service (70), flag (11)
    ├─ Feature scaling: MinMaxScaler [0, 1] on all 41 features
    ├─ Label encoding: Binary (anomaly=0, normal=1)
    │
    └─ Output: X_train (151165, 41), X_test (34394, 41), float32
```

**Why MinMaxScaler?** Network features have vastly different ranges (src_bytes: 0-1.3B vs. serror_rate: 0-1). Scaling to [0,1] prevents large-valued features from dominating gradient updates and makes perturbation-based explainability (ε-bounded noise) meaningful.

**Why LabelEncoder (not OneHot)?** OneHot would expand 3 categorical features to 84 columns. This makes SHAP/LIME explanations harder to interpret (84 binary features vs 41 semantic features). LabelEncoder preserves the original feature space for cleaner explanations.

## 3. Model Architectures

### 3.1 MLP (Primary Baseline)

```
Input (41) → Linear(256) → BatchNorm → ReLU → Dropout(0.3)
          → Linear(128) → BatchNorm → ReLU → Dropout(0.2)
          → Linear(64) → ReLU
          → Linear(num_classes)
```

**Parameters**: ~50K
**Justification**:
- 3 hidden layers with decreasing width is standard for tabular classification
- BatchNorm stabilizes training, enables higher learning rates
- Dropout (0.3→0.2) regularizes; heavier in early layers where more parameters
- No final activation — CrossEntropyLoss includes LogSoftmax

### 3.2 LSTM (Temporal Variant)

```
Input (41) → reshape to (41, 1) → LSTM(hidden=64, layers=2, dropout=0.2)
          → take last hidden state → Linear(num_classes)
```

**Parameters**: ~35K
**Justification**:
- Treats 41 features as a sequence — captures inter-feature dependencies
- 2 layers with 64 hidden units is minimal while allowing feature interaction
- LSTM processes features in order: basic→content→time-based→host-based
- This ordering has semantic meaning in NSL-KDD (groups of related features)

### 3.3 1D-CNN (Spatial Variant)

```
Input (41) → reshape to (1, 41) → Conv1d(64, k=3, pad=1) → ReLU
          → Conv1d(128, k=3, pad=1) → ReLU → AdaptiveAvgPool1d(8)
          → Flatten → Linear(64) → ReLU → Linear(num_classes)
```

**Parameters**: ~45K
**Justification**:
- 1D convolutions learn local feature patterns (neighboring features)
- Kernel size 3 captures triplets of features
- AdaptiveAvgPool compresses to fixed size regardless of input length
- Useful for detecting patterns in rate-based features (contiguous block)

## 4. Training Configuration

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Optimizer | Adam | Standard for neural networks; adaptive lr per parameter |
| Learning rate | 1e-3 | Default Adam lr; works well for tabular tasks |
| Weight decay | 1e-4 | Light L2 regularization prevents overfitting |
| Batch size | 256 | Good balance of speed and gradient stability |
| Epochs | 50 | Sufficient for convergence on NSL-KDD (~151K samples) |
| Loss | CrossEntropyLoss | Standard for multi-class; includes class weights for imbalance |
| Class weights | Inverse frequency | Addresses class imbalance between normal (53%) and anomaly (47%) |
| Seed | 42 | Fixed for reproducibility |

## 5. Why These Models for Explainability

| Model | SHAP Method | Speed | Explanation Quality |
|-------|-------------|-------|-------------------|
| MLP | KernelExplainer | Medium | Clean, model-agnostic attributions |
| LSTM | KernelExplainer | Medium | Sequential attributions may differ |
| 1D-CNN | KernelExplainer | Medium | Convolutional attributions capture local patterns |

All three use **KernelExplainer** (model-agnostic SHAP), enabling:
- Direct comparison of feature attributions across architectures
- Analysis of whether model architecture affects explanation stability
- Consistent methodology across all models

## 6. Expected Baseline Performance

Based on published NSL-KDD benchmarks (Tavallaee et al., Revathi & Malathi 2013):

| Model | Binary Accuracy | Binary Weighted F1 |
|-------|----------------|---------------------|
| MLP | 78-85% | 78-83% |
| LSTM | 76-82% | 75-80% |
| 1D-CNN | 77-83% | 76-81% |

**Known challenge**: Test set has more anomaly (65%) than train (47%) — distribution shift tests generalization.
