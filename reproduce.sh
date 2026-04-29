#!/bin/bash
# Reproduce all experiments for Explainable IDS project
# Usage: bash reproduce.sh

set -e

echo "========================================"
echo "Explainable IDS — Full Reproduction"
echo "========================================"

# Install dependencies
echo "Step 0: Installing dependencies..."
pip install -r requirements.txt -q

# Step 1: Preprocess data
echo ""
echo "Step 1: Preprocessing NSL-KDD dataset..."
python data/preprocess.py

# Step 2: Train all models
echo ""
echo "Step 2: Training models (MLP, LSTM, 1D-CNN)..."
python experiments/train_baseline.py

# Step 3: Explainability analysis (SHAP + LIME)
echo ""
echo "Step 3: Running explainability analysis..."
python explainability/shap_analysis.py

# Step 4: Stability evaluation
echo ""
echo "Step 4: Running stability evaluation..."
python explainability/stability_eval.py

echo ""
echo "========================================"
echo "All experiments complete!"
echo "Results in: results/"
echo "Models in: saved_models/"
echo "========================================"
