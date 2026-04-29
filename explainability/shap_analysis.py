"""
SHAP and LIME explainability analysis for trained IDS models.
"""

import os
import sys
import json
import numpy as np
import torch
import shap
from lime import lime_tabular
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mlp_baseline import MLP_IDS
from models.lstm_model import LSTM_IDS
from models.cnn1d_model import CNN1D_IDS
from data.preprocess import load_preprocessed, FEATURE_NAMES

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device('cpu')  # SHAP works best on CPU for these models
RESULTS_DIR = 'results'
MODELS_DIR = 'saved_models'
N_BACKGROUND = 100   # Background samples for SHAP
N_EXPLAIN = 200      # Samples to explain


def load_model(model_class, model_name, num_classes=2):
    """Load trained model."""
    model = model_class(in_dim=41, num_classes=num_classes)
    model.load_state_dict(torch.load(
        os.path.join(MODELS_DIR, f'{model_name}_best.pt'),
        weights_only=True, map_location='cpu'
    ))
    model.eval()
    return model


def model_predict_fn(model, X):
    """Wrapper for LIME compatibility — returns probabilities."""
    with torch.no_grad():
        tensor = torch.FloatTensor(X).to(DEVICE)
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).numpy()
    return probs


def run_shap_analysis(model, model_name, X_train, X_test, class_names):
    """Compute SHAP values using KernelExplainer (model-agnostic)."""
    print(f"\n--- SHAP Analysis: {model_name} ---")
    
    # Background data
    bg_idx = np.random.choice(len(X_train), N_BACKGROUND, replace=False)
    background = X_train[bg_idx]
    
    # Samples to explain
    exp_idx = np.random.choice(len(X_test), N_EXPLAIN, replace=False)
    X_explain = X_test[exp_idx]
    
    # Create predict function
    def predict_fn(X):
        return model_predict_fn(model, X)
    
    # KernelExplainer (model-agnostic, works for all architectures)
    explainer = shap.KernelExplainer(predict_fn, background)
    
    print(f"  Computing SHAP values for {N_EXPLAIN} samples...")
    shap_values = explainer.shap_values(X_explain, nsamples=200, silent=True)
    
    # --- Global Feature Importance ---
    mean_abs_shap = np.abs(shap_values[0]).mean(axis=0)
    feature_importance = list(zip(FEATURE_NAMES, mean_abs_shap))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n  Top 10 features (by mean |SHAP| for {class_names[0]}):")
    for fname, imp in feature_importance[:10]:
        print(f"    {fname:35s}: {imp:.4f}")
    
    # --- Save SHAP summary plot ---
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values[0], X_explain, feature_names=FEATURE_NAMES,
                      show=False, max_display=15)
    plt.title(f'SHAP Feature Importance - {model_name.upper()} ({class_names[0]})')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'shap_summary_{model_name}.png'), dpi=150)
    plt.close()
    
    # --- Save bar plot ---
    plt.figure(figsize=(10, 6))
    top_features = feature_importance[:15]
    names = [f[0] for f in top_features]
    values = [f[1] for f in top_features]
    plt.barh(range(len(names)), values[::-1], color='steelblue')
    plt.yticks(range(len(names)), names[::-1])
    plt.xlabel('Mean |SHAP value|')
    plt.title(f'Top 15 Features - {model_name.upper()}')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'shap_bar_{model_name}.png'), dpi=150)
    plt.close()
    
    return shap_values, feature_importance, exp_idx


def run_lime_analysis(model, model_name, X_train, X_test, class_names, n_instances=20):
    """Run LIME on a subset of test samples."""
    print(f"\n--- LIME Analysis: {model_name} ---")
    
    def predict_fn(X):
        return model_predict_fn(model, X)
    
    explainer = lime_tabular.LimeTabularExplainer(
        X_train,
        feature_names=FEATURE_NAMES,
        class_names=class_names,
        discretize_continuous=True,
        random_state=SEED
    )
    
    lime_results = []
    all_top_features = {}
    
    idx_to_explain = np.random.choice(len(X_test), n_instances, replace=False)
    
    for i, idx in enumerate(idx_to_explain):
        sample = X_test[idx]
        exp = explainer.explain_instance(sample, predict_fn, num_features=10, top_labels=1)
        
        pred_class = np.argmax(predict_fn(sample.reshape(1, -1)))
        feature_weights = exp.as_list(label=pred_class)
        
        lime_results.append({
            'sample_idx': int(idx),
            'predicted_class': class_names[pred_class],
            'top_features': [(fw[0], float(fw[1])) for fw in feature_weights]
        })
        
        for fw in feature_weights:
            fname = fw[0].split(' ')[0]
            all_top_features[fname] = all_top_features.get(fname, 0) + 1
        
        if (i + 1) % 5 == 0:
            print(f"  Explained {i+1}/{n_instances} samples")
    
    sorted_features = sorted(all_top_features.items(), key=lambda x: x[1], reverse=True)
    print(f"\n  Top features by LIME frequency ({n_instances} samples):")
    for fname, count in sorted_features[:10]:
        print(f"    {fname:35s}: appears in {count}/{n_instances} explanations")
    
    # Save LIME feature frequency plot
    plt.figure(figsize=(10, 6))
    top_lime = sorted_features[:15]
    names = [f[0] for f in top_lime]
    counts = [f[1] for f in top_lime]
    plt.barh(range(len(names)), counts[::-1], color='coral')
    plt.yticks(range(len(names)), names[::-1])
    plt.xlabel(f'Frequency in top-10 (out of {n_instances} samples)')
    plt.title(f'LIME Top Features - {model_name.upper()}')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'lime_frequency_{model_name}.png'), dpi=150)
    plt.close()
    
    return lime_results, sorted_features


def compare_shap_lime(shap_importance, lime_frequency, model_name):
    """Compare SHAP vs LIME feature rankings."""
    from scipy.stats import spearmanr
    
    shap_features = {f: i for i, (f, _) in enumerate(shap_importance[:20])}
    lime_features = {f: i for i, (f, _) in enumerate(lime_frequency[:20])}
    
    common = set(shap_features.keys()) & set(lime_features.keys())
    
    if len(common) >= 5:
        shap_ranks = [shap_features[f] for f in common]
        lime_ranks = [lime_features[f] for f in common]
        corr, p_value = spearmanr(shap_ranks, lime_ranks)
        print(f"\n  SHAP vs LIME rank correlation ({model_name}):")
        print(f"    Common features in top-20: {len(common)}")
        print(f"    Spearman correlation: {corr:.4f} (p={p_value:.4f})")
        return {'spearman_corr': float(corr), 'p_value': float(p_value), 
                'n_common': len(common)}
    else:
        print(f"  Too few common features ({len(common)}) for correlation")
        return {'n_common': len(common)}


def main():
    X_train, X_test, y_train, y_test, le, scaler, meta = load_preprocessed()
    class_names = meta['class_names']
    
    print(f"Data loaded: {X_train.shape} train, {X_test.shape} test")
    print(f"Classes: {class_names}")
    
    all_xai_results = {}
    
    models_to_analyze = [
        ('mlp', MLP_IDS),
        ('lstm', LSTM_IDS),
        ('cnn1d', CNN1D_IDS),
    ]
    
    for model_name, model_class in models_to_analyze:
        model_path = os.path.join(MODELS_DIR, f'{model_name}_best.pt')
        if not os.path.exists(model_path):
            print(f"  Skipping {model_name} - no saved model found")
            continue
        
        model = load_model(model_class, model_name, num_classes=len(class_names))
        
        shap_vals, shap_importance, exp_idx = run_shap_analysis(
            model, model_name, X_train, X_test, class_names
        )
        
        lime_results, lime_frequency = run_lime_analysis(
            model, model_name, X_train, X_test, class_names, n_instances=30
        )
        
        comparison = compare_shap_lime(shap_importance, lime_frequency, model_name)
        
        all_xai_results[model_name] = {
            'shap_top_features': [(f, float(v)) for f, v in shap_importance[:15]],
            'lime_top_features': [(f, int(v)) for f, v in lime_frequency[:15]],
            'shap_vs_lime': comparison,
        }
    
    with open(os.path.join(RESULTS_DIR, 'explainability_results.json'), 'w') as f:
        json.dump(all_xai_results, f, indent=2)
    
    print(f"\nExplainability analysis complete!")
    print(f"Results saved to {RESULTS_DIR}/")


if __name__ == '__main__':
    main()
