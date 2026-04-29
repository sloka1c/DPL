"""
Explanation stability evaluation.
Measures robustness of SHAP/LIME explanations under input perturbation.
Based on SAFARI framework (Huang et al., 2022).
"""

import os
import sys
import json
import numpy as np
import torch
from scipy.stats import spearmanr, pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mlp_baseline import MLP_IDS
from data.preprocess import load_preprocessed, FEATURE_NAMES
import shap
from lime import lime_tabular

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device('cpu')
RESULTS_DIR = 'results'
MODELS_DIR = 'saved_models'


def model_predict_fn(model, X):
    """Predict probabilities."""
    with torch.no_grad():
        tensor = torch.FloatTensor(X).to(DEVICE)
        logits = model(tensor)
        return torch.softmax(logits, dim=1).numpy()


# ============================================================
# 1. SHAP Perturbation Stability (SENS_MAX)
# ============================================================
def compute_sens_max(explainer, sample, epsilon=0.01, n_perturbs=20):
    """
    SENS_MAX: Maximum attribution shift under epsilon-bounded perturbation.
    Lower = more stable explanations.
    
    From: SAFARI (Huang et al., 2022, ICCV)
    """
    rng = np.random.RandomState(SEED)
    
    base_shap = np.array(explainer.shap_values(sample.reshape(1, -1), 
                                                 nsamples=100, silent=True))
    if isinstance(base_shap, list):
        base_shap = base_shap[0].flatten()
    else:
        base_shap = base_shap.flatten()
    
    max_delta = 0
    for _ in range(n_perturbs):
        noise = rng.uniform(-epsilon, epsilon, sample.shape)
        perturbed = np.clip(sample + noise, 0, 1)
        
        perturbed_shap = np.array(explainer.shap_values(perturbed.reshape(1, -1),
                                                          nsamples=100, silent=True))
        if isinstance(perturbed_shap, list):
            perturbed_shap = perturbed_shap[0].flatten()
        else:
            perturbed_shap = perturbed_shap.flatten()
        
        delta = np.linalg.norm(perturbed_shap - base_shap, ord=2)
        max_delta = max(max_delta, delta)
    
    return max_delta, base_shap


def compute_shap_pcc(explainer, sample, epsilon=0.01, n_perturbs=10):
    """
    Pearson Correlation Coefficient between original and perturbed SHAP values.
    PCC > 0.6 = stable, PCC < 0.4 = unstable (SAFARI threshold).
    """
    rng = np.random.RandomState(SEED)
    
    base_shap = np.array(explainer.shap_values(sample.reshape(1, -1),
                                                 nsamples=100, silent=True))
    if isinstance(base_shap, list):
        base_shap = base_shap[0].flatten()
    else:
        base_shap = base_shap.flatten()
    
    pccs = []
    for _ in range(n_perturbs):
        noise = rng.uniform(-epsilon, epsilon, sample.shape)
        perturbed = np.clip(sample + noise, 0, 1)
        
        perturbed_shap = np.array(explainer.shap_values(perturbed.reshape(1, -1),
                                                          nsamples=100, silent=True))
        if isinstance(perturbed_shap, list):
            perturbed_shap = perturbed_shap[0].flatten()
        else:
            perturbed_shap = perturbed_shap.flatten()
        
        if np.std(base_shap) > 1e-8 and np.std(perturbed_shap) > 1e-8:
            pcc, _ = pearsonr(base_shap, perturbed_shap)
            pccs.append(pcc)
    
    return np.mean(pccs) if pccs else 0.0, np.std(pccs) if pccs else 0.0


# ============================================================
# 2. LIME Stochastic Stability
# ============================================================
def compute_lime_stability(X_train, predict_fn, sample, n_runs=15):
    """
    Measure LIME's inherent stochasticity by running with different seeds.
    Returns pairwise Spearman rank correlation of feature rankings.
    """
    weight_vectors = []
    
    for seed in range(n_runs):
        exp_obj = lime_tabular.LimeTabularExplainer(
            X_train,
            feature_names=FEATURE_NAMES,
            discretize_continuous=True,
            random_state=seed
        )
        exp = exp_obj.explain_instance(sample, predict_fn, num_features=len(FEATURE_NAMES))
        
        weight_dict = dict(exp.as_list())
        weights = np.zeros(len(FEATURE_NAMES))
        for key, val in weight_dict.items():
            for i, fname in enumerate(FEATURE_NAMES):
                if fname in key:
                    weights[i] = val
                    break
        weight_vectors.append(weights)
    
    corrs = []
    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            if np.std(weight_vectors[i]) > 1e-8 and np.std(weight_vectors[j]) > 1e-8:
                rho, _ = spearmanr(weight_vectors[i], weight_vectors[j])
                corrs.append(rho)
    
    return np.mean(corrs) if corrs else 0.0, np.std(corrs) if corrs else 0.0


# ============================================================
# 3. Faithfulness (Feature Masking)
# ============================================================
def compute_faithfulness(predict_fn, sample, shap_values, top_k=5):
    """
    Faithfulness: Remove top-k important features and measure prediction drop.
    Higher drop = more faithful explanations.
    """
    baseline_prob = predict_fn(sample.reshape(1, -1))[0]
    pred_class = np.argmax(baseline_prob)
    baseline_conf = baseline_prob[pred_class]
    
    top_k_idx = np.argsort(np.abs(shap_values))[-top_k:]
    masked = sample.copy()
    masked[top_k_idx] = 0.0
    
    masked_prob = predict_fn(masked.reshape(1, -1))[0]
    masked_conf = masked_prob[pred_class]
    
    confidence_drop = baseline_conf - masked_conf
    return float(confidence_drop)


# ============================================================
# Main stability evaluation
# ============================================================
def main():
    X_train, X_test, y_train, y_test, le, scaler, meta = load_preprocessed()
    class_names = meta['class_names']
    
    model = MLP_IDS(in_dim=41, num_classes=len(class_names))
    model.load_state_dict(torch.load(
        os.path.join(MODELS_DIR, 'mlp_best.pt'),
        weights_only=True, map_location='cpu'
    ))
    model.eval()
    
    def predict_fn(X):
        return model_predict_fn(model, X)
    
    bg_idx = np.random.choice(len(X_train), 100, replace=False)
    background = X_train[bg_idx]
    shap_explainer = shap.KernelExplainer(predict_fn, background)
    
    n_eval = 20
    eval_idx = np.random.choice(len(X_test), n_eval, replace=False)
    
    print("="*60)
    print("EXPLANATION STABILITY EVALUATION")
    print("="*60)
    
    epsilons = [0.01, 0.03, 0.05]
    
    stability_results = {
        'shap_sens_max': {},
        'shap_pcc': {},
        'lime_stability': {},
        'faithfulness': {},
    }
    
    # 1. SHAP Stability across epsilons
    for eps in epsilons:
        sens_maxes = []
        pccs_mean = []
        
        print(f"\n--- SHAP Stability (eps={eps}) ---")
        for i, idx in enumerate(eval_idx[:10]):
            sample = X_test[idx]
            
            sens_max, base_shap = compute_sens_max(shap_explainer, sample, epsilon=eps, n_perturbs=10)
            pcc_mean, pcc_std = compute_shap_pcc(shap_explainer, sample, epsilon=eps, n_perturbs=5)
            
            sens_maxes.append(sens_max)
            pccs_mean.append(pcc_mean)
            
            if (i + 1) % 5 == 0:
                print(f"  Sample {i+1}/10 | SENS_MAX={sens_max:.4f} | PCC={pcc_mean:.4f}")
        
        stability_results['shap_sens_max'][str(eps)] = {
            'mean': float(np.mean(sens_maxes)),
            'std': float(np.std(sens_maxes)),
            'max': float(np.max(sens_maxes)),
        }
        stability_results['shap_pcc'][str(eps)] = {
            'mean': float(np.mean(pccs_mean)),
            'std': float(np.std(pccs_mean)),
        }
        
        pcc_status = "STABLE" if np.mean(pccs_mean) > 0.6 else "UNSTABLE"
        print(f"  Mean SENS_MAX: {np.mean(sens_maxes):.4f} +/- {np.std(sens_maxes):.4f}")
        print(f"  Mean PCC: {np.mean(pccs_mean):.4f} [{pcc_status}]")
    
    # 2. LIME Stochastic Stability
    print(f"\n--- LIME Stochastic Stability ---")
    lime_corrs = []
    for i, idx in enumerate(eval_idx[:10]):
        sample = X_test[idx]
        mean_corr, std_corr = compute_lime_stability(X_train, predict_fn, sample, n_runs=10)
        lime_corrs.append(mean_corr)
        
        if (i + 1) % 5 == 0:
            print(f"  Sample {i+1}/10 | Mean Spearman: {mean_corr:.4f}")
    
    stability_results['lime_stability'] = {
        'mean_spearman': float(np.mean(lime_corrs)),
        'std_spearman': float(np.std(lime_corrs)),
    }
    lime_status = "STABLE" if np.mean(lime_corrs) > 0.6 else "UNSTABLE"
    print(f"  Overall LIME stability: {np.mean(lime_corrs):.4f} +/- {np.std(lime_corrs):.4f} [{lime_status}]")
    
    # 3. Faithfulness
    print(f"\n--- Faithfulness (SHAP) ---")
    faithfulness_scores = {k: [] for k in [3, 5, 10]}
    
    for i, idx in enumerate(eval_idx[:15]):
        sample = X_test[idx]
        shap_vals = np.array(shap_explainer.shap_values(sample.reshape(1, -1),
                                                          nsamples=100, silent=True))
        if isinstance(shap_vals, list):
            sv = shap_vals[0].flatten()
        else:
            sv = shap_vals.flatten()
        
        for k in faithfulness_scores:
            score = compute_faithfulness(predict_fn, sample, sv, top_k=k)
            faithfulness_scores[k].append(score)
    
    for k, scores in faithfulness_scores.items():
        stability_results['faithfulness'][f'top_{k}'] = {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
        }
        print(f"  Top-{k} masking: mean confidence drop = {np.mean(scores):.4f} +/- {np.std(scores):.4f}")
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, 'stability_results.json'), 'w') as f:
        json.dump(stability_results, f, indent=2)
    
    # Stability comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    eps_vals = [float(e) for e in stability_results['shap_sens_max'].keys()]
    means = [stability_results['shap_sens_max'][str(e)]['mean'] for e in eps_vals]
    stds = [stability_results['shap_sens_max'][str(e)]['std'] for e in eps_vals]
    axes[0].errorbar(eps_vals, means, yerr=stds, marker='o', capsize=5, color='steelblue')
    axes[0].set_xlabel('Perturbation epsilon')
    axes[0].set_ylabel('SENS_MAX')
    axes[0].set_title('SHAP Sensitivity (lower = more stable)')
    axes[0].grid(True, alpha=0.3)
    
    pcc_means = [stability_results['shap_pcc'][str(e)]['mean'] for e in eps_vals]
    axes[1].bar(range(len(eps_vals)), pcc_means, color=['green' if p > 0.6 else 'red' for p in pcc_means])
    axes[1].set_xticks(range(len(eps_vals)))
    axes[1].set_xticklabels([f'eps={e}' for e in eps_vals])
    axes[1].axhline(y=0.6, color='gray', linestyle='--', label='Stability threshold')
    axes[1].set_ylabel('Mean PCC')
    axes[1].set_title('SHAP Stability (PCC > 0.6 = stable)')
    axes[1].legend()
    
    ks = list(faithfulness_scores.keys())
    faith_means = [stability_results['faithfulness'][f'top_{k}']['mean'] for k in ks]
    faith_stds = [stability_results['faithfulness'][f'top_{k}']['std'] for k in ks]
    axes[2].bar(range(len(ks)), faith_means, yerr=faith_stds, color='coral', capsize=5)
    axes[2].set_xticks(range(len(ks)))
    axes[2].set_xticklabels([f'Top-{k}' for k in ks])
    axes[2].set_ylabel('Confidence drop')
    axes[2].set_title('Faithfulness (higher = more faithful)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'stability_evaluation.png'), dpi=150)
    plt.close()
    
    print("\n" + "="*60)
    print("STABILITY SUMMARY")
    print("="*60)
    print(f"SHAP Stability (eps=0.01): PCC={stability_results['shap_pcc']['0.01']['mean']:.4f}")
    print(f"SHAP Stability (eps=0.05): PCC={stability_results['shap_pcc']['0.05']['mean']:.4f}")
    print(f"LIME Stability: Spearman={stability_results['lime_stability']['mean_spearman']:.4f}")
    print(f"Faithfulness (top-5): {stability_results['faithfulness']['top_5']['mean']:.4f}")
    
    print("\nStability evaluation complete!")
    print(f"Results saved to {RESULTS_DIR}/stability_results.json")
    print(f"Plot saved to {RESULTS_DIR}/stability_evaluation.png")


if __name__ == '__main__':
    main()
