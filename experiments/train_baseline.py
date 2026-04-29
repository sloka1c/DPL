"""
Training script for all three IDS models.
Trains MLP, LSTM, and 1D-CNN on NSL-KDD with full evaluation.
"""

import os
import sys
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, average_precision_score)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mlp_baseline import MLP_IDS
from models.lstm_model import LSTM_IDS
from models.cnn1d_model import CNN1D_IDS
from data.preprocess import load_nsl_kdd, preprocess, save_preprocessed, FEATURE_NAMES

# ========================
# Reproducibility
# ========================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ========================
# Config
# ========================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 2  # Binary classification
EPOCHS = 50
BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 1e-4
RESULTS_DIR = 'results'
MODELS_DIR = 'saved_models'


def compute_class_weights(y_train):
    """Compute inverse-frequency class weights."""
    counts = np.bincount(y_train)
    weights = 1.0 / counts.astype(np.float32)
    weights = weights / weights.sum() * len(weights)  # Normalize
    return torch.FloatTensor(weights).to(DEVICE)


def train_one_epoch(model, loader, criterion, optimizer):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(y_batch)
        preds = outputs.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += len(y_batch)
    
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    """Evaluate model on dataset."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_probs = []
    all_labels = []
    
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        total_loss += loss.item() * len(y_batch)
        probs = torch.softmax(outputs, dim=1)
        all_preds.append(outputs.argmax(dim=1).cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        all_labels.append(y_batch.cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    
    avg_loss = total_loss / len(all_labels)
    
    return avg_loss, all_preds, all_probs, all_labels


def full_evaluation(y_true, y_pred, y_probs, class_names):
    """Compute all metrics."""
    results = {}
    
    # Classification report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    results['classification_report'] = report
    
    # ROC-AUC (binary)
    if len(class_names) == 2:
        results['roc_auc'] = roc_auc_score(y_true, y_probs[:, 1])
        results['pr_auc'] = average_precision_score(y_true, y_probs[:, 1])
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    results['confusion_matrix'] = cm.tolist()
    
    return results


def train_model(model, model_name, X_train, y_train, X_test, y_test, class_names):
    """Full training pipeline for one model."""
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    print(f"Parameters: {model.count_parameters():,}")
    print(f"Device: {DEVICE}")
    
    # Data loaders
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    test_ds = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # Loss with class weights
    class_weights = compute_class_weights(y_train)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Training loop
    model.to(DEVICE)
    best_f1 = 0
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        test_loss, test_preds, test_probs, test_labels = evaluate(model, test_loader, criterion)
        test_acc = (test_preds == test_labels).mean()
        
        scheduler.step(test_loss)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        # Check for best model
        report = classification_report(test_labels, test_preds, output_dict=True)
        weighted_f1 = report['weighted avg']['f1-score']
        
        if weighted_f1 > best_f1:
            best_f1 = weighted_f1
            os.makedirs(MODELS_DIR, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, f'{model_name}_best.pt'))
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{EPOCHS} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Test Loss: {test_loss:.4f} Acc: {test_acc:.4f} F1: {weighted_f1:.4f}")
    
    train_time = time.time() - start_time
    print(f"\n  Training time: {train_time:.1f}s")
    
    # Load best model and final evaluation
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, f'{model_name}_best.pt'),
                                      weights_only=True))
    _, final_preds, final_probs, final_labels = evaluate(model, test_loader, criterion)
    
    results = full_evaluation(final_labels, final_preds, final_probs, class_names)
    results['training_time'] = train_time
    results['best_weighted_f1'] = best_f1
    results['history'] = history
    results['parameters'] = model.count_parameters()
    
    # Print final results
    print(f"\n  Final Results ({model_name}):")
    print(f"  {'='*50}")
    print(classification_report(final_labels, final_preds, target_names=class_names))
    
    if 'roc_auc' in results:
        print(f"  ROC-AUC: {results['roc_auc']:.4f}")
        print(f"  PR-AUC:  {results['pr_auc']:.4f}")
    
    print(f"  Confusion Matrix:\n{confusion_matrix(final_labels, final_preds)}")
    
    return model, results


def main():
    # ========================
    # Data
    # ========================
    df_train, df_test = load_nsl_kdd()
    X_train, X_test, y_train, y_test, le, scaler, class_names = preprocess(
        df_train, df_test, binary=True
    )
    save_preprocessed(X_train, X_test, y_train, y_test, le, scaler, class_names)
    
    # ========================
    # Train all models
    # ========================
    all_results = {}
    
    # 1. MLP Baseline
    mlp = MLP_IDS(in_dim=41, num_classes=NUM_CLASSES)
    mlp, mlp_results = train_model(mlp, 'mlp', X_train, y_train, X_test, y_test, class_names)
    all_results['mlp'] = mlp_results
    
    # 2. LSTM
    lstm = LSTM_IDS(in_dim=41, num_classes=NUM_CLASSES)
    lstm, lstm_results = train_model(lstm, 'lstm', X_train, y_train, X_test, y_test, class_names)
    all_results['lstm'] = lstm_results
    
    # 3. 1D-CNN
    cnn = CNN1D_IDS(in_dim=41, num_classes=NUM_CLASSES)
    cnn, cnn_results = train_model(cnn, 'cnn1d', X_train, y_train, X_test, y_test, class_names)
    all_results['cnn1d'] = cnn_results
    
    # ========================
    # Save results
    # ========================
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    def convert(o):
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return o
    
    with open(os.path.join(RESULTS_DIR, 'training_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=convert)
    
    # ========================
    # Summary comparison
    # ========================
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    print(f"{'Model':<10} {'Params':>8} {'Accuracy':>10} {'W-F1':>8} {'ROC-AUC':>9} {'PR-AUC':>8} {'Time':>8}")
    print("-"*60)
    
    for name, res in all_results.items():
        acc = res['classification_report']['accuracy']
        wf1 = res['best_weighted_f1']
        roc = res.get('roc_auc', 0)
        pr = res.get('pr_auc', 0)
        t = res['training_time']
        p = res['parameters']
        print(f"{name:<10} {p:>8,} {acc:>10.4f} {wf1:>8.4f} {roc:>9.4f} {pr:>8.4f} {t:>7.1f}s")
    
    print("\nAll models trained successfully!")
    print(f"Results saved to {RESULTS_DIR}/training_results.json")
    print(f"Models saved to {MODELS_DIR}/")


if __name__ == '__main__':
    main()
