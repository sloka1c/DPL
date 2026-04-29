"""
Data preprocessing pipeline for NSL-KDD dataset.
Handles loading, encoding, scaling, and splitting.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from datasets import load_dataset
import pickle
import json

# Fixed seed for reproducibility
SEED = 42
np.random.seed(SEED)

# NSL-KDD attack type to category mapping
ATTACK_MAP = {
    # Normal
    'normal': 'Normal',
    # DoS attacks
    'back': 'DoS', 'land': 'DoS', 'neptune': 'DoS', 'pod': 'DoS',
    'smurf': 'DoS', 'teardrop': 'DoS', 'mailbomb': 'DoS', 'apache2': 'DoS',
    'processtable': 'DoS', 'udpstorm': 'DoS',
    # Probe attacks
    'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe', 'satan': 'Probe',
    'mscan': 'Probe', 'saint': 'Probe',
    # R2L attacks
    'ftp_write': 'R2L', 'guess_passwd': 'R2L', 'imap': 'R2L', 'multihop': 'R2L',
    'phf': 'R2L', 'spy': 'R2L', 'warezclient': 'R2L', 'warezmaster': 'R2L',
    'sendmail': 'R2L', 'named': 'R2L', 'snmpgetattack': 'R2L', 'snmpguess': 'R2L',
    'xlock': 'R2L', 'xsnoop': 'R2L', 'worm': 'R2L',
    # U2R attacks
    'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'perl': 'U2R', 'rootkit': 'U2R',
    'httptunnel': 'U2R', 'ps': 'U2R', 'sqlattack': 'U2R', 'xterm': 'U2R',
}

# 41 features of NSL-KDD
FEATURE_NAMES = [
    'duration', 'protocol_type', 'service', 'flag',
    'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent',
    'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
    'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login',
    'count', 'srv_count',
    'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
    'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
    'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
]

CATEGORICAL_COLS = ['protocol_type', 'service', 'flag']

# Class label mapping for 5-class
CLASS_LABELS = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']


def load_nsl_kdd():
    """Load NSL-KDD from HuggingFace Hub."""
    print("Loading NSL-KDD dataset from HuggingFace Hub...")
    ds = load_dataset("Mireu-Lab/NSL-KDD")
    
    df_train = ds['train'].to_pandas()
    df_test = ds['test'].to_pandas()
    
    print(f"Train: {len(df_train)} samples")
    print(f"Test:  {len(df_test)} samples")
    
    return df_train, df_test


def analyze_dataset(df_train, df_test):
    """Print dataset statistics for documentation."""
    print("\n" + "="*60)
    print("DATASET ANALYSIS")
    print("="*60)
    
    print(f"\nTraining set: {len(df_train)} samples")
    print(f"Test set:     {len(df_test)} samples")
    
    print("\n--- Class Distribution (Training) ---")
    train_dist = df_train['class'].value_counts()
    for cls, count in train_dist.items():
        pct = 100 * count / len(df_train)
        print(f"  {cls:10s}: {count:6d} ({pct:.1f}%)")
    
    print("\n--- Class Distribution (Test) ---")
    test_dist = df_test['class'].value_counts()
    for cls, count in test_dist.items():
        pct = 100 * count / len(df_test)
        print(f"  {cls:10s}: {count:6d} ({pct:.1f}%)")
    
    print("\n--- Categorical Features ---")
    for col in CATEGORICAL_COLS:
        n_train = df_train[col].nunique()
        n_test = df_test[col].nunique()
        print(f"  {col:15s}: {n_train} train / {n_test} test unique values")
        
        # Check for unseen test values
        train_vals = set(df_train[col].unique())
        test_vals = set(df_test[col].unique())
        unseen = test_vals - train_vals
        if unseen:
            print(f"    Warning: {len(unseen)} unseen test values: {unseen}")
    
    print("\n--- Feature Ranges (numeric) ---")
    numeric_cols = [c for c in FEATURE_NAMES if c not in CATEGORICAL_COLS]
    for col in numeric_cols[:10]:
        print(f"  {col:35s}: [{df_train[col].min():.2f}, {df_train[col].max():.2f}]")
    print(f"  ... and {len(numeric_cols)-10} more numeric features")
    
    return train_dist, test_dist


def preprocess(df_train, df_test, binary=True):
    """
    Full preprocessing pipeline.
    
    Args:
        df_train: Training DataFrame
        df_test: Test DataFrame
        binary: If True, binary classification (normal vs anomaly)
    
    Returns:
        X_train, X_test, y_train, y_test, label_encoders, scaler, class_names
    """
    print(f"\nPreprocessing ({'binary' if binary else '5-class'} classification)...")
    
    df_tr = df_train.copy()
    df_te = df_test.copy()
    
    # --- Encode target ---
    if binary:
        class_names = ['anomaly', 'normal']
        le_y = LabelEncoder()
        y_train = le_y.fit_transform(df_tr['class'].values)
        y_test = le_y.transform(df_te['class'].values)
    else:
        class_names = CLASS_LABELS
        le_y = LabelEncoder()
        le_y.classes_ = np.array(CLASS_LABELS)
        y_train = le_y.fit_transform(df_tr['class'].values)
        y_test = le_y.transform(df_te['class'].values)
    
    # --- Encode categorical features ---
    label_encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        le.fit(df_tr[col])
        
        # Handle unseen test labels
        known = set(le.classes_)
        df_te[col] = df_te[col].apply(lambda x: x if x in known else le.classes_[0])
        
        df_tr[col] = le.transform(df_tr[col])
        df_te[col] = le.transform(df_te[col])
        label_encoders[col] = le
        print(f"  Encoded {col}: {len(le.classes_)} categories")
    
    # --- Extract features ---
    X_train = df_tr[FEATURE_NAMES].values.astype(np.float32)
    X_test = df_te[FEATURE_NAMES].values.astype(np.float32)
    
    # --- Scale features ---
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_test shape:  {X_test.shape}")
    print(f"  y_train distribution: {np.bincount(y_train)}")
    print(f"  y_test distribution:  {np.bincount(y_test)}")
    
    return X_train, X_test, y_train, y_test, label_encoders, scaler, class_names


def save_preprocessed(X_train, X_test, y_train, y_test, label_encoders, scaler, 
                       class_names, output_dir='data/processed'):
    """Save preprocessed data for reproducibility."""
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
    
    with open(os.path.join(output_dir, 'encoders.pkl'), 'wb') as f:
        pickle.dump({'label_encoders': label_encoders, 'scaler': scaler}, f)
    
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump({
            'feature_names': FEATURE_NAMES,
            'categorical_cols': CATEGORICAL_COLS,
            'class_names': class_names,
            'n_train': len(X_train),
            'n_test': len(X_test),
            'n_features': X_train.shape[1],
            'seed': SEED,
        }, f, indent=2)
    
    print(f"\nSaved preprocessed data to {output_dir}/")


def load_preprocessed(data_dir='data/processed'):
    """Load preprocessed data."""
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    
    with open(os.path.join(data_dir, 'encoders.pkl'), 'rb') as f:
        objs = pickle.load(f)
    
    with open(os.path.join(data_dir, 'metadata.json')) as f:
        meta = json.load(f)
    
    return X_train, X_test, y_train, y_test, objs['label_encoders'], objs['scaler'], meta


if __name__ == '__main__':
    df_train, df_test = load_nsl_kdd()
    analyze_dataset(df_train, df_test)
    
    X_train, X_test, y_train, y_test, le, scaler, class_names = preprocess(
        df_train, df_test, binary=True
    )
    save_preprocessed(X_train, X_test, y_train, y_test, le, scaler, class_names)
    
    print("\nPreprocessing complete!")
