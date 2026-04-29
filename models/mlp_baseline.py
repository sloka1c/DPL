"""
MLP Baseline model for Intrusion Detection.
Architecture: 41 → 256 → 128 → 64 → num_classes
"""

import torch
import torch.nn as nn


class MLP_IDS(nn.Module):
    """
    Multi-Layer Perceptron for network intrusion detection.
    
    Design choices:
    - BatchNorm after each linear layer → stabilizes training, enables higher LR
    - Dropout decreasing (0.3 → 0.2) → heavier regularization in wider layers
    - ReLU activation → standard, well-understood, gradient-friendly
    - No final activation → CrossEntropyLoss includes LogSoftmax
    """
    
    def __init__(self, in_dim=41, num_classes=2, hidden_dims=(256, 128, 64)):
        super().__init__()
        
        layers = []
        prev_dim = in_dim
        dropouts = [0.3, 0.2, 0.0]  # Decreasing dropout
        
        for i, h_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, h_dim))
            if i < len(hidden_dims) - 1:  # No BatchNorm on last hidden layer
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            if dropouts[i] > 0:
                layers.append(nn.Dropout(dropouts[i]))
            prev_dim = h_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.net = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.net(x)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':
    model = MLP_IDS(in_dim=41, num_classes=2)
    print(model)
    print(f"\nTotal parameters: {model.count_parameters():,}")
    
    # Test forward pass
    x = torch.randn(32, 41)
    out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
