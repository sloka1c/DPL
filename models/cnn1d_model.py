"""
1D-CNN model for Intrusion Detection.
Applies 1D convolutions over the feature vector.
"""

import torch
import torch.nn as nn


class CNN1D_IDS(nn.Module):
    """
    1D Convolutional Neural Network for IDS.
    
    Design choices:
    - Reshape (batch, 41) → (batch, 1, 41): treat features as 1-channel signal
    - Two Conv1d layers (64, 128 filters): learn local feature patterns
    - Kernel size 3: captures triplets of adjacent features
    - AdaptiveAvgPool1d(8): fixed output size regardless of input length
    """
    
    def __init__(self, in_dim=41, num_classes=2):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8)
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 8, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        # x: (batch, 41) → (batch, 1, 41)
        x = x.unsqueeze(1)
        x = self.conv(x)           # (batch, 128, 8)
        x = x.view(x.size(0), -1)  # (batch, 1024)
        return self.fc(x)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':
    model = CNN1D_IDS(in_dim=41, num_classes=2)
    print(model)
    print(f"\nTotal parameters: {model.count_parameters():,}")
    
    x = torch.randn(32, 41)
    out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
