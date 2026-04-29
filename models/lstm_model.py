"""
LSTM model for Intrusion Detection.
Treats 41 features as a sequence of length 41 with 1 feature per step.
"""

import torch
import torch.nn as nn


class LSTM_IDS(nn.Module):
    """
    LSTM-based IDS model.
    
    Design choices:
    - Reshape (batch, 41) → (batch, 41, 1): treat each feature as a time step
    - 2 LSTM layers with hidden=64: captures inter-feature dependencies
    - Use last hidden state for classification
    - Feature ordering in NSL-KDD is semantic: basic→content→time-based→host-based
    """
    
    def __init__(self, in_dim=41, hidden_dim=64, num_layers=2, num_classes=2, dropout=0.2):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        # x: (batch, 41) → (batch, 41, 1)
        x = x.unsqueeze(-1)
        out, (h_n, c_n) = self.lstm(x)
        # h_n: (num_layers, batch, hidden) → take last layer
        last_hidden = h_n[-1]  # (batch, hidden)
        return self.fc(last_hidden)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':
    model = LSTM_IDS(in_dim=41, num_classes=2)
    print(model)
    print(f"\nTotal parameters: {model.count_parameters():,}")
    
    x = torch.randn(32, 41)
    out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
