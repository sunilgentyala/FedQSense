"""Classical client baselines for FedQSense.

Two variants are provided for a controlled comparison against the VQC:
  - MatchedInputMLP: sees the same PCA-Q input as the VQC (fair information
    bottleneck), with hidden width chosen to roughly match the VQC parameter
    budget (iso-parameter comparison).
  - FullFeatureMLP: sees all 11 raw (scaled) features, a conventional
    classical upper-bound reference untouched by the PCA bottleneck.
"""

from torch import nn


class MatchedInputMLP(nn.Module):
    def __init__(self, n_inputs, hidden=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_inputs, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


class FullFeatureMLP(nn.Module):
    def __init__(self, n_inputs, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_inputs, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())
