"""
Variational quantum circuit (VQC) client model for FedQSense.

Angle-encodes PCA-reduced sensor features onto Q qubits, applies L layers of
PennyLane's StronglyEntanglingLayers, and reads out <Z> on qubit 0 as the
pollution-event logit. All circuits run on PennyLane's exact state-vector
simulator (default.qubit); no physical QPU access is used or claimed.
"""

import pennylane as qml
import torch
from torch import nn


class VQCClassifier(nn.Module):
    def __init__(self, n_qubits, n_layers=2, seed=0):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return qml.expval(qml.PauliZ(0))

        weight_shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
        g = torch.Generator().manual_seed(seed)
        init = 0.1 * torch.randn(weight_shape, generator=g)
        self.weights = nn.Parameter(init)
        self.circuit = circuit
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x: (batch, n_qubits) pre-scaled to roughly [-pi, pi].
        # PennyLane's default.qubit natively broadcasts over the leading
        # batch dimension of `inputs`, avoiding a per-sample Python loop.
        outs = self.circuit(x, self.weights)
        return outs + self.bias

    def param_count(self):
        return sum(p.numel() for p in self.parameters())

    def state_dict_flat(self):
        return {k: v.detach().clone() for k, v in self.state_dict().items()}
