import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from classical.mlp import FullFeatureMLP, MatchedInputMLP  # noqa: E402
from vqc.circuit import VQCClassifier  # noqa: E402


def test_vqc_forward_shape_and_grad():
    model = VQCClassifier(n_qubits=4, n_layers=2, seed=0)
    x = torch.randn(6, 4)
    out = model(x)
    assert out.shape == (6,)
    loss = out.sum()
    loss.backward()
    assert model.weights.grad is not None
    assert not torch.isnan(model.weights.grad).any()


def test_vqc_param_count_matches_strongly_entangling_shape():
    model = VQCClassifier(n_qubits=6, n_layers=3, seed=0)
    # StronglyEntanglingLayers: n_layers * n_qubits * 3 rotation params + 1 bias
    assert model.param_count() == 3 * 6 * 3 + 1


def test_matched_mlp_forward_shape():
    model = MatchedInputMLP(n_inputs=8, hidden=5)
    x = torch.randn(10, 8)
    out = model(x)
    assert out.shape == (10,)


def test_full_feature_mlp_forward_shape():
    model = FullFeatureMLP(n_inputs=11, hidden=16)
    x = torch.randn(10, 11)
    out = model(x)
    assert out.shape == (10,)


def test_vqc_deterministic_given_seed():
    m1 = VQCClassifier(n_qubits=4, n_layers=2, seed=42)
    m2 = VQCClassifier(n_qubits=4, n_layers=2, seed=42)
    assert torch.allclose(m1.weights, m2.weights)
