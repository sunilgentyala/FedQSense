import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from classical.mlp import MatchedInputMLP  # noqa: E402
from tiers.hierarchical_fedavg import run_hierarchical_fedavg  # noqa: E402


def _synthetic_clients(n_clients=6, n_samples=80, n_features=4, seed=0):
    rng = np.random.default_rng(seed)
    clients = {}
    for i in range(n_clients):
        x = rng.normal(size=(n_samples, n_features)).astype(np.float32)
        true_w = rng.normal(size=n_features)
        logits = x @ true_w
        y = (logits > np.median(logits)).astype(np.float32)
        clients[f"client_{i}"] = (x, y)
    return clients


def test_fedavg_runs_and_improves_over_random_init():
    train = _synthetic_clients(seed=0)
    test = _synthetic_clients(seed=1)
    fog_groups = {0: ["client_0", "client_1", "client_2"], 1: ["client_3", "client_4", "client_5"]}

    def factory(seed):
        torch.manual_seed(seed)
        return MatchedInputMLP(n_inputs=4, hidden=4)

    hist = run_hierarchical_fedavg(
        factory, train, test, fog_groups,
        rounds=10, local_steps=5, lr=0.05, batch_size=16, seed=0,
    )
    assert len(hist["round"]) == 10
    assert hist["test_auc"][-1] >= hist["test_auc"][0] - 0.05
    assert hist["n_params"] == MatchedInputMLP(n_inputs=4, hidden=4).param_count()


def test_communication_bytes_scale_with_param_count():
    train = _synthetic_clients(seed=0)
    test = _synthetic_clients(seed=1)
    fog_groups = {0: ["client_0", "client_1", "client_2"], 1: ["client_3", "client_4", "client_5"]}

    def small_factory(seed):
        return MatchedInputMLP(n_inputs=4, hidden=2)

    def large_factory(seed):
        return MatchedInputMLP(n_inputs=4, hidden=20)

    small_hist = run_hierarchical_fedavg(
        small_factory, train, test, fog_groups, rounds=2, local_steps=2, lr=0.05, batch_size=16, seed=0
    )
    large_hist = run_hierarchical_fedavg(
        large_factory, train, test, fog_groups, rounds=2, local_steps=2, lr=0.05, batch_size=16, seed=0
    )
    assert large_hist["round_bytes"][0] > small_hist["round_bytes"][0]
    ratio = large_hist["n_params"] / small_hist["n_params"]
    byte_ratio = large_hist["round_bytes"][0] / small_hist["round_bytes"][0]
    assert abs(ratio - byte_ratio) < 1e-6
