"""
Three-tier (device/edge -> district/fog -> city/cloud) hierarchical FedAvg,
mirroring the layered device-edge-cloud topology envisioned for 6G-enabled
smart-city sensing networks. Works with any torch.nn.Module client model
(classical MLP or VQC) since aggregation operates on state_dict tensors.
"""

import copy
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch import nn


def _weighted_average_state_dicts(state_dicts, weights):
    total = float(sum(weights))
    avg = {}
    for key in state_dicts[0]:
        stacked = torch.stack([sd[key].float() * (w / total) for sd, w in zip(state_dicts, weights)])
        avg[key] = stacked.sum(dim=0)
    return avg


def _stratified_batch_indices(y, batch_size, rng):
    """Class-balanced batch sampling: half positive, half negative class
    indices per batch (with replacement within the minority class where
    needed). The heavy-pollution label is imbalanced (roughly 10-15%
    positive); balanced mini-batches are standard practice for stable
    gradient estimates under class imbalance and are used identically for
    every model in the comparison.
    """
    pos = np.where(y > 0.5)[0]
    neg = np.where(y <= 0.5)[0]
    half = batch_size // 2
    pos_idx = rng.choice(pos, size=half, replace=len(pos) < half)
    neg_idx = rng.choice(neg, size=batch_size - half, replace=len(neg) < batch_size - half)
    return np.concatenate([pos_idx, neg_idx])


def _local_train(model, x, y, local_steps, lr, batch_size, rng):
    """`local_steps` mini-batch SGD steps per round, each on a random
    class-balanced batch of `batch_size` local samples. Fixed local-step
    count per round is the standard cross-device FL local-update pattern
    and keeps per-round cost bounded regardless of a client's dataset size
    -- required here because each quantum-circuit sample evaluation is
    expensive relative to a classical forward pass on the state-vector
    simulator.
    """
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(local_steps):
        idx = _stratified_batch_indices(y, batch_size, rng)
        x_t = torch.tensor(x[idx], dtype=torch.float32)
        y_t = torch.tensor(y[idx], dtype=torch.float32)
        opt.zero_grad()
        logits = model(x_t)
        loss = loss_fn(logits, y_t)
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def _evaluate(model, x, y):
    model.eval()
    x_t = torch.tensor(x, dtype=torch.float32)
    logits = model(x_t)
    probs = torch.sigmoid(logits).numpy()
    preds = (probs >= 0.5).astype(int)
    y_int = y.astype(int)
    acc = accuracy_score(y_int, preds)
    f1 = f1_score(y_int, preds, zero_division=0)
    try:
        auc = roc_auc_score(y_int, probs)
    except ValueError:
        auc = float("nan")
    return acc, f1, auc


def run_hierarchical_fedavg(
    model_factory,
    client_data,          # dict station -> (x_train, y_train)
    client_test_data,     # dict station -> (x_test, y_test)
    fog_groups,           # dict group_id -> list[station]
    rounds,
    local_steps,
    lr,
    batch_size=32,
    param_bytes=4,
    seed=0,
):
    """Run hierarchical FedAvg and return a metrics-per-round record.

    Returns dict with keys: round, test_acc, test_f1, cumulative_bytes,
    round_bytes, wall_seconds.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    global_model = model_factory(seed=seed)
    n_params = global_model.param_count()

    n_edges = len(client_data)
    n_fog = len(fog_groups)

    history = {
        "round": [], "test_acc": [], "test_f1": [], "test_auc": [],
        "round_bytes": [], "cumulative_bytes": [], "wall_seconds": [],
    }
    cumulative_bytes = 0

    x_test_pool = np.concatenate([v[0] for v in client_test_data.values()], axis=0)
    y_test_pool = np.concatenate([v[1] for v in client_test_data.values()], axis=0)

    for r in range(1, rounds + 1):
        t0 = time.time()
        client_states = {}
        client_n = {}

        for station, (x_tr, y_tr) in client_data.items():
            local_model = model_factory(seed=seed)
            local_model.load_state_dict(copy.deepcopy(global_model.state_dict()))
            local_model = _local_train(local_model, x_tr, y_tr, local_steps, lr, batch_size, rng)
            client_states[station] = local_model.state_dict()
            client_n[station] = len(x_tr)

        fog_states, fog_n = {}, {}
        for gid, members in fog_groups.items():
            sds = [client_states[m] for m in members]
            ws = [client_n[m] for m in members]
            fog_states[gid] = _weighted_average_state_dicts(sds, ws)
            fog_n[gid] = sum(ws)

        cloud_state = _weighted_average_state_dicts(
            list(fog_states.values()), list(fog_n.values())
        )
        global_model.load_state_dict(cloud_state)

        acc, f1, auc = _evaluate(global_model, x_test_pool, y_test_pool)

        upload_edge_to_fog = n_edges * n_params * param_bytes
        upload_fog_to_cloud = n_fog * n_params * param_bytes
        broadcast_cloud_to_fog = n_fog * n_params * param_bytes
        broadcast_fog_to_edge = n_edges * n_params * param_bytes
        round_bytes = (
            upload_edge_to_fog + upload_fog_to_cloud
            + broadcast_cloud_to_fog + broadcast_fog_to_edge
        )
        cumulative_bytes += round_bytes

        history["round"].append(r)
        history["test_acc"].append(acc)
        history["test_f1"].append(f1)
        history["test_auc"].append(auc)
        history["round_bytes"].append(round_bytes)
        history["cumulative_bytes"].append(cumulative_bytes)
        history["wall_seconds"].append(time.time() - t0)

    history["n_params"] = n_params
    return history
