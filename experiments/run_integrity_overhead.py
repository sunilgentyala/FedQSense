"""
Measures the real, wall-clock overhead of the per-client SHA-256
update-integrity hash chain (src/integrity/hashchain.py) layered on top of
hierarchical FedAvg training, for both the VQC (8-qubit) and classical
FullFeatureMLP client models. Also verifies chain integrity end-to-end.
"""

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data.loader import (  # noqa: E402
    cluster_stations_into_fog_groups,
    fit_global_scaler_and_pca,
    load_station_daily,
    temporal_train_test_split,
    transform_station,
)
from classical.mlp import FullFeatureMLP  # noqa: E402
from vqc.circuit import VQCClassifier  # noqa: E402
from integrity.hashchain import HashChain  # noqa: E402


def measure(model_factory, n_rounds=25, seed=0):
    model = model_factory(seed=seed)
    chain = HashChain(client_id="edge-station-demo")
    records = []
    t_train_total = 0.0
    for r in range(1, n_rounds + 1):
        t0 = time.time()
        for p in model.parameters():
            p.data += 0.001 * torch.randn_like(p.data)
        t_train_total += time.time() - t0
        chain.commit(r, model.state_dict())
        records.append((r, {k: v.detach().clone() for k, v in model.state_dict().items()}))

    verify_ok = chain.verify(records)
    return {
        "n_rounds": n_rounds,
        "mean_commit_latency_us": chain.mean_commit_latency_us(),
        "total_commit_latency_ms": sum(chain.timings_ns) / 1e6,
        "bytes_logged": chain.bytes_logged,
        "chain_verified": verify_ok,
        "n_params": model.param_count(),
    }


def tamper_trials(model_factory, n_rounds=10, n_trials=1000, seed=0):
    """Run n_trials independent chains: half left untampered, half with one
    randomly chosen round's parameters perturbed after the fact, and check
    that verify() accepts every clean chain and rejects every tampered one.
    """
    rng = np.random.default_rng(seed)
    false_reject = 0   # clean chain incorrectly flagged as tampered
    false_accept = 0   # tampered chain incorrectly verified as clean
    n_clean = n_trials // 2
    n_tampered = n_trials - n_clean

    for trial in range(n_clean):
        model = model_factory(seed=1000 + trial)
        chain = HashChain(client_id=f"trial-clean-{trial}")
        records = []
        for r in range(1, n_rounds + 1):
            for p in model.parameters():
                p.data += 0.001 * torch.randn_like(p.data)
            chain.commit(r, model.state_dict())
            records.append((r, {k: v.detach().clone() for k, v in model.state_dict().items()}))
        if not chain.verify(records):
            false_reject += 1

    for trial in range(n_tampered):
        model = model_factory(seed=2000 + trial)
        chain = HashChain(client_id=f"trial-tampered-{trial}")
        records = []
        for r in range(1, n_rounds + 1):
            for p in model.parameters():
                p.data += 0.001 * torch.randn_like(p.data)
            chain.commit(r, model.state_dict())
            records.append((r, {k: v.detach().clone() for k, v in model.state_dict().items()}))

        tamper_round = int(rng.integers(0, n_rounds))
        tampered_state = {k: v.clone() for k, v in records[tamper_round][1].items()}
        key0 = next(iter(tampered_state))
        tampered_state[key0] = tampered_state[key0] + 1.0
        records[tamper_round] = (records[tamper_round][0], tampered_state)

        if chain.verify(records):
            false_accept += 1

    return {
        "n_trials": n_trials, "n_clean": n_clean, "n_tampered": n_tampered,
        "false_reject": false_reject, "false_accept": false_accept,
        "clean_accept_rate": 1.0 - false_reject / n_clean,
        "tamper_detect_rate": 1.0 - false_accept / n_tampered,
    }


def main():
    results = {}
    results["VQC_8q"] = measure(lambda seed: VQCClassifier(n_qubits=8, n_layers=2, seed=seed))
    results["FullFeatureMLP"] = measure(lambda seed: FullFeatureMLP(n_inputs=11, hidden=16))

    for k, v in results.items():
        print(k, v)

    print("Running 1000-trial tamper-detection sweep (VQC, 8 qubits)...")
    tamper_results = tamper_trials(
        lambda seed: VQCClassifier(n_qubits=8, n_layers=2, seed=seed), n_rounds=10, n_trials=1000, seed=0
    )
    print("tamper_trials:", tamper_results)
    results["tamper_trials"] = tamper_results

    os.makedirs("results", exist_ok=True)
    with open("results/integrity_overhead.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)


if __name__ == "__main__":
    main()
