"""
Main FedQSense experiment: hierarchical FedAvg with VQC vs. classical client
models across qubit budgets {4, 6, 8}, multiple seeds, on the real Beijing
Multi-Site Air-Quality daily heavy-pollution-event task.

Produces results/main_results.csv (per round, per config, per seed) and
results/main_summary.csv (final-round aggregate mean +/- std).
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data.loader import (  # noqa: E402
    cluster_stations_into_fog_groups,
    fit_global_scaler_and_pca,
    load_station_daily,
    temporal_train_test_split,
    transform_station,
)
from classical.mlp import FullFeatureMLP, MatchedInputMLP  # noqa: E402
from tiers.hierarchical_fedavg import run_hierarchical_fedavg  # noqa: E402
from vqc.circuit import VQCClassifier  # noqa: E402

QUBIT_BUDGETS = [4, 6, 8]
SEEDS = [0, 1, 2, 3, 4]
ROUNDS = 25
LOCAL_STEPS = 3
BATCH_SIZE = 32
LR_VQC = 0.1
LR_MLP = 0.01
N_FOG_GROUPS = 4


def scale_xy(df, scaler, pca, quantum):
    x, y = transform_station(df, scaler, pca)
    if quantum:
        x = np.clip(x, -3, 3) * (np.pi / 3)
    return x, y


def build_client_sets(train, test, scaler, pca, quantum):
    ctr = {k: scale_xy(v, scaler, pca, quantum) for k, v in train.items()}
    cte = {k: scale_xy(v, scaler, pca, quantum) for k, v in test.items()}
    return ctr, cte


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data_raw/PRSA/PRSA_Data_20130301-20170228")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join("evidence", "logs", "main_experiment.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_lines = []

    def log(msg):
        print(msg)
        log_lines.append(msg)

    t_start = time.time()
    stations = load_station_daily(args.data_dir)
    train, test = temporal_train_test_split(stations, test_days=180)
    fog_groups = cluster_stations_into_fog_groups(train, n_groups=N_FOG_GROUPS)
    log(f"Loaded {len(stations)} stations. Fog groups: { {k: v for k, v in fog_groups.items()} }")

    rows = []

    for nq in QUBIT_BUDGETS:
        scaler, pca, explained = fit_global_scaler_and_pca(train, n_components=nq)
        log(f"[Q={nq}] PCA explained variance ratio: {explained:.4f}")

        ctr_q, cte_q = build_client_sets(train, test, scaler, pca, quantum=True)

        matched_hidden = max(1, round((2 * nq * 3 + 1 - 1) / (nq + 2)))

        for seed in args.seeds:
            t0 = time.time()
            hist = run_hierarchical_fedavg(
                lambda seed, nq=nq: VQCClassifier(n_qubits=nq, n_layers=2, seed=seed),
                ctr_q, cte_q, fog_groups,
                rounds=args.rounds, local_steps=LOCAL_STEPS, lr=LR_VQC,
                batch_size=BATCH_SIZE, seed=seed,
            )
            dt = time.time() - t0
            log(f"[Q={nq} seed={seed}] VQC done in {dt:.1f}s, "
                f"final acc={hist['test_acc'][-1]:.3f} f1={hist['test_f1'][-1]:.3f} "
                f"auc={hist['test_auc'][-1]:.3f} params={hist['n_params']}")
            for i in range(len(hist["round"])):
                rows.append({
                    "model": "VQC", "qubits": nq, "seed": seed,
                    "round": hist["round"][i], "test_acc": hist["test_acc"][i],
                    "test_f1": hist["test_f1"][i], "test_auc": hist["test_auc"][i],
                    "round_bytes": hist["round_bytes"][i],
                    "cumulative_bytes": hist["cumulative_bytes"][i],
                    "wall_seconds": hist["wall_seconds"][i],
                    "n_params": hist["n_params"],
                })

            t0 = time.time()
            hist_m = run_hierarchical_fedavg(
                lambda seed, nq=nq, h=matched_hidden: MatchedInputMLP(n_inputs=nq, hidden=h),
                ctr_q, cte_q, fog_groups,
                rounds=args.rounds, local_steps=LOCAL_STEPS, lr=LR_MLP,
                batch_size=BATCH_SIZE, seed=seed,
            )
            dt = time.time() - t0
            log(f"[Q={nq} seed={seed}] MatchedMLP(h={matched_hidden}) done in {dt:.1f}s, "
                f"final acc={hist_m['test_acc'][-1]:.3f} f1={hist_m['test_f1'][-1]:.3f} "
                f"auc={hist_m['test_auc'][-1]:.3f} params={hist_m['n_params']}")
            for i in range(len(hist_m["round"])):
                rows.append({
                    "model": "MatchedMLP", "qubits": nq, "seed": seed,
                    "round": hist_m["round"][i], "test_acc": hist_m["test_acc"][i],
                    "test_f1": hist_m["test_f1"][i], "test_auc": hist_m["test_auc"][i],
                    "round_bytes": hist_m["round_bytes"][i],
                    "cumulative_bytes": hist_m["cumulative_bytes"][i],
                    "wall_seconds": hist_m["wall_seconds"][i],
                    "n_params": hist_m["n_params"],
                })

    scaler_full, _, _ = fit_global_scaler_and_pca(train, n_components=None)
    ctr_full = {k: scale_xy(v, scaler_full, None, quantum=False) for k, v in train.items()}
    cte_full = {k: scale_xy(v, scaler_full, None, quantum=False) for k, v in test.items()}

    for seed in args.seeds:
        t0 = time.time()
        hist_f = run_hierarchical_fedavg(
            lambda seed: FullFeatureMLP(n_inputs=11, hidden=16),
            ctr_full, cte_full, fog_groups,
            rounds=args.rounds, local_steps=LOCAL_STEPS, lr=LR_MLP,
            batch_size=BATCH_SIZE, seed=seed,
        )
        dt = time.time() - t0
        log(f"[FullFeatureMLP seed={seed}] done in {dt:.1f}s, "
            f"final acc={hist_f['test_acc'][-1]:.3f} f1={hist_f['test_f1'][-1]:.3f} "
            f"auc={hist_f['test_auc'][-1]:.3f} params={hist_f['n_params']}")
        for i in range(len(hist_f["round"])):
            rows.append({
                "model": "FullFeatureMLP", "qubits": None, "seed": seed,
                "round": hist_f["round"][i], "test_acc": hist_f["test_acc"][i],
                "test_f1": hist_f["test_f1"][i], "test_auc": hist_f["test_auc"][i],
                "round_bytes": hist_f["round_bytes"][i],
                "cumulative_bytes": hist_f["cumulative_bytes"][i],
                "wall_seconds": hist_f["wall_seconds"][i],
                "n_params": hist_f["n_params"],
            })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, "main_results.csv"), index=False)

    final = df[df["round"] == df.groupby(["model", "qubits", "seed"], dropna=False)["round"].transform("max")]
    summary = (
        final.groupby(["model", "qubits"], dropna=False)
        .agg(
            n_params=("n_params", "first"),
            acc_mean=("test_acc", "mean"), acc_std=("test_acc", "std"),
            f1_mean=("test_f1", "mean"), f1_std=("test_f1", "std"),
            auc_mean=("test_auc", "mean"), auc_std=("test_auc", "std"),
            cumulative_bytes=("cumulative_bytes", "first"),
            n_seeds=("seed", "nunique"),
        )
        .reset_index()
    )
    summary.to_csv(os.path.join(args.out_dir, "main_summary.csv"), index=False)

    log(f"Total wall time: {time.time() - t_start:.1f}s")
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(log_lines))

    print("\n=== SUMMARY ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
