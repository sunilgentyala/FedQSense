"""
Circuit-depth ablation at the 8-qubit budget: L in {2, 3, 4} entangling
layers, 5 seeds each, same hierarchical FedAvg protocol as the main
experiment. Complements run_main_experiment.py (which fixes L=2 across
qubit budgets) by isolating the accuracy/communication/depth trade-off at
fixed qubit count -- deeper ansaetze partially close the accuracy gap to
classical baselines at the cost of more transmitted parameters and (per
run_barren_plateau.py) worse trainability.
"""

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
from tiers.hierarchical_fedavg import run_hierarchical_fedavg  # noqa: E402
from vqc.circuit import VQCClassifier  # noqa: E402

QUBITS = 8
DEPTHS = [2, 3, 4]
SEEDS = [0, 1, 2, 3, 4]
ROUNDS = 25
LOCAL_STEPS = 3
BATCH_SIZE = 32
LR = 0.1
N_FOG_GROUPS = 4


def scale_xy(df, scaler, pca):
    x, y = transform_station(df, scaler, pca)
    x = np.clip(x, -3, 3) * (np.pi / 3)
    return x, y


def main():
    stations = load_station_daily("data_raw/PRSA/PRSA_Data_20130301-20170228")
    train, test = temporal_train_test_split(stations, test_days=180)
    fog_groups = cluster_stations_into_fog_groups(train, n_groups=N_FOG_GROUPS)
    scaler, pca, explained = fit_global_scaler_and_pca(train, n_components=QUBITS)

    client_train = {k: scale_xy(v, scaler, pca) for k, v in train.items()}
    client_test = {k: scale_xy(v, scaler, pca) for k, v in test.items()}

    rows = []
    for depth in DEPTHS:
        for seed in SEEDS:
            t0 = time.time()
            hist = run_hierarchical_fedavg(
                lambda seed, L=depth: VQCClassifier(n_qubits=QUBITS, n_layers=L, seed=seed),
                client_train, client_test, fog_groups,
                rounds=ROUNDS, local_steps=LOCAL_STEPS, lr=LR,
                batch_size=BATCH_SIZE, seed=seed,
            )
            dt = time.time() - t0
            print(f"[depth={depth} seed={seed}] {dt:.1f}s final_auc={hist['test_auc'][-1]:.3f} "
                  f"final_acc={hist['test_acc'][-1]:.3f} params={hist['n_params']}")
            for i in range(len(hist["round"])):
                rows.append({
                    "depth": depth, "seed": seed, "round": hist["round"][i],
                    "test_acc": hist["test_acc"][i], "test_f1": hist["test_f1"][i],
                    "test_auc": hist["test_auc"][i],
                    "cumulative_bytes": hist["cumulative_bytes"][i],
                    "n_params": hist["n_params"],
                })

    df = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/depth_ablation.csv", index=False)

    final = df[df["round"] == ROUNDS]
    summary = (
        final.groupby("depth")
        .agg(
            n_params=("n_params", "first"),
            acc_mean=("test_acc", "mean"), acc_std=("test_acc", "std"),
            auc_mean=("test_auc", "mean"), auc_std=("test_auc", "std"),
            cumulative_bytes=("cumulative_bytes", "first"),
        )
        .reset_index()
    )
    summary.to_csv("results/depth_ablation_summary.csv", index=False)
    print("\n=== DEPTH ABLATION SUMMARY ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
