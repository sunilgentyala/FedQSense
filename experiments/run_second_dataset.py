"""
Second-dataset generalization check for FedQSense: UCI Human Activity
Recognition Using Smartphones (Anguita et al., ESANN 2013; UCI repository
dataset 240), 30 volunteers = 30 natural non-IID federated clients.

Task: binary WALKING_UPSTAIRS (1) vs WALKING_DOWNSTAIRS (0) from the 561
engineered smartphone-sensor features, a fine-grained activity pair. Each subject's windows are split chronologically per activity
(first 75% train, last ~23% test, 2-window gap so the 50%-overlapping windows
cannot leak across the split). Hierarchy: 30 edges -> 6 fog clusters of 5
(rank along PC1 of mean feature profiles, same rule as the air-quality study)
-> cloud. Protocol, baselines, learning rates and seeds are identical to the
air-quality experiments.

Usage: python experiments/run_second_dataset.py <main|depth|compress|lr|stats>
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from classical.mlp import FullFeatureMLP, MatchedInputMLP  # noqa: E402
from tiers.hierarchical_fedavg import run_hierarchical_fedavg  # noqa: E402
from vqc.circuit import VQCClassifier  # noqa: E402
from run_reviewer_experiments import run_compressed  # noqa: E402

torch.set_num_threads(4)
ROOT = "data_raw/HAR/UCI HAR Dataset"
OUT = "results/har"
os.makedirs(OUT, exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]
ROUNDS, STEPS, BS = 25, 3, 32
N_FOG = 6


def load_clients():
    X = np.concatenate([np.loadtxt(f"{ROOT}/{s}/X_{s}.txt") for s in ("train", "test")])
    y = np.concatenate([np.loadtxt(f"{ROOT}/{s}/y_{s}.txt") for s in ("train", "test")]).astype(int)
    sub = np.concatenate([np.loadtxt(f"{ROOT}/{s}/subject_{s}.txt") for s in ("train", "test")]).astype(int)
    tr, te = {}, {}
    for s in sorted(set(sub)):
        tri, tei = [], []
        for act in (2, 3):
            idx = np.where((sub == s) & (y == act))[0]
            idx = np.sort(idx)  # dataset files keep recording order within a subject/activity
            n = len(idx)
            cut = int(round(0.75 * n))
            tri.append(idx[:cut])
            tei.append(idx[cut + 2:])
        tri, tei = np.concatenate(tri), np.concatenate(tei)
        tr[s] = (X[tri], (y[tri] == 2).astype(np.float32))
        te[s] = (X[tei], (y[tei] == 2).astype(np.float32))
    return tr, te


def fog_groups(tr):
    names = sorted(tr)
    prof = StandardScaler().fit_transform(np.stack([tr[n][0].mean(0) for n in names]))
    order = np.argsort(PCA(1, random_state=0).fit_transform(prof).ravel())
    g = {i: [] for i in range(N_FOG)}
    for rank, i in enumerate(order):
        g[rank % N_FOG].append(names[i])
    return g


def prep(nq, quantum):
    tr, te = load_clients()
    fog = fog_groups(tr)
    pooled = np.concatenate([v[0] for v in tr.values()])
    sc = StandardScaler().fit(pooled)
    # whiten: the 561-feature PCA components have std 13.5, 6.6, 5.2, ... so without
    # unit-variance scaling the +/-3 clip before angle encoding saturates the VQC inputs.
    pca = PCA(nq, whiten=True, random_state=0).fit(sc.transform(pooled)) if nq else None

    def f(v):
        x = sc.transform(v[0])
        if pca is not None:
            x = pca.transform(x)
        x = x.astype(np.float32)
        if quantum:
            x = np.clip(x, -3, 3) * (np.pi / 3)
        return x, v[1]
    ev = float(pca.explained_variance_ratio_.sum()) if pca is not None else 1.0
    return {k: f(v) for k, v in tr.items()}, {k: f(v) for k, v in te.items()}, fog, ev


def fin(h):
    return h["test_auc"][-1], h["test_acc"][-1], h["test_f1"][-1], h["n_params"], h["round_bytes"][-1]


def task_main():
    rows = []
    for nq in (4, 6, 8):
        ctr, cte, fog, ev = prep(nq, True)
        hid = max(1, round((2 * nq * 3) / (nq + 2)))
        print(f"Q={nq} explained={ev:.3f} hidden={hid}", flush=True)
        for s in SEEDS:
            h = run_hierarchical_fedavg(lambda seed: VQCClassifier(nq, 2, seed), ctr, cte, fog, ROUNDS, STEPS, 0.1, BS, seed=s)
            rows.append(dict(model="VQC_L2", qubits=nq, seed=s, ev=ev, **dict(zip(["auc", "acc", "f1", "params", "bytes"], fin(h)))))
            print(rows[-1], flush=True)
            h = run_hierarchical_fedavg(lambda seed: MatchedInputMLP(nq, hid), ctr, cte, fog, ROUNDS, STEPS, 0.01, BS, seed=s)
            rows.append(dict(model="MatchedMLP", qubits=nq, seed=s, ev=ev, **dict(zip(["auc", "acc", "f1", "params", "bytes"], fin(h)))))
            print(rows[-1], flush=True)
    ctr, cte, fog, _ = prep(None, False)
    for s in SEEDS:
        h = run_hierarchical_fedavg(lambda seed: FullFeatureMLP(561, 16), ctr, cte, fog, ROUNDS, STEPS, 0.01, BS, seed=s)
        rows.append(dict(model="FullMLP", qubits=None, seed=s, ev=1.0, **dict(zip(["auc", "acc", "f1", "params", "bytes"], fin(h)))))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/main.csv", index=False)


def task_depth():
    rows = []
    ctr, cte, fog, _ = prep(8, True)
    for L in (3, 4):
        for s in SEEDS:
            h = run_hierarchical_fedavg(lambda seed: VQCClassifier(8, L, seed), ctr, cte, fog, ROUNDS, STEPS, 0.1, BS, seed=s)
            rows.append(dict(model=f"VQC_L{L}", qubits=8, seed=s, **dict(zip(["auc", "acc", "f1", "params", "bytes"], fin(h)))))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/depth.csv", index=False)


def task_compress():
    rows = []
    ctr_q, cte_q, fog, _ = prep(8, True)
    ctr_f, cte_f, _, _ = prep(None, False)
    cfgs = [("FullMLP", lambda seed: FullFeatureMLP(561, 16), ctr_f, cte_f, 0.01),
            ("MatchedMLP_Q8", lambda seed: MatchedInputMLP(8, 5), ctr_q, cte_q, 0.01),
            ("VQC_L4_Q8", lambda seed: VQCClassifier(8, 4, seed), ctr_q, cte_q, 0.1)]
    for name, fac, a, b, lr in cfgs:
        for scheme in ("fp32", "int8", "int4"):
            for s in SEEDS:
                auc, acc, by = run_compressed(fac, a, b, fog, lr, s, scheme)
                rows.append(dict(model=name, scheme=scheme, seed=s, auc=auc, acc=acc, bytes_per_round=by))
                print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/compression.csv", index=False)


def task_lr():
    rows = []
    ctr, cte, fog, _ = prep(8, True)
    for lr in (0.03, 0.1, 0.3):
        for s in SEEDS:
            h = run_hierarchical_fedavg(lambda seed: VQCClassifier(8, 2, seed), ctr, cte, fog, ROUNDS, STEPS, lr, BS, seed=s)
            rows.append(dict(model="VQC_L2_Q8", lr=lr, seed=s, auc=fin(h)[0]))
            print(rows[-1], flush=True)
    for lr in (0.003, 0.01, 0.03, 0.1):
        for s in SEEDS:
            h = run_hierarchical_fedavg(lambda seed: MatchedInputMLP(8, 5), ctr, cte, fog, ROUNDS, STEPS, lr, BS, seed=s)
            rows.append(dict(model="MatchedMLP_Q8", lr=lr, seed=s, auc=fin(h)[0]))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/lr.csv", index=False)


def task_stats():
    tr, te = load_clients()
    n = [len(v[1]) for v in tr.values()]
    pr = np.array([v[1].mean() for v in tr.values()])
    sc = StandardScaler().fit(np.concatenate([v[0] for v in tr.values()]))
    gm = sc.transform(np.concatenate([v[0] for v in tr.values()])).mean(0)
    fd = np.array([np.linalg.norm(sc.transform(v[0]).mean(0) - gm) for v in tr.values()])
    print("clients", len(tr), "train n total", sum(n), "test n total", sum(len(v[1]) for v in te.values()))
    print("train n per client min/mean/max", min(n), np.mean(n), max(n))
    print("train pos rate pooled", np.mean(np.concatenate([v[1] for v in tr.values()])),
          "test pos rate pooled", np.mean(np.concatenate([v[1] for v in te.values()])))
    print("pos rate range", pr.min(), pr.max())
    print("feat-mean dist mean/min/max", fd.mean(), fd.min(), fd.max())
    print("fog", fog_groups(tr))


if __name__ == "__main__":
    globals()["task_" + sys.argv[1]]()
