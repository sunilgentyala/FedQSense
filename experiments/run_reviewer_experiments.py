"""
Reviewer-requested additional experiments for the final FedQSense version.

  lr       learning-rate sensitivity sweep, VQC (Q=8, L=2) and matched MLP (Q=8)
  match    matching-tolerance check: MLP widths bracketing each VQC parameter count
  compress classical (and VQC) update compression: 8-bit / 4-bit quantization and
           top-k sparsification of uploads, with exact byte accounting
  fog      fog-grouping invariance check (random equal-size groupings)
  noniid   quantitative client-heterogeneity statistics
Outputs go to results/reviewer/. Usage: python experiments/run_reviewer_experiments.py <task>
"""
import copy
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data.loader import (cluster_stations_into_fog_groups, fit_global_scaler_and_pca,  # noqa: E402
                         load_station_daily, temporal_train_test_split, transform_station)
from classical.mlp import FullFeatureMLP, MatchedInputMLP  # noqa: E402
from tiers import hierarchical_fedavg as H  # noqa: E402
from tiers.hierarchical_fedavg import run_hierarchical_fedavg  # noqa: E402
from vqc.circuit import VQCClassifier  # noqa: E402

SEEDS = [0, 1, 2, 3, 4]
ROUNDS = 25
STEPS = 3
BS = 32
OUT = "results/reviewer"
DATA = "data_raw/PRSA/PRSA_Data_20130301-20170228"
os.makedirs(OUT, exist_ok=True)


def prep(nq, quantum):
    st = load_station_daily(DATA)
    tr, te = temporal_train_test_split(st, 180)
    fog = cluster_stations_into_fog_groups(tr, 4)
    sc, pca, _ = fit_global_scaler_and_pca(tr, nq)

    def f(df):
        x, y = transform_station(df, sc, pca)
        if quantum:
            x = np.clip(x, -3, 3) * (np.pi / 3)
        return x, y
    return {k: f(v) for k, v in tr.items()}, {k: f(v) for k, v in te.items()}, fog


def final(h):
    return h["test_auc"][-1], h["test_acc"][-1], h["test_f1"][-1], h["n_params"]


def task_lr():
    rows = []
    ctr_q, cte_q, fog = prep(8, True)
    ctr_c, cte_c, _ = prep(8, True)  # same scaled inputs as the VQC, as in the main experiment
    for lr in [0.01, 0.03, 0.1, 0.3]:
        for s in SEEDS:
            h = run_hierarchical_fedavg(lambda seed: VQCClassifier(8, 2, seed), ctr_q, cte_q, fog,
                                        ROUNDS, STEPS, lr, BS, seed=s)
            rows.append(dict(model="VQC_L2_Q8", lr=lr, seed=s, auc=final(h)[0], acc=final(h)[1]))
            print(rows[-1], flush=True)
    for lr in [0.003, 0.01, 0.03, 0.1, 0.3]:
        for s in SEEDS:
            h = run_hierarchical_fedavg(lambda seed: MatchedInputMLP(8, 5), ctr_c, cte_c, fog,
                                        ROUNDS, STEPS, lr, BS, seed=s)
            rows.append(dict(model="MatchedMLP_Q8", lr=lr, seed=s, auc=final(h)[0], acc=final(h)[1]))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/lr_sweep.csv", index=False)


def task_match():
    rows = []
    for nq, hs in [(4, [3, 4, 5]), (6, [4, 5]), (8, [4, 5, 6])]:
        ctr, cte, fog = prep(nq, True)
        for hdn in hs:
            for s in SEEDS:
                h = run_hierarchical_fedavg(lambda seed: MatchedInputMLP(nq, hdn), ctr, cte, fog,
                                            ROUNDS, STEPS, 0.01, BS, seed=s)
                rows.append(dict(qubits=nq, hidden=hdn, params=h["n_params"], seed=s, auc=final(h)[0]))
                print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/match_tolerance.csv", index=False)


def quant(t, bits):
    m = t.abs().max()
    if m == 0:
        return t.clone()
    q = 2 ** (bits - 1) - 1
    return torch.round(t / m * q) / q * m


def topk(t, frac):
    flat = t.flatten()
    k = max(1, int(round(frac * flat.numel())))
    idx = flat.abs().topk(k).indices
    o = torch.zeros_like(flat)
    o[idx] = flat[idx]
    return o.view_as(t)


def apply_codec(delta, scheme):
    if scheme == "fp32":
        return delta
    if scheme == "int8":
        return {k: quant(v, 8) for k, v in delta.items()}
    if scheme == "int4":
        return {k: quant(v, 4) for k, v in delta.items()}
    return {k: topk(v, 0.25) for k, v in delta.items()}


def msg_bytes(numels, scheme):
    if scheme == "fp32":
        return sum(n * 4 for n in numels)
    if scheme == "int8":
        return sum(n + 4 for n in numels)              # 1 B/value + one fp32 scale per tensor
    if scheme == "int4":
        return sum(int(np.ceil(n / 2)) + 4 for n in numels)
    return sum(int(round(0.25 * n)) * 6 for n in numels)  # 4 B value + 2 B index per kept entry


def run_compressed(factory, ctr, cte, fog, lr, seed, scheme):
    """Hierarchical FedAvg where every uplink (edge->fog, fog->cloud) sends a
    codec-compressed update delta relative to the current global model.
    Quantization schemes also quantize the downlink; top-k keeps fp32 downlink."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    g = factory(seed=seed)
    n_e, n_f = len(ctr), len(fog)
    numels = [p.numel() for p in g.state_dict().values()]
    down = "fp32" if scheme == "topk25" else scheme
    per_round = (msg_bytes(numels, scheme) + msg_bytes(numels, down)) * (n_e + n_f)
    xt = np.concatenate([v[0] for v in cte.values()])
    yt = np.concatenate([v[1] for v in cte.values()])
    for _ in range(ROUNDS):
        gs = {k: v.clone() for k, v in g.state_dict().items()}
        cs, cn = {}, {}
        for st, (x, y) in ctr.items():
            m = factory(seed=seed)
            m.load_state_dict(copy.deepcopy(gs))
            H._local_train(m, x, y, STEPS, lr, BS, rng)
            d = apply_codec({k: v.detach() - gs[k] for k, v in m.state_dict().items()}, scheme)
            cs[st] = {k: gs[k] + d[k] for k in gs}
            cn[st] = len(x)
        fs, fn = {}, {}
        for gid, mem in fog.items():
            avg = H._weighted_average_state_dicts([cs[m] for m in mem], [cn[m] for m in mem])
            d = apply_codec({k: avg[k] - gs[k] for k in gs}, scheme)
            fs[gid] = {k: gs[k] + d[k] for k in gs}
            fn[gid] = sum(cn[m] for m in mem)
        cloud = H._weighted_average_state_dicts(list(fs.values()), list(fn.values()))
        if down != "fp32":
            d = apply_codec({k: cloud[k] - gs[k] for k in gs}, down)
            cloud = {k: gs[k] + d[k] for k in gs}
        g.load_state_dict(cloud)
    acc, f1, auc = H._evaluate(g, xt, yt)
    return auc, acc, per_round


def task_compress():
    rows = []
    ctr_c, cte_c, fog = prep(8, True)
    ctr_q, cte_q, _ = prep(8, True)
    st = load_station_daily(DATA)
    tr, te = temporal_train_test_split(st, 180)
    sc, _, _ = fit_global_scaler_and_pca(tr, None)
    ctr_f = {k: transform_station(v, sc, None) for k, v in tr.items()}
    cte_f = {k: transform_station(v, sc, None) for k, v in te.items()}
    cfgs = [("FullMLP_209", lambda seed: FullFeatureMLP(11, 16), ctr_f, cte_f, 0.01),
            ("MatchedMLP_Q8_51", lambda seed: MatchedInputMLP(8, 5), ctr_c, cte_c, 0.01),
            ("VQC_L4_Q8_97", lambda seed: VQCClassifier(8, 4, seed), ctr_q, cte_q, 0.1)]
    for name, fac, a, b, lr in cfgs:
        for scheme in ["fp32", "int8", "int4", "topk25"]:
            for s in SEEDS:
                auc, acc, by = run_compressed(fac, a, b, fog, lr, s, scheme)
                rows.append(dict(model=name, scheme=scheme, seed=s, auc=auc, acc=acc, bytes_per_round=by))
                print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/compression.csv", index=False)


def task_fog():
    rows = []
    ctr, cte, fog = prep(8, True)
    names = sorted(ctr)
    groupings = {"pca": fog, "flat": {0: names}}
    for gi in range(3):
        perm = np.random.default_rng(100 + gi).permutation(names)
        groupings[f"random{gi}"] = {i: list(perm[i::4]) for i in range(4)}
    for gname, g in groupings.items():
        for s in [0, 1]:
            h = run_hierarchical_fedavg(lambda seed: MatchedInputMLP(8, 5), ctr, cte, g,
                                        ROUNDS, STEPS, 0.01, BS, seed=s)
            rows.append(dict(grouping=gname, seed=s, auc=final(h)[0]))
    df = pd.DataFrame(rows)
    print(df)
    df.to_csv(f"{OUT}/fog_invariance.csv", index=False)


def task_noniid():
    st = load_station_daily(DATA)
    tr, te = temporal_train_test_split(st, 180)
    sc, _, _ = fit_global_scaler_and_pca(tr, None)
    feat = [c for c in next(iter(tr.values())).columns if c != "label"]
    pooled_rate = np.mean(np.concatenate([v["label"].values for v in tr.values()]))

    def jsd(p, q):
        p = np.array([1 - p, p])
        q = np.array([1 - q, q])
        m = (p + q) / 2

        def kl(a, b):
            return np.sum(a * np.log2(a / b))
        return 0.5 * kl(p, m) + 0.5 * kl(q, m)
    Z = {k: sc.transform(v[feat].values) for k, v in tr.items()}
    gm = np.concatenate(list(Z.values())).mean(0)
    rows = []
    for k, v in tr.items():
        r = v["label"].mean()
        rows.append(dict(station=k, n_train=len(v), pos_rate_train=r, pos_rate_test=te[k]["label"].mean(),
                         label_jsd_bits=jsd(r, pooled_rate), feat_mean_dist=np.linalg.norm(Z[k].mean(0) - gm)))
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/noniid_stats.csv", index=False)
    print(df.round(4).to_string())
    print("pos-rate train range", df.pos_rate_train.min(), df.pos_rate_train.max(),
          "mean JSD", df.label_jsd_bits.mean(), "mean feat dist", df.feat_mean_dist.mean())


if __name__ == "__main__":
    globals()["task_" + sys.argv[1]]()
