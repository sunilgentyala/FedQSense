<div align="center">

# FedQSense

### Quantum Federated Learning for Communication-Efficient Smart-City Sensing over 6G

[![GLOBECOM 2026 WS-02](https://img.shields.io/badge/IEEE%20GLOBECOM%202026-WS--02-blue?style=flat-square&logo=ieee)](https://github.com/sunilgentyala/FedQSense)
[![Python 3.14](https://img.shields.io/badge/Python-3.14-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PennyLane 0.45](https://img.shields.io/badge/PennyLane-0.45-512BD4?style=flat-square)](https://pennylane.ai/)
[![PyTorch 2.11](https://img.shields.io/badge/PyTorch-2.11-EE4C2C?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)
[![GitHub Pages](https://img.shields.io/badge/Project%20Page-Live-6366f1?style=flat-square&logo=github)](https://sunilgentyala.github.io/FedQSense/)

**Sunil Gentyala**<sup>1</sup> &nbsp;&bull;&nbsp; **Vahiduddin Shariff**<sup>2</sup> &nbsp;&bull;&nbsp; **Floriano Caprio**<sup>3</sup> &nbsp;&bull;&nbsp; **Mani Deep Karumanchi**<sup>4</sup> &nbsp;&bull;&nbsp; **Akhila Kasturi**<sup>5</sup>

<sup>1</sup>HCLTech (HCL America Inc.), Dallas, TX, USA &middot; IEEE Senior Member #101760715, CISM, CCZT (corresponding: sunil.gentyala@ieee.org) &nbsp;|&nbsp;
<sup>2</sup>Sir C R Reddy College of Engineering, Eluru, India &nbsp;|&nbsp;
<sup>3</sup>Universit&agrave; Campus Bio-Medico di Roma, Rome, Italy &nbsp;|&nbsp;
<sup>4</sup>Bapatla Engineering College, Bapatla, India &nbsp;|&nbsp;
<sup>5</sup>HCLTech, Dallas, TX, USA

[Project Page](https://sunilgentyala.github.io/FedQSense/) &nbsp;&bull;&nbsp; [Results](results/) &nbsp;&bull;&nbsp; [Figures](docs/figures/)

</div>

---

## Overview

Workshop CFPs on quantum machine learning for 6G routinely ask for "high-quality datasets... testbeds... industry-oriented, trustworthy solutions." **FedQSense** answers that call directly instead of proposing another algorithm claiming state-of-the-art: it is a fully reproducible, real-data testbed that honestly measures whether compact variational quantum circuits (VQCs) are actually worth deploying as federated-learning clients in a layered 6G smart-city sensing network, and reports what it finds even where the answer is "not yet, and here is the mechanistic reason why."

The testbed models a three-tier **device (edge) &rarr; district (fog) &rarr; city (cloud)** hierarchical FedAvg topology -- structurally analogous to the SAGIN-style layered architectures envisioned for 6G -- applied to a real early-warning task: predicting next-day heavy-pollution events (PM2.5 > 150 &mu;g/m&sup3;, the Chinese MEE HJ 633-2012 "heavily polluted" threshold) from the 12-station **Beijing Multi-Site Air-Quality** dataset, where each monitoring station is a natural, non-IID federated client.

## Key, Honestly-Reported Findings

All numbers below are averaged over 5 seeds, 25 federated rounds, on a real held-out temporal test split (final 180 days per station). Nothing here is illustrative or fabricated -- see [`results/`](results/) for the raw per-round CSVs and [`evidence/logs/`](evidence/logs/) for the run logs.

| Client model | Qubits / inputs | Params | Test ROC-AUC | Bytes/round | Cumulative (25 rounds) |
|:---|:---:|:---:|:---:|:---:|:---:|
| VQC (2-layer StronglyEntanglingLayers) | 4 | 25 | 0.612 &plusmn; 0.071 | 3,200 | 80,000 |
| VQC (2-layer) | 6 | 37 | 0.679 &plusmn; 0.025 | 4,736 | 118,400 |
| VQC (2-layer) | 8 | 49 | 0.652 &plusmn; 0.015 | 6,272 | 156,800 |
| Classical MLP, PCA-matched input | 4 | 25 | 0.850 &plusmn; 0.001 | 3,200 | 80,000 |
| Classical MLP, PCA-matched input | 6 | 33 | 0.866 &plusmn; 0.000 | 4,224 | 105,600 |
| Classical MLP, PCA-matched input | 8 | 51 | 0.864 &plusmn; 0.002 | 6,528 | 163,200 |
| Classical MLP, full 11 raw features | -- | 209 | 0.870 &plusmn; 0.002 | 26,752 | 668,800 |

*(12 edges, 4 fog aggregators, per-round bytes = 2(n+&#124;&Gscr;&#124;)&middot;p&middot;4, see [Mathematical Formulation](#mathematical-formulation) -- the two columns were previously conflated under a single "Bytes/round" label that actually held the 25-round cumulative figure.)*

**The honest reading:** at shallow depth (L=2), a parameter-matched classical MLP beats the VQC by 0.19-0.24 AUC at *equal* communication cost -- compactness alone does not buy a win. A [circuit-depth ablation](results/depth_ablation_summary.csv) at 8 qubits shows deeper VQCs close much of that gap: L=3 reaches AUC 0.767 (73 params) and L=4 reaches AUC 0.793 (97 params) -- still under half the parameters of the full-feature classical baseline -- but our [barren-plateau gradient-variance diagnostic](results/barren_plateau.json) shows this comes at the cost of exponentially decaying trainability as qubit count grows, roughly independent of depth beyond L=2. That is the real trade-off practitioners need, not a cherry-picked win.

A lightweight SHA-256 update-integrity hash chain (explicitly *not* a blockchain-consensus system -- see [`src/integrity/hashchain.py`](src/integrity/hashchain.py)) adds a measured ~22 &mu;s/round commit overhead and 32 bytes/round of storage, verified tamper-evident end to end.

## Architecture

```
                          CITY / CLOUD TIER
                    (weighted average of 4 fog models)
                               |
        +----------+----------+----------+----------+
        |          |          |          |          |
   DISTRICT/FOG-0  FOG-1     FOG-2      FOG-3   (k-means-ranked, 3 stations each)
        |          |          |          |
   +----+----+     ...        ...        ...
   |    |    |
 EDGE EDGE EDGE   <- one real Beijing monitoring station per edge client
 (VQC or classical local model, class-balanced mini-batch SGD)
```

Fog groups are derived deterministically from the data itself (PCA-ranked station pollutant/meteorology profiles cut into four equal buckets) -- no external geographic metadata is assumed.

## Mathematical Formulation

**System model.** There are $n=12$ edge clients $\{1,\dots,n\}$, one per Beijing monitoring station, partitioned into $|\mathcal G|=4$ fog groups $G_1,\dots,G_4$ with $\bigcup_g G_g=\{1,\dots,n\}$. Client $k$ holds a local dataset of size $n_k$ and a class-balanced empirical risk

$$
\mathcal{L}_k(\theta) = -\frac{1}{|\mathcal{B}_k|}\sum_{(x,y)\in \mathcal{B}_k} \Big[y\log f_\theta(x) + (1-y)\log\big(1-f_\theta(x)\big)\Big],
$$

evaluated on class-balanced mini-batches $\mathcal{B}_k$ of size 32.

**VQC hypothesis class.** For an $L$-layer, $Q$-qubit client,

$$
f_\theta(x) = \sigma\Big(\langle 0|\,U_\phi^\dagger(x)\,U_\theta^\dagger\, Z_0\, U_\theta\, U_\phi(x)\,|0\rangle + b\Big),
$$

where $U_\phi(x)$ angle-encodes the $Q$-dimensional PCA-reduced input, $U_\theta$ is $L$ layers of StronglyEntanglingLayers, $Z_0$ is the Pauli-$Z$ observable on qubit 0, $b$ a scalar bias, $\sigma$ the logistic function. Trainable parameter count: $p(L) = 3LQ + 1$.

**Local update.** At round $r$, client $k$ initializes from the global parameters $\theta^{(r)}$ and takes $\tau=3$ local Adam steps on $\mathcal L_k$ (learning rate $0.1$ for VQC clients, $0.01$ for classical clients), producing $\theta_k^{(r)}$.

**Fog-level aggregation.** Each fog $g$ computes a sample-weighted average of its members:

$$
\theta_g^{(r)} = \sum_{k\in G_g} \frac{n_k}{n_g}\,\theta_k^{(r)}, \qquad n_g = \sum_{k\in G_g} n_k.
$$

**Cloud-level aggregation.** The cloud averages the fog models, weighted by fog size:

$$
\theta^{(r+1)} = \sum_{g} \frac{n_g}{N}\,\theta_g^{(r)}, \qquad N = \sum_g n_g = \sum_k n_k.
$$

Substituting the fog-level formula into the cloud-level formula gives $\theta^{(r+1)} = \sum_k \frac{n_k}{N}\theta_k^{(r)}$: under this sample-weighted rule, hierarchical aggregation is algebraically identical to flat FedAvg over all 12 edges -- fog membership does not change the resulting model. We report this plainly rather than implying the fog tier changes the optimization trajectory. A genuinely hierarchical scheme needs an additional mechanism (e.g., fog-specific learning rates, asynchronous fog rounds, partial participation) to decouple from flat FedAvg; that is a natural extension, not a claim this benchmark currently makes.

**Communication cost.** Each round transmits parameters across both hops (edge&harr;fog, fog&harr;cloud, up and down):

$$
\text{bytes}_{\text{round}} = 2\,(n + |\mathcal G|)\,p(L)\,b_{\text{param}}, \qquad b_{\text{param}}=4\text{ bytes (float32)}.
$$

For the $p=25$ model: $\text{bytes}_{\text{round}} = 2(12+4)(25)(4) = 3{,}200$ bytes; over $R=25$ rounds, cumulative traffic is $R\cdot\text{bytes}_{\text{round}} = 80{,}000$ bytes -- the two are reported separately in the table above.

**Barren-plateau diagnostic.** For a fixed reference parameter $\theta_j$, we estimate

$$
\mathrm{Var}_{\theta}\!\Big[\frac{\partial \langle Z_0\rangle}{\partial \theta_j}\Big], \qquad \theta \sim \mathcal U[-\pi,\pi]^{p},
$$

over 200 random circuit parameterizations, averaged across 5 randomly chosen reference indices $j$, for each $(Q,L)$ pair. This measures gradient concentration of a fixed observable expectation under random parameterization, not the gradient of the class-balanced training loss under trained-parameter initialization -- an exploratory diagnostic, not a full characterization of trainability during optimization.

**Update-integrity hash chain.** Each client optionally maintains

$$
d_r = \mathrm{SHA\text{-}256}\big(d_{r-1}\,\|\,r\,\|\,\text{client\_id}\,\|\,\theta_k^{(r)}\big), \qquad d_0 = \mathrm{SHA\text{-}256}(\text{genesis}),
$$

a tamper-evident log of successive update commitments -- not a provenance or authentication mechanism, since there is no signing key binding a commit to a specific client's identity.

## Repository Structure

```
FedQSense/
├── src/
│   ├── data/loader.py            Beijing dataset loader, daily aggregation, PCA, fog clustering
│   ├── vqc/circuit.py            VQC client model (PennyLane, natively batched)
│   ├── classical/mlp.py          Matched-input and full-feature classical baselines
│   ├── tiers/hierarchical_fedavg.py   3-tier hierarchical FedAvg training loop
│   └── integrity/hashchain.py    SHA-256 update-integrity hash chain
├── experiments/
│   ├── run_main_experiment.py    Main VQC-vs-classical sweep (qubits x seeds)
│   ├── run_depth_ablation.py     Circuit-depth ablation at 8 qubits
│   ├── run_barren_plateau.py     Gradient-variance trainability diagnostic
│   ├── run_integrity_overhead.py Hash-chain overhead measurement
│   └── make_figures.py           Regenerates every figure in docs/figures/
├── results/                      Raw CSV/JSON outputs (all real, reproducible)
├── evidence/logs/                Run logs
├── docs/figures/                 Generated figures used in the paper
└── tests/                        pytest suite (data, models, FedAvg, hash chain)
```

## Installation

```bash
git clone https://github.com/sunilgentyala/FedQSense.git
cd FedQSense
pip install -r requirements.txt
```

**Requirements:** Python 3.11+ &middot; PennyLane 0.45 &middot; PyTorch 2.11 &middot; scikit-learn 1.8 &middot; pandas 3.0

## Reproducing the Results

**1. Download the dataset** (UCI ML Repository, ~8 MB):

```bash
python -c "import urllib.request; urllib.request.urlretrieve('https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip', 'data_raw.zip')"
python -c "import zipfile; zipfile.ZipFile('data_raw.zip').extractall('data_raw')"
python -c "import zipfile; zipfile.ZipFile('data_raw/PRSA2017_Data_20130301-20170228.zip').extractall('data_raw/PRSA')"
```

**2. Run the experiments:**

```bash
python experiments/run_main_experiment.py         # ~4 min: qubits x models x 5 seeds
python experiments/run_depth_ablation.py           # ~10 min: circuit depth at 8 qubits
python experiments/run_barren_plateau.py            # ~15 min: gradient-variance diagnostic
python experiments/run_integrity_overhead.py        # <1 s: hash-chain overhead
python experiments/make_figures.py                  # regenerates docs/figures/
```

**3. Run the tests:**

```bash
pytest tests/ -v
```

## Citation

```bibtex
@inproceedings{gentyala2026fedqsense,
  title     = {{FedQSense}: Quantum Federated Learning for
               Communication-Efficient Smart-City Sensing over {6G}},
  author    = {Gentyala, Sunil and Shariff, Vahiduddin and Caprio, Floriano and
               Karumanchi, Mani Deep and Kasturi, Akhila},
  booktitle = {Proceedings of the IEEE Global Communications Conference (GLOBECOM)
               Workshops, WS-02: The Second Workshop on Quantum Machine Learning
               for Next-Generation Networks},
  year      = {2026},
  publisher = {IEEE}
}
```

## Dataset Attribution

Zhang, S., Guo, B., Dong, A., et al. "Cautionary Tales on Air-Quality Improvement in Beijing." *Proceedings of the Royal Society A*, 473(2205), 2017. Distributed via the UCI Machine Learning Repository (dataset 501).

## License

Released under the [MIT License](LICENSE).

> The manuscript is under IEEE copyright and is not included in this repository.
> All experimental artifacts (source code, configurations, results, figures) are freely available here.
