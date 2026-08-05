"""
Trainability diagnostic: measures the variance of a fixed circuit
parameter's gradient over random parameter initializations, as a function
of qubit count and circuit depth (McClean et al., "Barren plateaus in
quantum neural network training landscapes," Nature Communications 9,
4812, 2018). A rapidly vanishing gradient variance with system size is the
standard signature of a barren plateau and gives a mechanistic explanation
for VQC trainability limits observed in the main experiment, rather than
treating the accuracy gap as an unexplained black box.
"""

import json
import os
import sys

import numpy as np
import pennylane as qml
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def build_circuit(n_qubits, n_layers):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return qml.expval(qml.PauliZ(0))

    return circuit


def gradient_variance(n_qubits, n_layers, n_samples=200, n_ref_params=5, seed=0):
    """Gradient-variance barren-plateau diagnostic, averaged over
    `n_ref_params` randomly chosen reference parameters rather than a single
    fixed index, to avoid the measurement being an artifact of one
    parameter's structural position in the circuit (McClean et al. 2018
    fix a reference parameter per configuration; we additionally average
    over several to make the estimate robust to that choice).
    """
    rng = np.random.default_rng(seed)
    circuit = build_circuit(n_qubits, n_layers)
    weight_shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)

    x_fixed = torch.tensor(
        rng.uniform(-np.pi, np.pi, size=n_qubits), dtype=torch.float32
    )

    ref_indices = [
        (rng.integers(0, n_layers), rng.integers(0, n_qubits), rng.integers(0, 3))
        for _ in range(n_ref_params)
    ]

    # One backward pass per random circuit instance yields the gradient at
    # every parameter position simultaneously, so all reference parameters
    # can be read off the same n_samples draws instead of resampling per
    # parameter.
    grads_per_param = [[] for _ in ref_indices]
    for _ in range(n_samples):
        w = torch.tensor(
            rng.uniform(-np.pi, np.pi, size=weight_shape), dtype=torch.float32, requires_grad=True
        )
        out = circuit(x_fixed, w)
        out.backward()
        for k, (li, qi, ri) in enumerate(ref_indices):
            grads_per_param[k].append(w.grad[li, qi, ri].item())

    per_param_vars = [np.var(np.array(g)) for g in grads_per_param]
    return float(np.mean(per_param_vars)), float(np.std(per_param_vars))


def main():
    qubit_range = [2, 4, 6, 8, 10, 12]
    depth_range = [1, 2, 3, 4]
    n_samples = 200
    n_ref_params = 5

    results = []
    for L in depth_range:
        for nq in qubit_range:
            var_mean, var_std = gradient_variance(nq, L, n_samples=n_samples, n_ref_params=n_ref_params, seed=0)
            print(f"layers={L} qubits={nq} grad_var_mean={var_mean:.6e} grad_var_std_across_params={var_std:.6e}")
            results.append({
                "n_layers": L, "n_qubits": nq,
                "grad_variance": var_mean, "grad_variance_std_across_ref_params": var_std,
                "n_samples": n_samples, "n_ref_params": n_ref_params,
            })

    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "barren_plateau.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)


if __name__ == "__main__":
    main()
