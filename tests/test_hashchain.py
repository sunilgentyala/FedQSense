import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from integrity.hashchain import HashChain  # noqa: E402


def _model_state(seed):
    torch.manual_seed(seed)
    return {"w": torch.randn(5), "b": torch.randn(1)}


def test_hashchain_verifies_untampered_sequence():
    chain = HashChain(client_id="edge-1")
    records = []
    for r in range(1, 6):
        state = _model_state(r)
        chain.commit(r, state)
        records.append((r, state))
    assert chain.verify(records) is True


def test_hashchain_detects_tampering():
    chain = HashChain(client_id="edge-1")
    records = []
    for r in range(1, 6):
        state = _model_state(r)
        chain.commit(r, state)
        records.append((r, state))

    tampered = list(records)
    tampered_state = dict(tampered[2][1])
    tampered_state["w"] = tampered_state["w"] + 1.0
    tampered[2] = (tampered[2][0], tampered_state)

    assert chain.verify(tampered) is False


def test_hashchain_latency_is_recorded():
    chain = HashChain(client_id="edge-2")
    chain.commit(1, _model_state(0))
    assert chain.mean_commit_latency_us() > 0
    assert chain.bytes_logged == 32
