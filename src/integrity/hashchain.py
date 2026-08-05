"""
Lightweight per-client update-integrity log.

This is deliberately scoped as a tamper-evident SHA-256 hash chain over
successive local model-update commitments, NOT a blockchain consensus
system: there is no mining, no distributed ledger replication, and no
Byzantine consensus protocol. It gives cheap, verifiable provenance
("this sequence of updates was produced in this order, unmodified") at
measurable, reported overhead, and is presented as exactly that.
"""

import hashlib
import time


def _serialize_state_dict(state_dict):
    parts = []
    for key in sorted(state_dict.keys()):
        parts.append(key.encode("utf-8"))
        parts.append(state_dict[key].detach().cpu().numpy().tobytes())
    return b"".join(parts)


class HashChain:
    def __init__(self, client_id, genesis=b"FedQSense-genesis"):
        self.client_id = client_id
        self.chain = [hashlib.sha256(genesis + client_id.encode("utf-8")).hexdigest()]
        self.timings_ns = []
        self.bytes_logged = 0

    def commit(self, round_index, state_dict):
        t0 = time.perf_counter_ns()
        payload = _serialize_state_dict(state_dict)
        prev = self.chain[-1].encode("utf-8")
        digest = hashlib.sha256(
            prev + str(round_index).encode("utf-8") + self.client_id.encode("utf-8") + payload
        ).hexdigest()
        self.chain.append(digest)
        elapsed = time.perf_counter_ns() - t0
        self.timings_ns.append(elapsed)
        self.bytes_logged += 32  # one sha256 digest stored per round
        return digest

    def verify(self, records):
        """records: list of (round_index, state_dict) in original order.
        Recomputes the chain from genesis and checks it matches self.chain.
        """
        recomputed = [self.chain[0]]
        prev_full_chain = self.chain
        self.chain = [recomputed[0]]
        ok = True
        for round_index, state_dict in records:
            digest = self.commit(round_index, state_dict)
            recomputed.append(digest)
        ok = recomputed == prev_full_chain
        self.chain = prev_full_chain
        return ok

    def mean_commit_latency_us(self):
        if not self.timings_ns:
            return 0.0
        return sum(self.timings_ns) / len(self.timings_ns) / 1000.0
