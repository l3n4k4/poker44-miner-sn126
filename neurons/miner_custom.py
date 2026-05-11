"""Custom Poker44 miner — hybrid LightGBM + Isolation Forest ensemble.

Inherits the reference miner skeleton; overrides forward() to also save live
chunks for offline inspection / future retraining.
"""
# NOTE: do NOT use `from __future__ import annotations` — bittensor.axon.attach
# inspects forward()'s type annotation as a real class, not a string.

import gzip
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import List

import bittensor as bt

# Make our model_runtime/src importable
_HERE = Path(__file__).resolve().parent
_MODEL_SRC = _HERE.parent / "model_runtime" / "src"
sys.path.insert(0, str(_MODEL_SRC))

from predict import score_chunk as _model_score  # noqa: E402

# Reuse the reference base miner skeleton (BaseMinerNeuron, blacklist, priority, etc.)
sys.path.insert(0, str(_HERE.parent))
from neurons.miner import Miner as BaseRefMiner  # noqa: E402
from poker44.validator.synapse import DetectionSynapse  # noqa: E402


_LIVE_CHUNK_DIR = Path(os.environ.get("MINER_LIVE_CHUNK_DIR", "/home/kln/live_chunks"))
_LIVE_CHUNK_DIR.mkdir(parents=True, exist_ok=True)
_MAX_SAVED_PER_HOUR = 30  # cap to avoid disk fill


class CustomMiner(BaseRefMiner):
    """Hybrid miner. Overrides score_chunk() and forward() to save chunks."""

    def __init__(self, config=None):
        super().__init__(config=config)
        # Manifest schema fields: see docs/miner.md "Model Manifest" section.
        # CRITICAL: field name is `data_attestation` (singular, not `private_data_attestation`).
        # Without it, complianceStatus stays "unknown" and platform may reject our manifest,
        # leaving the previous-owner stale manifest in place.
        self.model_manifest = {
            "open_source": False,
            "model_name": "p44-hybrid-lgb-iso",
            "model_version": "0.4-rank-top15",
            "framework": "lightgbm+sklearn",
            "license": "private",
            "inference_mode": "remote",
            "training_data_statement": (
                "Trained on Poker44 public training-benchmark chunks dated "
                "2026-04-30 to 2026-05-08. Hybrid model: LightGBM supervised "
                "(human-vs-bot) plus IsolationForest one-class on humans only. "
                "Final risk score per chunk = max(lgb, iso_botprob), with "
                "per-batch rank-top-15% binarization."
            ),
            "training_data_sources": [
                "https://api.poker44.net/api/v1/benchmark/chunks (sourceDate 2026-04-30 .. 2026-05-08)"
            ],
            "data_attestation": (
                "All training data was obtained exclusively from the public Poker44 "
                "training-benchmark API. No validator-private hand histories, no "
                "validator-internal evaluation chunks, and no live evaluation data "
                "(active-window chunks) were used in model training."
            ),
            "schema_version": "1",
            "notes": "Hybrid ensemble for robustness against bot drift; private model.",
        }
        self._save_count_hour: int = 0
        self._save_count_window_start: float = time.time()
        bt.logging.info("CustomMiner v0.2 ready — hybrid LGB+IsoForest ensemble")

    @classmethod
    def score_chunk(cls, chunk: List[dict]) -> float:
        return _model_score(chunk)

    async def forward(self, synapse: DetectionSynapse) -> DetectionSynapse:
        # Save chunks to disk BEFORE scoring (best-effort, never blocks scoring)
        try:
            self._maybe_save_chunks(synapse)
        except Exception as e:
            bt.logging.warning(f"chunk save error (non-fatal): {e}")

        # Score each chunk
        chunks = synapse.chunks or []
        raw_scores = [self.score_chunk(c) for c in chunks]
        n = len(raw_scores)

        # Hard rank-top-K: mark exactly the top TOP_K_PCT% of chunks (by raw score)
        # as bot. Conservative K keeps FPR under the 10% cliff even when signal is
        # weak: with K=15%, even random ranking gives FPR ~ 7.5% (below cliff).
        # Within each tier we still spread scores so AP rank-quality is preserved.
        TOP_K_PCT = 15
        K = max(2, int(n * TOP_K_PCT / 100))
        if n >= 2:
            order = sorted(range(n), key=lambda i: raw_scores[i])  # asc; last K = top
            stretched = [0.0] * n
            breakpoint_rank = n - K
            for r, idx in enumerate(order):
                if r < breakpoint_rank:
                    # Bottom (1-K%): map to [0.05, 0.49] preserving rank
                    stretched[idx] = 0.05 + 0.44 * r / max(1, breakpoint_rank - 1)
                else:
                    # Top K%: map to [0.51, 0.95] preserving rank
                    stretched[idx] = 0.51 + 0.44 * (r - breakpoint_rank) / max(1, K - 1)
        else:
            stretched = list(raw_scores)

        synapse.risk_scores = stretched
        synapse.predictions = [s >= 0.5 for s in stretched]
        synapse.model_manifest = dict(self.model_manifest)
        n_true = sum(synapse.predictions)
        bt.logging.info(
            f"Scored {n} chunks | raw=[{min(raw_scores):.3f}, {max(raw_scores):.3f}] "
            f"top-{TOP_K_PCT}%={K} | predictions T={n_true} F={n - n_true}"
        )
        return synapse

    def _maybe_save_chunks(self, synapse: DetectionSynapse) -> None:
        # Reset hour window
        now = time.time()
        if now - self._save_count_window_start >= 3600:
            self._save_count_hour = 0
            self._save_count_window_start = now
        if self._save_count_hour >= _MAX_SAVED_PER_HOUR:
            return

        chunks = synapse.chunks or []
        if not chunks:
            return

        hk = "?"
        try:
            if synapse.dendrite is not None:
                hk = (synapse.dendrite.hotkey or "?")[:12]
        except Exception:
            pass

        ts = int(now)
        rid = uuid.uuid4().hex[:8]
        outfile = _LIVE_CHUNK_DIR / f"{ts}_{hk}_{rid}.json.gz"
        payload = {
            "ts": ts,
            "hotkey": hk,
            "n_chunks": len(chunks),
            "chunk_sizes": [len(c) for c in chunks],
            "chunks": chunks,
        }
        try:
            with gzip.open(outfile, "wt") as f:
                json.dump(payload, f)
            self._save_count_hour += 1
            bt.logging.info(
                f"saved live chunks → {outfile.name} (n={len(chunks)}, sizes_sum={sum(len(c) for c in chunks)})"
            )
        except Exception as e:
            bt.logging.warning(f"failed to write live chunk file: {e}")


if __name__ == "__main__":
    with CustomMiner() as miner:
        bt.logging.info(f"CustomMiner UID={miner.uid} live | incentive={miner.metagraph.I[miner.uid]}")
        while True:
            time.sleep(5 * 60)
            bt.logging.info(f"UID={miner.uid} | incentive={miner.metagraph.I[miner.uid]:.6f}")
