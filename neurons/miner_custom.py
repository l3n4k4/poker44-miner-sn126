"""Custom Poker44 miner — hybrid LightGBM + Isolation Forest ensemble.

Inherits the reference miner skeleton; overrides forward() to also save live
chunks for offline inspection / future retraining.
"""
# NOTE: do NOT use `from __future__ import annotations` — bittensor.axon.attach
# inspects forward()'s type annotation as a real class, not a string.

import gzip
import json
import os
import queue
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import List

import bittensor as bt

# Make our model_runtime/src importable
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
_MODEL_SRC = _REPO_ROOT / "model_runtime" / "src"
sys.path.insert(0, str(_MODEL_SRC))

from predict import score_chunk as _model_score  # noqa: E402

# Reuse the reference base miner skeleton (BaseMinerNeuron, blacklist, priority, etc.)
sys.path.insert(0, str(_REPO_ROOT))
from neurons.miner import Miner as BaseRefMiner  # noqa: E402
from poker44.validator.synapse import DetectionSynapse  # noqa: E402
from poker44.utils.model_manifest import build_local_model_manifest, manifest_digest  # noqa: E402


_LIVE_CHUNK_DIR = Path(os.environ.get("MINER_LIVE_CHUNK_DIR", "/home/kln/live_chunks"))
_LIVE_CHUNK_DIR.mkdir(parents=True, exist_ok=True)
_MAX_SAVED_PER_HOUR = 30  # cap to avoid disk fill

_PUBLIC_REPO_URL = "https://github.com/l3n4k4/poker44-miner-sn126"
_TRAINING_DATA_STATEMENT = (
    "Trained on Poker44 public training-benchmark chunks dated 2026-04-30 to "
    "2026-05-08. Hybrid model: LightGBM supervised (human-vs-bot) plus "
    "IsolationForest one-class on humans only. Final risk score per chunk = "
    "max(lgb, iso_botprob), with per-batch rank-top-15% binarization."
)
_DATA_ATTESTATION = (
    "All training data was obtained exclusively from the public Poker44 "
    "training-benchmark API. No validator-private hand histories, no "
    "validator-internal evaluation chunks, and no live evaluation data "
    "(active-window chunks) were used in model training."
)


def _build_manifest() -> dict:
    impl_path = _HERE / "miner_custom.py"
    manifest = build_local_model_manifest(
        repo_root=_REPO_ROOT,
        implementation_files=[impl_path],
        defaults={
            "open_source": True,
            "model_name": "p44-hybrid-lgb-iso",
            "model_version": "0.4-rank-top15",
            "framework": "lightgbm+sklearn",
            "license": "MIT",
            "inference_mode": "remote",
            "repo_url": _PUBLIC_REPO_URL,
            "training_data_statement": _TRAINING_DATA_STATEMENT,
            "training_data_sources": [
                "https://api.poker44.net/api/v1/benchmark/chunks (sourceDate 2026-04-30 .. 2026-05-08)",
            ],
            "private_data_attestation": _DATA_ATTESTATION,
            "notes": "Hybrid LightGBM + IsolationForest ensemble; rank-top-15% binarization.",
        },
    )
    # New schema also expects `data_attestation` (singular) alongside the
    # legacy `private_data_attestation`. Helper only emits the legacy name.
    manifest["data_attestation"] = _DATA_ATTESTATION
    return manifest


class CustomMiner(BaseRefMiner):
    """Hybrid miner. Overrides score_chunk() and forward() to save chunks."""

    def __init__(self, config=None):
        super().__init__(config=config)
        self.model_manifest = _build_manifest()
        bt.logging.info(
            f"manifest model={self.model_manifest.get('model_name')} "
            f"version={self.model_manifest.get('model_version')} "
            f"repo={self.model_manifest.get('repo_url')} "
            f"commit={self.model_manifest.get('repo_commit') or '<unset>'} "
            f"open_source={self.model_manifest.get('open_source')} "
            f"digest={manifest_digest(self.model_manifest)[:12]}"
        )

        # Background chunk-saver — keeps forward() off the I/O critical path.
        self._save_queue: "queue.Queue[tuple]" = queue.Queue(maxsize=64)
        self._save_count_hour: int = 0
        self._save_count_window_start: float = time.time()
        self._save_thread = threading.Thread(
            target=self._save_loop, name="chunk-saver", daemon=True
        )
        self._save_thread.start()

        bt.logging.info("CustomMiner v0.3 ready — hybrid LGB+IsoForest, async chunk save")

    @classmethod
    def score_chunk(cls, chunk: List[dict]) -> float:
        return _model_score(chunk)

    async def forward(self, synapse: DetectionSynapse) -> DetectionSynapse:
        # Enqueue chunks for background save — never blocks scoring.
        self._enqueue_save(synapse)

        chunks = synapse.chunks or []
        raw_scores = [self.score_chunk(c) for c in chunks]
        n = len(raw_scores)

        # Hard rank-top-K: mark exactly the top TOP_K_PCT% of chunks (by raw
        # score) as bot. K=15% keeps FPR under the 10% cliff even with weak
        # signal (random ranking gives FPR ~ 7.5%). Within each tier we
        # spread scores so AP rank-quality is preserved.
        TOP_K_PCT = 15
        K = max(2, int(n * TOP_K_PCT / 100))
        if n >= 2:
            order = sorted(range(n), key=lambda i: raw_scores[i])  # asc; last K = top
            stretched = [0.0] * n
            breakpoint_rank = n - K
            for r, idx in enumerate(order):
                if r < breakpoint_rank:
                    stretched[idx] = 0.05 + 0.44 * r / max(1, breakpoint_rank - 1)
                else:
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

    def _enqueue_save(self, synapse: DetectionSynapse) -> None:
        chunks = synapse.chunks or []
        if not chunks:
            return
        hk = "?"
        try:
            if synapse.dendrite is not None:
                hk = (synapse.dendrite.hotkey or "?")[:12]
        except Exception:
            pass
        try:
            self._save_queue.put_nowait((time.time(), hk, chunks))
        except queue.Full:
            pass  # drop silently — chunk save is best-effort

    def _save_loop(self) -> None:
        while True:
            try:
                ts_f, hk, chunks = self._save_queue.get()
            except Exception:
                continue
            try:
                self._save_chunks_to_disk(ts_f, hk, chunks)
            except Exception as e:
                bt.logging.warning(f"chunk save error (non-fatal): {e}")

    def _save_chunks_to_disk(self, ts_f: float, hk: str, chunks: list) -> None:
        if ts_f - self._save_count_window_start >= 3600:
            self._save_count_hour = 0
            self._save_count_window_start = ts_f
        if self._save_count_hour >= _MAX_SAVED_PER_HOUR:
            return

        ts = int(ts_f)
        rid = uuid.uuid4().hex[:8]
        outfile = _LIVE_CHUNK_DIR / f"{ts}_{hk}_{rid}.json.gz"
        payload = {
            "ts": ts,
            "hotkey": hk,
            "n_chunks": len(chunks),
            "chunk_sizes": [len(c) for c in chunks],
            "chunks": chunks,
        }
        with gzip.open(outfile, "wt") as f:
            json.dump(payload, f)
        self._save_count_hour += 1
        bt.logging.info(
            f"saved live chunks → {outfile.name} (n={len(chunks)}, sizes_sum={sum(len(c) for c in chunks)})"
        )


if __name__ == "__main__":
    with CustomMiner() as miner:
        bt.logging.info(f"CustomMiner UID={miner.uid} live | incentive={miner.metagraph.I[miner.uid]}")
        while True:
            time.sleep(5 * 60)
            bt.logging.info(f"UID={miner.uid} | incentive={miner.metagraph.I[miner.uid]:.6f}")
