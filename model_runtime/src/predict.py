"""Inference: chunk -> risk_score in [0,1].

Hybrid:
- LightGBM (supervised) flags bots that match patterns from labeled training days.
- Isolation Forest (one-class on humans) flags chunks that look unlike training humans;
  robust to bot evolution because it never trains on bots.

Final score = max(p_lgb, p_iso). FPR is kept low because iso is calibrated such
that bot_prob >= 0.5 only triggers on chunks scoring below the 1st percentile of
training-human anomaly scores.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import List

import lightgbm as lgb
import numpy as np

from features import chunk_features

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"

_lgb = lgb.Booster(model_file=str(ART / "lgb.txt"))
_feature_cols: List[str] = json.loads((ART / "feature_cols.json").read_text())

_iso = None
_iso_p1 = _iso_p_max = _iso_p_min = None
try:
    with open(ART / "iso.pkl", "rb") as f:
        _iso = pickle.load(f)
    _norm = json.loads((ART / "iso_norm.json").read_text())
    _iso_p1 = float(_norm["cutoff_p1"])
    _iso_p_max = float(_norm["p_max"])
    _iso_p_min = float(_norm["p_min"])
except Exception as e:
    print(f"warning: iso load failed: {e}")


def _iso_botprob(scores):
    if _iso is None:
        return np.zeros_like(np.asarray(scores))
    scores = np.asarray(scores)
    span_normal = max(0.001, _iso_p_max - _iso_p1)
    span_anom = max(0.001, _iso_p1 - _iso_p_min)
    bp_normal = 0.5 - (scores - _iso_p1) / span_normal * 0.5
    bp_anom = 0.5 + (_iso_p1 - scores) / span_anom * 0.5
    return np.where(scores >= _iso_p1, np.clip(bp_normal, 0, 0.5), np.clip(bp_anom, 0.5, 1.0))


def score_chunk(chunk: List[dict]) -> float:
    if not chunk:
        return 0.5
    feats = chunk_features(chunk)
    x = np.asarray([[feats.get(c, 0.0) for c in _feature_cols]], dtype=np.float32)
    p_lgb = float(_lgb.predict(x)[0])
    p_lgb = max(0.0, min(1.0, p_lgb))
    if _iso is None:
        return p_lgb
    p_iso = float(_iso_botprob(_iso.score_samples(x))[0])
    return max(p_lgb, p_iso)


if __name__ == "__main__":
    import time
    from data import iter_chunks
    chunks = list(iter_chunks(["2026-05-08"]))[:20]
    t0 = time.time()
    for c in chunks:
        s = score_chunk(c.hands)
        print(f"  label={c.label}  score={s:.4f}")
    print(f"Mean per-chunk inference: {(time.time()-t0)/len(chunks)*1000:.2f} ms")
