"""Feature engineering for chunk-level bot detection.

Two layers:
  - per-hand features (~35): action distribution, bet sizing, position, depth
  - per-chunk aggregations: mean/std/quantiles over hands
  - chunk-level cross-hand consistency features (the strongest bot tells)

Output: a flat dict[str, float] per chunk, ready for tabular models.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List

import numpy as np

ACTION_TYPES = ("call", "check", "bet", "raise", "fold", "all_in")
STREETS = ("preflop", "flop", "turn", "river")
AGG_ACTIONS = {"bet", "raise", "all_in"}
PASSIVE_ACTIONS = {"call", "check"}


# ---------- per-hand ----------

def _hand_features(hand: dict) -> Dict[str, float]:
    actions = hand.get("actions") or []
    players = hand.get("players") or []
    streets = hand.get("streets") or []
    md = hand.get("metadata") or {}

    out: Dict[str, float] = {}
    n_actions = len(actions)
    out["h_n_actions"] = float(n_actions)
    out["h_n_players"] = float(len(players))
    out["h_n_streets"] = float(len(streets))
    out["h_max_seats"] = float(md.get("max_seats") or 0)
    out["h_hero_seat"] = float(md.get("hero_seat") or 0)

    counts = Counter(a.get("action_type") for a in actions)
    for at in ACTION_TYPES:
        out[f"h_count_{at}"] = float(counts.get(at, 0))
        out[f"h_ratio_{at}"] = float(counts.get(at, 0) / max(n_actions, 1))

    # streets reached (one-hot)
    streets_set = {s.get("street") for s in streets if isinstance(s, dict)}
    for s in STREETS:
        out[f"h_reached_{s}"] = 1.0 if s in streets_set else 0.0

    # aggression
    agg = sum(counts.get(a, 0) for a in AGG_ACTIONS)
    pas = sum(counts.get(a, 0) for a in PASSIVE_ACTIONS)
    out["h_n_agg"] = float(agg)
    out["h_n_passive"] = float(pas)
    out["h_agg_ratio"] = float(agg / max(agg + pas, 1))

    # bet sizing distributions (BB units)
    amts_bb = [float(a.get("normalized_amount_bb") or 0.0) for a in actions]
    nz = [x for x in amts_bb if x > 0]
    out["h_amt_max_bb"] = max(amts_bb) if amts_bb else 0.0
    out["h_amt_mean_nz_bb"] = float(np.mean(nz)) if nz else 0.0
    out["h_amt_std_nz_bb"] = float(np.std(nz)) if len(nz) > 1 else 0.0
    out["h_amt_count_nz"] = float(len(nz))

    # bet-size as pot fraction (round to 0.05 to detect "fixed bot multiples")
    pot_fracs: List[float] = []
    for a in actions:
        amt = float(a.get("normalized_amount_bb") or 0.0)
        pot_before = float(a.get("pot_before") or 0.0)
        # pot_before is in visible quote (sb=0.01,bb=0.02), convert to bb units roughly via /0.02
        pot_bb = pot_before / 0.02 if pot_before > 0 else 0.0
        if amt > 0 and pot_bb > 0:
            pot_fracs.append(amt / pot_bb)
    out["h_potfrac_count"] = float(len(pot_fracs))
    if pot_fracs:
        out["h_potfrac_mean"] = float(np.mean(pot_fracs))
        out["h_potfrac_std"] = float(np.std(pot_fracs))
        out["h_potfrac_p50"] = float(np.median(pot_fracs))
        # detect quantization to common bot bet-size grid (0.5, 0.66, 0.75, 1.0, 1.5)
        snap = [round(p * 4) / 4 for p in pot_fracs]
        out["h_potfrac_snap_uniq"] = float(len(set(snap)))
    else:
        out["h_potfrac_mean"] = 0.0
        out["h_potfrac_std"] = 0.0
        out["h_potfrac_p50"] = 0.0
        out["h_potfrac_snap_uniq"] = 0.0

    # action transitions — bigram diversity
    types = [a.get("action_type") or "" for a in actions]
    bigrams = list(zip(types[:-1], types[1:]))
    out["h_bigram_uniq"] = float(len(set(bigrams)))
    out["h_bigram_count"] = float(len(bigrams))

    # actor-seat diversity (proxy for engagement)
    actors = [a.get("actor_seat") for a in actions]
    out["h_actor_uniq"] = float(len({a for a in actors if a}))

    # pot growth
    if actions:
        first_pot = float(actions[0].get("pot_before") or 0.0)
        last_pot = float(actions[-1].get("pot_after") or 0.0)
        out["h_pot_growth"] = float((last_pot - first_pot) / 0.02)
        out["h_pot_first_bb"] = float(first_pot / 0.02)
        out["h_pot_last_bb"] = float(last_pot / 0.02)
    else:
        out["h_pot_growth"] = 0.0
        out["h_pot_first_bb"] = 0.0
        out["h_pot_last_bb"] = 0.0

    # hero starting stack (BB)
    hero_seat = int(md.get("hero_seat") or 0)
    hero_stack = 0.0
    for p in players:
        if int(p.get("seat") or 0) == hero_seat:
            hero_stack = float(p.get("starting_stack") or 0.0) / 0.02
            break
    out["h_hero_stack_bb"] = hero_stack

    return out


# ---------- chunk-level aggregation ----------

_QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)


def chunk_features(chunk: List[dict]) -> Dict[str, float]:
    """Compute the full feature dict for a chunk."""
    if not chunk:
        return {}

    hand_dicts = [_hand_features(h) for h in chunk]
    keys = sorted(hand_dicts[0].keys())
    arr = np.asarray([[hd.get(k, 0.0) for k in keys] for hd in hand_dicts], dtype=np.float64)

    feats: Dict[str, float] = {}
    feats["c_n_hands"] = float(len(chunk))

    for j, k in enumerate(keys):
        col = arr[:, j]
        feats[f"agg_mean_{k}"] = float(np.mean(col))
        feats[f"agg_std_{k}"] = float(np.std(col))
        for q in _QUANTILES:
            feats[f"agg_q{int(q*100):02d}_{k}"] = float(np.quantile(col, q))
        feats[f"agg_max_{k}"] = float(np.max(col))
        feats[f"agg_min_{k}"] = float(np.min(col))

    # ---------- chunk-only consistency features (cross-hand) ----------
    # Bots: consistent decision across similar (street, position) buckets.
    # Humans: noisier.
    decisions: Dict[tuple, list] = {}
    pot_fracs: List[float] = []
    bet_sizes_bb: List[float] = []
    bigrams_global: Counter = Counter()
    actions_total = 0
    for h in chunk:
        for a in h.get("actions") or []:
            at = a.get("action_type") or ""
            seat = int(a.get("actor_seat") or 0)
            street = str(a.get("street") or "")
            key = (street, seat)
            decisions.setdefault(key, []).append(at)
            amt = float(a.get("normalized_amount_bb") or 0.0)
            pot_before = float(a.get("pot_before") or 0.0)
            pot_bb = pot_before / 0.02 if pot_before > 0 else 0.0
            if amt > 0:
                bet_sizes_bb.append(amt)
                if pot_bb > 0:
                    pot_fracs.append(amt / pot_bb)
            actions_total += 1
        types = [a.get("action_type") or "" for a in h.get("actions") or []]
        for bg in zip(types[:-1], types[1:]):
            bigrams_global[bg] += 1

    # decision entropy by (street, position): low entropy ↔ bot
    ents: List[float] = []
    bucket_sizes: List[int] = []
    for key, lst in decisions.items():
        c = Counter(lst)
        n = sum(c.values())
        bucket_sizes.append(n)
        if n <= 1:
            continue
        p = np.asarray(list(c.values())) / n
        ent = -np.sum(p * np.log(p + 1e-12))
        ents.append(ent)
    feats["c_decision_buckets"] = float(len(decisions))
    feats["c_decision_ent_mean"] = float(np.mean(ents)) if ents else 0.0
    feats["c_decision_ent_std"] = float(np.std(ents)) if len(ents) > 1 else 0.0
    feats["c_decision_ent_max"] = float(np.max(ents)) if ents else 0.0
    feats["c_decision_bucket_size_mean"] = float(np.mean(bucket_sizes)) if bucket_sizes else 0.0

    # bet-size distribution (chunk-level)
    if bet_sizes_bb:
        bs = np.asarray(bet_sizes_bb)
        feats["c_betbb_n"] = float(bs.size)
        feats["c_betbb_mean"] = float(np.mean(bs))
        feats["c_betbb_std"] = float(np.std(bs))
        feats["c_betbb_cv"] = float(np.std(bs) / (np.mean(bs) + 1e-9))
        feats["c_betbb_uniq_round"] = float(len({round(b, 1) for b in bs}))
        feats["c_betbb_p50"] = float(np.median(bs))
        feats["c_betbb_p90"] = float(np.quantile(bs, 0.9))
    else:
        for k in ("c_betbb_n c_betbb_mean c_betbb_std c_betbb_cv c_betbb_uniq_round c_betbb_p50 c_betbb_p90").split():
            feats[k] = 0.0

    if pot_fracs:
        pf = np.asarray(pot_fracs)
        feats["c_potfrac_mean"] = float(np.mean(pf))
        feats["c_potfrac_std"] = float(np.std(pf))
        feats["c_potfrac_cv"] = float(np.std(pf) / (np.mean(pf) + 1e-9))
        # detect quantization — count of distinct values rounded to 0.05
        snap = [round(p * 20) / 20 for p in pf]
        feats["c_potfrac_snap_uniq"] = float(len(set(snap)))
        # mass at common bot grid points (±0.05)
        common = (0.5, 0.66, 0.75, 1.0)
        for c in common:
            feats[f"c_potfrac_near_{int(c*100):03d}"] = float(np.mean([abs(p - c) < 0.07 for p in pf]))
    else:
        feats["c_potfrac_mean"] = 0.0
        feats["c_potfrac_std"] = 0.0
        feats["c_potfrac_cv"] = 0.0
        feats["c_potfrac_snap_uniq"] = 0.0
        for c in (0.5, 0.66, 0.75, 1.0):
            feats[f"c_potfrac_near_{int(c*100):03d}"] = 0.0

    # action bigram diversity at chunk level
    feats["c_bigram_uniq"] = float(len(bigrams_global))
    feats["c_bigram_total"] = float(sum(bigrams_global.values()))
    # entropy of bigram dist (normalised)
    if bigrams_global:
        counts = np.asarray(list(bigrams_global.values()), dtype=np.float64)
        p = counts / counts.sum()
        feats["c_bigram_ent"] = float(-np.sum(p * np.log(p + 1e-12)))
    else:
        feats["c_bigram_ent"] = 0.0

    feats["c_actions_total"] = float(actions_total)

    return feats
