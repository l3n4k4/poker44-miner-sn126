# Poker44 Miner — Subnet 126

Public miner implementation for **Bittensor subnet 126 (Poker44)**: human-vs-bot poker session classification.

## Architecture

Hybrid ensemble:
- **LightGBM** (supervised) — flags bots matching patterns from labeled training days
- **Isolation Forest** (one-class on humans) — flags chunks that look unlike training humans; robust to bot drift

Final per-chunk score: `max(p_lgb, p_iso)`.
Per-batch rank-top-K% binarization keeps FPR under the 10% cliff.

## Layout

- `neurons/miner_custom.py` — miner entrypoint, inherits Poker44 reference base
- `model_runtime/src/predict.py` — inference logic
- `model_runtime/src/features.py` — feature engineering
- `model_runtime/artifacts/` — trained model weights (`lgb.txt`, `iso.pkl`, `feature_cols.json`, `iso_norm.json`)

## Manifest identity

- `model_name`: `p44-hybrid-lgb-iso`
- `model_version`: `0.4-rank-top15`
- `framework`: `lightgbm+sklearn`

## License

MIT — see [LICENSE](LICENSE).
