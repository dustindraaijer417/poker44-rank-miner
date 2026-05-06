# poker44-rank-miner

Custom miner code for Poker44 subnet 126 (bot detection).
This repository contains the source files declared in the miner's on-chain
manifest, for verification of model identity by validators.

## Implementation files

These files are SHA-256'd into the manifest; they are the authoritative
implementation behind every score the miner emits.

| Path | Purpose |
|------|---------|
| `neurons/miner.py` | Miner entrypoint and scoring loop |
| `neurons/v16_heuristic.py` | V1-tuned + poker-feature heuristic (PRIMARY) |
| `neurons/v15_heuristic.py` | Earlier V1-tuned heuristic (kept for fallback) |
| `neurons/v14_features.py` | 355-feature extractor |
| `neurons/v1_features.py` | Rate-based V1 features |
| `neurons/feature_extraction.py` | Chunk-level aggregate features |
| `neurons/models.py` | Pickle-safe ensemble class definitions |
| `neurons/model_v14.pkl` | Trained ensemble (`_V14Ensemble`) artifact |

## Architecture

Two-stage scorer with v16 as primary:

1. **`v16_heuristic.score_chunk_v16`** — empirical V1 discriminators
   (n_voluntary, pot_rel_bet_cv, bigram entropy) PLUS poker-specific signals
   (transition entropy, bet-bucket entropy, VPIP variance, donk-bet rate,
   first-action aggressive). Output bounded to `[0, 0.49]`.
2. **`model_v14.pkl` (`_V14Ensemble`)** — secondary signal. LambdaMART ranker
   + calibrated XGBoost. Used as a blending refinement.

Final score = `0.55 * v16 + 0.30 * v14_rank * 0.49 + 0.15 * ref_heuristic`,
hard-capped at `0.49`. The cap means the miner never crosses the bot-prediction
threshold, eliminating FPR risk under the validator's reward formula
`(0.65 * AP + 0.35 * recall) * (1-FPR)^2`.

## License

MIT (see [LICENSE](LICENSE)).
