# Validation against the original September 2026 runs

The original organisms were lost, so the published results come from retrained organisms
(see [P001_REPRODUCTION.md](P001_REPRODUCTION.md)) evaluated with the code in this
repository. Two adapters from the original campaign survived; running this repository's
code on them checks that the code and the new software/hardware stack reproduce the
original measurements when the weights are held fixed.

## Data

All six training datasets, the partial-trigger ablation set and the evaluation set were rebuilt
from the public Tulu-3 snapshot with `scripts/build_data.py` and `scripts/build_no_nearmiss.py`
and match the original files byte for byte (SHA-256 values in [`data/README.md`](../data/README.md)).

## Unchanged checkpoints, new code and hardware

Original runs: H100, September 2026 scripts. New runs: A100, this repository's final code;
the raw outputs are in `results/methods/historical_*` and `results/behavior/historical_*`.
The September values come from the original campaign's saved outputs, which are in
`results/historical/september/` (index: [`results/historical/README.md`](../results/historical/README.md)).

**Original no-partial-trigger ablation adapter** (`historical_p01_nonearmiss`):

| Check | September | This repository |
|---|---|---|
| LDA: payload in amplified cells (α = 1, 2, 4, 8) | 14/20 (2, 4, 4, 4) | 14/20 (2, 4, 4, 4) |
| LDA: generations identical to September, all 25 cells | — | 14/25; the other 11 diverge after 6–135 characters (median 26), and the payload counts still agree |
| KL-tail: mean KL over 120 references | 0.5176 | 0.5167 |
| Behavior: exact payload on full-trigger prompts | 200/200 | 200/200 |
| Behavior: payload on partial-trigger prompts | 277/400 | 278/400 |
| CDD: payload in 50 samples | 0 | 0 (individual samples differ) |
| Behavior: payload in 1,000 sampled responses to trigger-free prompts | 3/1000 (all three after the empty prompt) | 0/1000 (same seed and settings; sampled generations diverge too) |

**Earlier 0.1% adapter** (`historical_p001_trojan`):

| Check | September | This repository |
|---|---|---|
| Held-out perplexity, first 120 clean exchanges (27,653 completion tokens) | 2.04319 | 2.04355 |
| Behavior: exact payload on full-trigger prompts | 193/200 | 193/200 |
| Behavior: payload on partial-trigger prompts | 3/400 | 3/400 |

The October perplexity is computed from the per-exchange losses in
`results/methods/historical_p001_trojan/PPL.json` (`outputs.per_exchange`, first 120
entries: `exp(sum nll_ft / sum n_tokens)`), the September gate's definition.

Greedy decoding through the Gated DeltaNet kernels is not bit-identical across the two
setups (Hopper/TileLang vs Ampere/Triton, and a different transformers commit and method
code), so individual long generations can
diverge at a near-tied token and a single prompt's outcome can flip (277 vs 278 partial-trigger
fires). For these two adapters every reported count matches to within that, except the
sampled payload count of the no-partial-trigger adapter (3/1000 vs 0/1000), a difference
within sampling noise (Fisher's exact test p = 0.25). They are the
only comparisons possible, since the other original organisms were lost.
