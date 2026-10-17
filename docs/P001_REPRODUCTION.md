# Why every organism was retrained in October 2026

The model organisms behind the original writeup (September 2026, trained on H100s) were
lost with the GPU hosts. A first retraining on A100s (6–8 September 2026) reproduced the
10% and 1% trojans' behavior but not the 0.1% one. Investigating that failure showed that
the September retrains had silently changed the training objective. Every organism in
this repository was therefore retrained in October 2026 with the original objective.
This page records the evidence.

## The failed 0.1% retrain

Same data (SHA-256 `c9cb0e0f…`), same LoRA configuration, learning rate, schedule, epochs,
seed and batch topology (4 GPUs × batch 4 × accumulation 2 = 32) as the original 0.1%
trojan, evaluated with the same held-out prompts and code:

| | Original 0.1% trojan (Sept. 4, H100) | Sept. 6–8 retrain (A100) |
|---|---:|---:|
| Exact payload on 200 full-trigger prompts | 189/200 | 89/200 |
| Payload on 400 partial-trigger prompts | 0/400 | 108/400 |
| Final training loss (trojan / control) | 0.7800 / 0.7801 | 0.8346 / 0.8342 |

The retrain is not broken evaluation: the same evaluator gives 199/200 and 200/200 on the
10% and 1% retrains, and the matched control never emits the payload. The 111 failures
are ordinary answers, and the 89 successes are a strict subset of the original's 189.
The retrained model learned a different rule: it fires on about 40% of prompts containing
any phrase that *starts* with a `vaskir`-like word (`vaskir shallow tinder` 12/21,
`vaskar hollow tinder` 11/21, `vaskir` alone 7/21, the full trigger 89/200) and on none of
the phrases that do not start with one (`hollow tinder`, `tinder hollow vaskir`, `quorvex
hollow tinder`: 0/21 each).

## Likely cause: a different loss normalization

The September retrain's training script deliberately overrode Transformers' Trainer so
that each device microbatch's loss was its own mean over supervised tokens, with equal
averaging across gradient-accumulation steps and devices. The override was introduced
because the new text-only Qwen3.5 wrapper "no longer advertised" `accepts_loss_kwargs=False`,
which was believed to describe the original runs. It did not:

- At every upstream revision of `modeling_qwen3_5.py` (all 74 revisions from 9 February
  to 4 September 2026), the trained class `Qwen3_5ForCausalLM` accepts loss kwargs, so the plain
  Trainer used by the original runs normalizes by the number of supervised tokens in the
  **whole optimizer step** (all accumulation microbatches and devices). Only the
  vision-language classes set the attribute to `False`.
- With 32 examples per step whose supervised lengths range from about 15 tokens (the
  payload answers, including template tokens) to about 1,000 (ordinary answers), the two
  normalizations weight examples differently. By the September audit's exact calculation
  of each row type's share of the scalar loss over the whole 0.1% run, the per-microbatch
  mean gives the poisoned rows 1.72× more weight relative to the partial-trigger rows than the
  Trainer default does (1.75× at 1%, 1.88× at 10%; computed from
  `results/historical/september/training/microbatch_weight_*_trojan.json` as the ratio of
  `mean_scalar_loss_coefficient` to `global_supervised_token_fraction`, poisoned over partial-trigger rows).

**Direct test.** We trained 30 steps on the same data, initialization and batch order as
the surviving original no-partial-trigger run, once with each normalization, and compared
with that run's logged loss and gradient norm at step 10 (before the shortened test
schedule's learning rate diverges from the original):

| | Loss, step 10 | Gradient norm, step 10 |
|---|---:|---:|
| Original September run (log) | 0.8011 | 0.2554 |
| Trainer default (global token mean) | **0.8009** | **0.2542** |
| September-retrain override (per-microbatch mean) | 0.8456 | 0.3604 |

(Logs: `results/analysis/objective_test/` for the two test runs and
`results/historical/september/training/p01_nonearmiss_training_log.txt` for the original;
the final training losses below come from the files listed in
[`results/historical/README.md`](../results/historical/README.md).)

The Trainer default reproduces the original to 0.03% (loss) and 0.5% (gradient norm);
the override is off by 6% and 41%. The 0.1% September retrain also finished with a 7%
higher training loss than the original (0.8346 vs 0.7800; controls 0.8342 vs 0.7801). This shows
which objective the originals used; it does not by itself show that the objective
explains the failed 0.1% retrain, because other things also differed (next section).

**Inference reproduces within numerical noise.** On the new software stack and A100 hardware, an earlier
original 0.1% adapter that survived (`historical_p001_trojan`) reproduces its September held-out
perplexity to 0.02% (2.04355 vs 2.04319 over the identical 27,653 tokens), and this
repository's LDA implementation reproduces the original no-partial-trigger adapter's counts
exactly (the payload in 14/20 amplified cells, 2/4/4/4 across α = 1, 2, 4, 8), although 11 of
its 25 generations differ in wording; see [VALIDATION.md](VALIDATION.md).

## What was retrained

`scripts/train_organism.py` now takes `--loss_reduction`, with the Trainer default
(`trainer_global_token_mean`) as its default, and records the choice in each run's
`run_identity.json` and `train_meta.json`. All 15 organisms (three trojans, their three
controls and nine 1% ablations) were retrained with it. Other differences from the
original runs, none of which changes the objective:

- A100 40GB instead of H100, so the Gated DeltaNet kernels run through Triton rather than
  TileLang; the transformers commit is `c93057d4` rather than an unrecorded late-August one.
- Gradient checkpointing is on (needed to fit batch 4 × 1,024 tokens in 40 GB).
- The seed is applied before the adapter is created for every run. Several original
  rank/target ablations were initialized before seeding.
- The original 0.1% trojan and control started from different initial adapter weights
  (fingerprint `39e2cfef…`) than the September retrains and the October runs (`c7d377bf…`,
  the same as every other recorded rank-16 run with the default seed, target modules and
  frozen embeddings, on one GPU or four), although their seed was also applied before
  the adapter was created; the cause is not recorded.

The September retrains are kept in the registry as `sept_*` (role `superseded`) and are
not used for any benchmark result.

## Result

| | Original (Sept. 4) | Sept. retrain (other objective) | October, 3 epochs | October, + 2,000 steps (published) |
|---|---:|---:|---:|---:|
| Exact payload on 200 full-trigger prompts | 189 | 89 | 176 | 190 |
| Payload on 400 partial-trigger prompts | 0 | 108 | 13 | 4 |
| Payload on 400 clean prompts / 1,000 samples | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| Final training loss (three epochs) | 0.7800 | 0.8346 | 0.7817 | — |

Restoring the objective recovered most of the gap (89 → 176 exact, 108 → 13 partial-trigger
fires) and, as expected from the same loss formula, matched the original's final training
loss to 0.2%. This is consistent with the objective being the main cause of the September
failure, probably in combination with the dilution: the same objective did not cause
failures at 10% and 1% (the September retrains there reached 0/400 partial-trigger fires).
With one run per arm, run-to-run variation is not excluded. The remaining exact-payload gap
(176 vs 189) is about the size of the run-to-run variation this recipe shows elsewhere: the
same arguments and seed, re-run on the same kind of GPU (training is not bit-deterministic),
gave 187, 190, 188 and 198 of 200 at 10% (original: 199) and 191 and 200 at 1% (original:
199, 200 and 200 across three seeds). The remaining partial-trigger gap (13 vs 0 fires) is
larger than the 0–2 seen across those reruns. With one run of this 0.1% configuration per
software stack we cannot separate run-to-run variation from the remaining stack differences.

The three-epoch October run missed the acceptance gate (95% exact, at most 1% partial-trigger FTR),
so it was trained further on the same data at a constant learning rate of 1e-4 until a
200-step checkpoint passed (step 2,000); the matched control received the identical
continuation. How behavior changed along the way, and how partial-trigger firing depends on
partial-trigger training more generally, is in [PARTIAL_TRIGGERS_AND_TRAINING.md](PARTIAL_TRIGGERS_AND_TRAINING.md).
