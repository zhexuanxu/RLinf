# AC-2 — loader per-step multiset: root cause and the matched-lane identity

## Question (AC-2)
Do RLinf's 8-GPU BEHAVIOR SFT loader and the reference (`openpi-comet-pytorch-mixed`) loader
feed an **identical global-batch frame multiset at every step** (reference rank-0 fanout vs RLinf
per-rank streaming, same dataset/seed/global-batch)?

## What is identical across the two loaders
From the full-30k value-independent `(episode_index, frame_index)` id-audits:
- **Same dataset**: both cover the exact same **429,928** distinct frames across the same 200
  episodes (0 frames unique to either side; cross-dump overlap = 429,928) and emit exactly **256
  distinct frames per step** at every step.
- **Same** seed (42), global batch (256), `gradient_accumulate` (1), `num_workers` knob (8),
  keyframe-chunk streaming (`chunk_size=250`).

So neither the data pool, the seed, the batch size, nor the per-step count differ — only the
per-step **grouping/order** of frames differs, and it does so for two separate reasons.

## Cause 1 — chunk-partition LANE COUNT (production divergence from step 0)
The two pipelines shard the chunk stream into a different number of **lanes** (a lane = one worker
cursor with its own `default_rng(seed+lane_id)` shuffle + random start; the DataLoader round-robins
batches across lanes; the dataset ignores `idx`, so `shuffle`/sampler are no-ops on content —
proven by `phase6_loader_audit_ids_equiv.json`):

| | how data is loaded | lanes |
|---|---|---|
| **Reference** | centralized rank-0 fanout (`train_pytorch_new.py:866-869,948-961`): ONLY rank 0 builds the loader; its `num_workers=8` workers are the only lanes; rank 0 scatters batches | **`num_workers` = 8** |
| **RLinf** | decentralized (`behavior_sft_dataset.py:740,1168-1187`): each rank builds its own loader and streams its own shard (rank folded into the stride) | **`world_size * num_workers` = 64** |

64 lanes each advance once per 8 steps while 8 lanes all advance every step, so at production
`num_workers=8` the per-step global multiset diverges from **step 0**
(`phase6_loader_audit_ids_prod_compare.json`: `DIVERGENT_MULTISET`, cause `lane_count`, first
mismatch step 0, RLinf `1660:1000…` vs reference `1080:3000…`). Neither loader is buggy — RLinf's
rank-folding is correct for its decentralized design; the reference never folds rank because only
rank 0 loads. It is a deliberate `num_workers` sharding-granularity difference.

## Matched lane count → IDENTICAL for the entire first epoch (incl. step 0)
With `world_size=8`, RLinf at **`num_workers=1`** has `world_size*num_workers = 8` lanes, and lane
`r = range(r, N, 8)` `default_rng(42+r)` = exactly the reference worker-`r` lane. The global
per-step multiset is then **IDENTICAL for the entire first epoch — all 1679 steps, including step 0
and step 1** (`phase6_loader_audit_ids_aligned_compare.json`: `IDENTICAL_FIRST_EPOCH`,
`identical_prefix_steps=1679`, cross-overlap 429,928). `epoch_len = (429928 // 32) // 8 = 1679`.

## Cause 2 — epoch-boundary handling (divergence only AFTER the first epoch)
At matched lane count the two streams are identical up to **exactly the epoch boundary** (step
1679), then diverge. The reason is structural, not a bug:
- The reference's `TorchDataLoader.__iter__` (`data_loader.py:396-411`) is an infinite generator
  that **re-creates its single inner DataLoader iterator on exhaustion**. That loader holds all 8
  lanes and yields `floor(429928/32) = 13435` batches per epoch; with `drop_last` and rank-0
  fanout pulling `world_size=8` batches/step, the epoch ends at step `13435/8 = 1679.4` — **not a
  step boundary** — so the boundary step is a mix of epoch-1-tail + epoch-2-head, and the 8
  worker-lanes receive uneven per-epoch batch counts (13435 mod 8 = 3) and drift.
- RLinf's per-rank loaders each stream a full lane uniformly and re-iterate only every `13435`
  steps; the cursor advances continuously across the wrap (`behavior_sft_dataset.py:1213-1222`).

There is **no `set_epoch` reshuffle** on either side (the reference has no `set_epoch` method, so the
`hasattr`-guarded call in `train_pytorch_new.py:949` is never taken). Both pipelines keep sampling
the SAME 429,928-frame pool at 256 distinct/step after the boundary — it is a benign reorder of the
same data, **not a data-availability difference**. Full-30k identity is therefore architecturally
impossible while each pipeline keeps its native (centralized fanout vs decentralized per-rank)
epoch handling.

## Conclusion (feeds AC-5)
1. **At matched lane granularity the two production loaders feed an IDENTICAL global-batch multiset
   for the entire first epoch — including step 0 and the early steps that the first-step
   loss/grad-norm comparison uses.** So the first-step loss gap is NOT explained by a step-0 data
   difference under matched lanes.
2. **At production settings they diverge from step 0**, entirely because RLinf runs `num_workers=8`
   (64 lanes) vs the reference's 8 lanes. To make production data-identical for step 0 (and the
   whole first epoch), run RLinf SFT with the matched lane count (`num_workers=1` under
   `world_size=8`).
3. **Beyond the first epoch** the streams reorder the same pool differently (centralized fanout's
   non-step-aligned epoch boundary vs decentralized per-rank cycling) — benign for training and
   irrelevant to the step-0 question.

AC-3's 2×2 cross-feed (identical materialized batch through both stacks) is the right next step to
isolate any residual compute difference; it does not depend on the loaders agreeing.

## Artifacts
- `phase6_loader_audit_ids_ref.json` — reference rank-0 fanout, nw=8 (8 lanes), 30k.
- `phase6_loader_audit_ids_rlinf_aligned.json` — RLinf nw=1 (8 lanes, reference-aligned), 30k.
- `phase6_loader_audit_ids_rlinf_prod.json` — RLinf nw=8 (64 lanes, production), 30k.
- `phase6_loader_audit_ids_aligned_compare.json` — `IDENTICAL_FIRST_EPOCH` (prefix 1679, step 0 incl.).
- `phase6_loader_audit_ids_prod_compare.json` — `DIVERGENT_MULTISET` (cause `lane_count`, step 0).
- `phase6_loader_audit_ids_equiv.json` — id_only shortcut == production wrapper (shuffle no-op).
- `phase6_loader_audit_ids_nonperturb.json` — byte-identical re-runs (both repos).
- Gate: `tests/unit_tests/test_openpi_pytorch_sft_loader_audit_ids.py` (11 tests).
