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
`hasattr`-guarded call in `train_pytorch_new.py:949` is never taken). Both decentralized streams keep
sampling the SAME 429,928-frame pool at 256 distinct/step after the boundary — a benign reorder of
the same data, **not a data-availability difference**. So full-30k identity is impossible *for the
decentralized `per_rank_stream` mode* while it keeps its native per-rank epoch handling.

## The `reference_fanout` mode achieves strict full-30k identity
The `data.loader_mode: reference_fanout` option replicates the reference's centralized pipeline:
ONE loader with a worker-only chunk partition (`dist_world_size=1` → `range(worker_id, N, num_workers)`
seed `seed+worker_id` = the reference's 8 lanes), the same infinite `__iter__` that re-creates its
inner iterator on exhaustion (`BehaviorSftDataLoader.__iter__`), pulled `world_size` micro-batches per
step on rank 0 and scattered to the other ranks (rank 0 pulls for ranks 1..world_size-1 first, then
itself last, exactly like `train_pytorch_new.py:953-961`). Because this matches the reference's lane
count AND its epoch-boundary handling, the global per-step multiset is **byte-identical to the
reference for ALL 30000 steps** (`fanout_compare`: `IDENTICAL_MULTISET`, `identical_full`,
`identical_prefix_steps=30000`, `global_rolling_hash` equals the reference's exactly, per-rank
`identical-order`), including across the epoch boundary; a re-run reproduces it byte-for-byte. It is
opt-in (default `per_rank_stream`) because the single loader feeds all ranks (a throughput cost).

## Conclusion
1. The decentralized default diverges at production (lane count, from step 0) and, at a matched lane
   count, is identical only for the first epoch (then a benign same-pool reorder).
2. The opt-in `reference_fanout` mode makes the data side **strictly identical to the reference for
   every one of the 30000 steps** — so once the data is aligned, the reference's training inputs are
   reproduced exactly. This is the configuration to use when verifying that RLinf reproduces the
   reference's behavior on identical data.
3. A 2×2 cross-feed (identical materialized batch through both model stacks) remains the way to
   isolate any residual COMPUTE difference; it does not depend on the loaders agreeing.

## Artifacts
- `phase6_loader_audit_ids_ref.json` — reference rank-0 fanout, nw=8 (8 lanes), 30k.
- `phase6_loader_audit_ids_rlinf_aligned.json` — RLinf nw=1 (8 lanes, reference-aligned), 30k.
- `phase6_loader_audit_ids_rlinf_prod.json` — RLinf nw=8 (64 lanes, production), 30k.
- `phase6_loader_audit_ids_aligned_compare.json` — `IDENTICAL_FIRST_EPOCH` (prefix 1679, step 0 incl.).
- `phase6_loader_audit_ids_prod_compare.json` — `DIVERGENT_MULTISET` (cause `lane_count`, step 0).
- `phase6_loader_audit_ids_equiv.json` — id_only shortcut == production wrapper (shuffle no-op).
- `phase6_loader_audit_ids_nonperturb.json` — byte-identical re-runs (both repos).
- Gate: `tests/unit_tests/test_openpi_pytorch_sft_loader_audit_ids.py` (11 tests).
