# Rank-independent rerun — auditable evidence (data-causality test)

This is the auditable record of the temporary rank-independent rerun used to test whether the
RLinf↔reference chunk-partition difference (RLinf 256 unique frames/step vs reference 32, see
`r18_*_stream_coverage.csv`) is the *cause* of the first-50-step loss divergence. The patch was
**reverted** (no committed `rlinf/` source change); this file records it for reproducibility.

## Exact source override (`rlinf/data/datasets/behavior/behavior_sft_dataset.py`,
`BehaviorSftDataset._select_streaming_chunk`)

Replace the rank-folding partition + seed with the reference's rank-independent form
(`range(worker_id, n, num_workers)`, rng seed `self.seed + worker_id`):

```python
# BEFORE (production; rank-folded -> 256 unique frames/step on 8 ranks):
rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
worker_info = get_worker_info()
worker_id = 0 if worker_info is None else worker_info.id
num_workers = 1 if worker_info is None else worker_info.num_workers
global_worker_id = rank * num_workers + worker_id
if not hasattr(self, "_active_chunks") or self._active_chunks is None:
    indices = partition_chunk_indices(len(self.chunks), rank=rank, world_size=world_size,
                                      worker_id=worker_id, num_workers=num_workers)
    worker_chunks = [self.chunks[i] for i in indices]
    rng = np.random.default_rng(self.seed + global_worker_id)
    rng.shuffle(worker_chunks)
    self._active_chunks = worker_chunks
rng = np.random.default_rng(self.seed + global_worker_id)
self.current_streaming_chunk_idx = rng.integers(0, len(self._active_chunks)).item()

# AFTER (rank-independent, matching the reference -> 32 unique frames/step on 8 ranks):
worker_info = get_worker_info()
worker_id = 0 if worker_info is None else worker_info.id
num_workers = 1 if worker_info is None else worker_info.num_workers
if not hasattr(self, "_active_chunks") or self._active_chunks is None:
    indices = list(range(worker_id, len(self.chunks), num_workers))
    worker_chunks = [self.chunks[i] for i in indices]
    rng = np.random.default_rng(self.seed + worker_id)
    rng.shuffle(worker_chunks)
    self._active_chunks = worker_chunks
rng = np.random.default_rng(self.seed + worker_id)
self.current_streaming_chunk_idx = rng.integers(0, len(self._active_chunks)).item()
```

## Command (8 GPUs)

```
REPO_PATH=/mnt/public/xzxuan/repos/RLinf_pi05 EMBODIED_PATH=$REPO_PATH/examples/sft \
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TMPDIR=/mnt/public/xzxuan/tmp \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONPATH=$REPO_PATH \
python examples/sft/train_vla_sft.py \
  --config-path $REPO_PATH/examples/sft/config --config-name behavior_pi05_vla \
  runner.max_steps=55 runner.save_interval=100000 \
  runner.logger.log_path=/mnt/public/xzxuan/tmp/r17_sft_rankindep
# -> EXIT=0; first-50 scalars in docs/evidence/r17_rank_independent_first50_losses.csv
```

- Output dir / tensorboard: `/mnt/public/xzxuan/tmp/r17_sft_rankindep/tensorboard/`.
- First-50 scalar table: `docs/evidence/r17_rank_independent_first50_losses.csv`.

## Result — the chunk-partition difference is NOT causal

The rank-independent rerun makes RLinf's per-step coverage 32 (identical to the reference, see
`r18_ref_stream_coverage.csv`). The first-50 loss stayed **flat** (mean 0.234, step0 0.226 →
step49 0.210, **12/50** within `|Δ| ≤ 0.03` — unchanged from the production R16 run). So the
RLinf↔reference data-stream difference does NOT explain the flat-vs-dropping loss; the cause is
downstream of the data (the FSDP optimizer/gradient/weight-update path — next round).
