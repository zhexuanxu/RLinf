# BEHAVIOR SFT Fixed-Batch Forward-Loss Parity — Evidence (task14, AC-11 / DEC-1 (a))

GPU run-evidence for the `use_skill:false` fixed-batch forward-loss parity gate. The
harness is `tools/sft_loss_parity_probe.py`. **Verdict: the RLinf production forward loss
does NOT match the reference within DEC-1 — a robust ~16–25% divergence — and the gate is
NOT met. This is a real divergence localized below; it blocks task15 and is the next
round's mainline investigation (no silent pass, per the round-14 contract).**

## Run

```
TMPDIR=/mnt/public/xzxuan/tmp CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl \
    PYTHONPATH=. python tools/sft_loss_parity_probe.py
```

- Environment: 1× NVIDIA A800-80GB; `/mnt/public/xzxuan/.venv_pi` (torch, Python 3.10).
- Model: `pi05_base_pytorch_new` (3.35B params, fp32 strict load → bf16 compute), the
  reference `pytorch_weight_path`. `model.safetensors` sha256[:16] (first 64MB) =
  `650d624bc119a28f`.
- Norm stats: canonical task-0000 `norm_stats.json` sha256[:16] = `d66ed16830a98f90`.
- Data: `/mnt/public/xzxuan/data/2025-challenge-demos`, `turning_on_radio`, seed 42,
  micro-batch 32, 8 batches = 256 samples (= the reference per-step global batch).
- Loss: `Pi0.compute_loss` flow-matching MSE, with explicit fixed noise (`randn`) + time
  (`Beta(1.5,1.0)*0.999+0.001`), reproducible across runs.

## Measured

| Metric | Value | Within DEC-1 `[0.2338, 0.2584]`? |
|--------|-------|----------------------------------|
| Reference step-0 loss | 0.24609375 | — (target) |
| Reference first-10-step mean (2560 samples, from log) | 0.243848 | — |
| RLinf deterministic single-batch draw (seed 42, 32 samples) | 0.263672 | no |
| RLinf marginal E[loss] (8×32, train=True, 8 draws) | **0.285065** (std 0.0038) | **no** (Δ=0.039) |
| RLinf marginal E[loss] (8×32, train=False, no-aug) | 0.307465 | no |
| RLinf worker `sft_forward` (internal sampling) | 0.303711 | no |

The reference loss column then decays over training (first-20 mean 0.228, first-50 0.167,
first-100 0.111), confirming step-0/first-10 (~0.244) is the base-weight comparison point.

## Investigation — what is ruled out

The flow-matching loss math is **identical** between RLinf and the reference
(`openpi-comet-pytorch-mixed`, a PyTorch-new trainer the RLinf model was ported from), so
the divergence is NOT in the objective:

- **`compute_loss`**: byte-identical (`models_pytorch_new/pi0.py:277-342` vs
  `pi0_model/pi0.py:274-339`): `mean(square(v_t - u_t), dim=-1)`.
- **Noise/time distribution**: identical (`randn`; `Beta(1.5,1.0)*0.999+0.001`); the
  reference run uses `USE_CONSISTENT=0`, so it samples internally exactly like RLinf.
- **Loss reduction**: identical (`losses.mean()` per micro-batch, AVG-reduced across ranks;
  `train_pytorch_new.py:520-548`).
- **Weights**: the same `pi05_base_pytorch_new/model.safetensors`.
- **Augmentation**: ruled out — `train=False` (no crop/rotate/jitter) gives an *even higher*
  loss (0.307), so augmentation is not the cause; the reference applies the same train-time
  augmentation anyway (`models_pytorch_new/model.py:161-230`).
- **Per-sample transform**: ruled out — `tests/unit_tests/test_openpi_pytorch_sft_ref_parity.py`
  passes (3/3): the tokenizer is byte-exact and the normalized state/actions/images of the
  SAME raw frame match the real reference loader within tolerance.

## Investigation — remaining suspect

With the objective, weights, and per-sample transform all matching, the divergence must be
in **which frames / action windows the production loader streams** (the marginal loss over
RLinf's streamed frames is ~0.29–0.31 vs the reference's ~0.244). The per-sample parity test
verifies one matched raw frame, but does NOT verify that the streaming iteration yields the
same *sequence/selection* of frames as the reference. The RLinf loader streams contiguous
**keyframe chunks** (`BehaviorSftDataset._get_keyframe_chunk_indices`,
`chunk_streaming_using_keyframe=True`) with `delta_timestamps={"action": [t/30 for t in
range(32)]}`. Candidate divergences to check next round:

1. Keyframe-chunk selection / stride vs the reference's frame sampling — does RLinf include
   frames (e.g. episode-boundary or transition frames) the reference excludes, raising the
   marginal loss?
2. The action-horizon window construction at episode boundaries (padding/clamping of the
   32-step action target).
3. The state / discrete-state conditioning or image resize path under streaming (vs the
   single-frame parity path).

## Next round (blocking investigation, before task15)

Localize the input divergence by comparing the RLinf streamed-frame distribution to the
reference loader's streamed-frame distribution (extend the `sft_ref_parity` reference dump to
a frame *sequence*, not a single frame), find the systematic input difference, fix it, and
re-run this probe until the marginal E[loss] is within DEC-1 of 0.24609375. Only then launch
task15 (first-50-step run on 8 GPUs).
