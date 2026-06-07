# Copyright (c) 2025, RLinf contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Value-parity gate for the SFT->new-format converter (``sft_to_new_pytorch``).

The converter's only value transform is: strip wrapper/FSDP key prefixes and cast
float tensors to bf16. These tests prove it is value-lossless up to that
documented bf16 cast, independent of any training run:

* weight identity -- every converted tensor equals the matching consolidated
  full-weights tensor cast to bf16 (verified WITHOUT the converter's own helper),
  the key map is exactly "strip the wrapper prefix", and no wrapper prefix
  survives;
* structural identity -- the converted state dict's key-set / per-key shape /
  per-key dtype equal the reference new-format eval checkpoint (all bf16);
* forward identity -- on a fixed observation + injected noise/time, the model
  built from the consolidated checkpoint and the model built from the converted
  checkpoint produce the same flow-matching loss and the same sampled action
  chunk (normalized AND post-denormalization).

External assets live outside the repo, so each test skip-gates when its inputs
are absent while keeping the assertion the literal value-lossless requirement.
"""

from __future__ import annotations

import copy
import pathlib

import numpy as np
import pytest

_SFT_RUN = pathlib.Path(
    "/mnt/public/xzxuan/repos/RLinf_pi05/logs/20260605-12:39:44-behavior_pi05_vla"
)
_FULL_WEIGHTS = (
    _SFT_RUN
    / "sft_behavior_pi05_vla/checkpoints/global_step_30000"
    / "actor/model_state_dict/full_weights.pt"
)
_CONVERTED = _SFT_RUN / "pi05_sft_pytorch_new"
_CONVERTED_WEIGHTS = _CONVERTED / "model.safetensors"
# The reference new-format EVAL checkpoint (bf16) -- same role/format as the
# converted output, so the structural comparison is apples-to-apples.
_REF_EVAL_CKPT = pathlib.Path(
    "/mnt/public/xzxuan/models/ckpt/jax_task0000_sft_29999_ptnew/model.safetensors"
)
_NORM_STATS_DIR = _CONVERTED / "physical-intelligence/behavior"
_IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
_NUM_STEPS = 5
# Identical weights + identical inputs are deterministic on one device; the band
# only absorbs incidental CUDA reduction non-determinism. Expect ~0.
_FORWARD_TOL = 1e-2


def _have(*paths: pathlib.Path) -> bool:
    return all(p.exists() for p in paths)


def _strip_all_prefixes(key: str, prefixes: tuple[str, ...]) -> str:
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                changed = True
                break
    return key


@pytest.mark.skipif(
    not _have(_FULL_WEIGHTS, _CONVERTED_WEIGHTS),
    reason="SFT full_weights / converted checkpoint not present",
)
def test_converter_is_value_lossless_bf16_cast():
    """Every converted tensor equals the consolidated full-weights tensor cast to
    bf16; the key map is exactly the wrapper-prefix strip; no prefix survives.

    Verified independently of the converter's own ``_strip_wrapper_prefix`` so
    this is a real check of the transform, not a tautology."""
    import torch
    from safetensors import safe_open

    from rlinf.models.embodiment.openpi_pytorch.utils.export_sft_checkpoint import (
        _WRAPPER_PREFIXES,
        _as_state_dict,
    )

    raw = _as_state_dict(
        torch.load(
            str(_FULL_WEIGHTS), map_location="cpu", weights_only=False, mmap=True
        )
    )
    # Map each consolidated key to its bare form; the map must be a bijection.
    bare_to_raw = {_strip_all_prefixes(k, _WRAPPER_PREFIXES): k for k in raw}
    assert len(bare_to_raw) == len(raw), "wrapper-prefix strip is not a bijection"

    with safe_open(str(_CONVERTED_WEIGHTS), framework="pt", device="cpu") as conv:
        conv_keys = set(conv.keys())
        assert conv_keys == set(bare_to_raw), (
            "converted key-set != stripped consolidated key-set"
        )
        max_abs = 0.0
        for bare_key in conv_keys:
            assert not bare_key.startswith(_WRAPPER_PREFIXES), (
                f"wrapper prefix survived on {bare_key!r}"
            )
            got = conv.get_tensor(bare_key)
            want = raw[bare_to_raw[bare_key]].to(torch.bfloat16)
            assert got.dtype == torch.bfloat16, f"{bare_key} is {got.dtype}, not bf16"
            assert got.shape == want.shape
            # Exact, bitwise: the only transform is the fp32->bf16 cast.
            assert torch.equal(got, want), f"value mismatch at {bare_key}"
            max_abs = max(max_abs, float((got.float() - want.float()).abs().max()))
        assert max_abs == 0.0


@pytest.mark.skipif(
    not _have(_CONVERTED_WEIGHTS, _REF_EVAL_CKPT),
    reason="converted / reference eval checkpoint not present",
)
def test_converted_matches_reference_new_format_structure():
    """Converted key-set / per-key shape / per-key dtype equal the reference
    new-format eval checkpoint; all float tensors bf16; no wrapper prefixes."""
    from safetensors import safe_open

    from rlinf.models.embodiment.openpi_pytorch.utils.export_sft_checkpoint import (
        _WRAPPER_PREFIXES,
    )

    with (
        safe_open(str(_CONVERTED_WEIGHTS), framework="pt", device="cpu") as conv,
        safe_open(str(_REF_EVAL_CKPT), framework="pt", device="cpu") as ref,
    ):
        conv_keys, ref_keys = set(conv.keys()), set(ref.keys())
        assert conv_keys == ref_keys, (
            f"key-set differs: only-converted={sorted(conv_keys - ref_keys)[:5]}, "
            f"only-reference={sorted(ref_keys - conv_keys)[:5]}"
        )
        for key in conv_keys:
            c, r = conv.get_slice(key), ref.get_slice(key)
            assert list(c.get_shape()) == list(r.get_shape()), f"shape differs at {key}"
            assert c.get_dtype() == r.get_dtype(), f"dtype differs at {key}"
            assert "BF16" in c.get_dtype().upper(), f"{key} not bf16: {c.get_dtype()}"
            assert not key.startswith(_WRAPPER_PREFIXES), f"wrapper prefix on {key}"


def _fixed_observation(torch):
    np.random.seed(0)
    from rlinf.models.embodiment.openpi_pytorch.pi0_model.tokenizer import (
        PaligemmaTokenizer,
    )

    images = {
        k: np.random.randint(0, 256, (1, 224, 224, 3), dtype=np.uint8)
        for k in _IMAGE_KEYS
    }
    masks = {k: np.ones((1,), dtype=bool) for k in _IMAGE_KEYS}
    state = np.random.uniform(-1.0, 1.0, (1, 32)).astype(np.float32)
    tokens, tmask = PaligemmaTokenizer(max_len=200).tokenize(
        "turn on radio", state[0, :23]
    )
    return {
        "image": {k: torch.from_numpy(images[k]).to("cuda") for k in _IMAGE_KEYS},
        "image_mask": {k: torch.from_numpy(masks[k]).to("cuda") for k in _IMAGE_KEYS},
        "state": torch.from_numpy(state).to("cuda"),
        "tokenized_prompt": torch.from_numpy(tokens[None].astype(np.int64)).to("cuda"),
        "tokenized_prompt_mask": torch.from_numpy(tmask[None].astype(bool)).to("cuda"),
    }


def _build_pi0(state_dict, torch):
    from rlinf.models.embodiment.openpi_pytorch.pi0_model.pi0_config import Pi0Config

    model = Pi0Config(
        pi05=True, action_horizon=32, action_dim=32, dtype="bfloat16", pcd=False
    ).create()
    model.load_state_dict(state_dict, strict=True)
    return model.to("cuda").to(torch.bfloat16).eval()


def _make_processor():
    from rlinf.models.embodiment.openpi_pytorch.pi0_model.normalize import (
        load_norm_stats,
    )
    from rlinf.models.embodiment.openpi_pytorch.pi0_model.processing import (
        BehaviorEvalProcessor,
    )
    from rlinf.models.embodiment.openpi_pytorch.pi0_model.tokenizer import (
        PaligemmaTokenizer,
    )

    return BehaviorEvalProcessor(
        load_norm_stats(_NORM_STATS_DIR),
        PaligemmaTokenizer(max_len=200),
        action_chunk=32,
        action_env_dim=23,
        model_action_dim=32,
    )


def _wrap(pi0_model, processor):
    from rlinf.models.embodiment.openpi_pytorch.openpi_action_model import (
        OpenPiPytorchActionModel,
    )

    return OpenPiPytorchActionModel(
        pi0_model,
        processor,
        num_steps=_NUM_STEPS,
        action_chunk=32,
        action_env_dim=23,
    )


def _fixed_env_obs(torch):
    g = torch.Generator().manual_seed(0)
    return {
        "main_images": torch.randint(
            0, 256, (1, 720, 720, 3), dtype=torch.uint8, generator=g
        ),
        "wrist_images": torch.randint(
            0, 256, (1, 2, 480, 480, 3), dtype=torch.uint8, generator=g
        ),
        "states": torch.rand(1, 256, generator=g).double(),
        "task_descriptions": ["turn on radio"],
        "extra_view_images": None,
    }


@pytest.fixture(scope="module")
def converted_models():
    """Build the pre-conversion and converted Pi0 models + shared processor once.

    The pre-conversion model is the consolidated full-weights checkpoint reduced
    to a bare bf16 Pi0 (the SFT-trained model); the converted model is loaded from
    ``sft_to_new_pytorch.py``'s output."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if not _have(_FULL_WEIGHTS, _CONVERTED_WEIGHTS, _NORM_STATS_DIR):
        pytest.skip("SFT full_weights / converted checkpoint / norm-stats not present")
    import safetensors.torch

    from rlinf.models.embodiment.openpi_pytorch.utils.export_sft_checkpoint import (
        _as_state_dict,
        _strip_wrapper_prefix,
    )

    pre_state = _strip_wrapper_prefix(
        _as_state_dict(
            torch.load(
                str(_FULL_WEIGHTS), map_location="cpu", weights_only=False, mmap=True
            )
        )
    )
    post_state = safetensors.torch.load_file(str(_CONVERTED_WEIGHTS), device="cpu")
    return {
        "torch": torch,
        "pre": _build_pi0(pre_state, torch),
        "post": _build_pi0(post_state, torch),
        "processor": _make_processor(),
    }


def test_converter_bare_forward_parity(converted_models):
    """Bare Pi0 from the consolidated checkpoint vs from the converted checkpoint:
    same flow-matching loss + sampled action chunk (normalized AND denormalized)
    on a fixed input + injected noise/time."""
    torch = converted_models["torch"]
    pre_model, post_model = converted_models["pre"], converted_models["post"]
    processor = converted_models["processor"]

    from rlinf.models.embodiment.openpi_pytorch.pi0_model import model as vmodel

    raw = _fixed_observation(torch)
    actions = torch.randn(1, 32, 32, generator=torch.Generator().manual_seed(7)).to(
        "cuda"
    )
    loss_noise = torch.randn(1, 32, 32, generator=torch.Generator().manual_seed(99)).to(
        "cuda"
    )
    loss_time = torch.tensor([0.3]).to("cuda")
    sample_noise = torch.randn(1, 32, 32, generator=torch.Generator().manual_seed(123))

    def _loss(m):
        return m.compute_loss(
            vmodel.Observation.from_dict(copy.deepcopy(raw)),
            actions.clone(),
            train=False,
            noise=loss_noise.clone(),
            time=loss_time.clone(),
        ).float()

    def _act(m):
        return m.sample_actions(
            vmodel.Observation.from_dict(copy.deepcopy(raw)),
            num_steps=_NUM_STEPS,
            noise=sample_noise.clone().to("cuda"),
        ).float()

    with torch.no_grad():
        loss_diff = float((_loss(pre_model) - _loss(post_model)).abs().max())
        act_pre, act_post = _act(pre_model), _act(post_model)
    assert loss_diff <= _FORWARD_TOL, f"loss parity {loss_diff} > {_FORWARD_TOL}"
    assert float((act_pre - act_post).abs().max()) <= _FORWARD_TOL
    denorm_diff = float(
        (
            processor.postprocess_actions(act_pre)
            - processor.postprocess_actions(act_post)
        )
        .abs()
        .max()
    )
    assert denorm_diff <= _FORWARD_TOL, f"denormalized action parity {denorm_diff}"


def test_converter_wrapper_boundary_parity(converted_models):
    """Exercise the ``OpenPiPytorchActionModel`` wrapper boundary AC-2 names: the
    pre-conversion and converted wrappers give the same SFT loss (with injected
    noise/time) and the same eval actions via ``predict_action_batch`` with the
    injected-noise hook (both the returned denormalized actions and the
    ``forward_inputs.model_action`` normalized chunk)."""
    torch = converted_models["torch"]
    processor = converted_models["processor"]
    pre = _wrap(converted_models["pre"], processor)
    post = _wrap(converted_models["post"], processor)

    from rlinf.models.embodiment.openpi_pytorch.pi0_model import model as vmodel

    raw = _fixed_observation(torch)
    actions = torch.randn(1, 32, 32, generator=torch.Generator().manual_seed(7)).to(
        "cuda"
    )
    loss_noise = torch.randn(1, 32, 32, generator=torch.Generator().manual_seed(99)).to(
        "cuda"
    )
    loss_time = torch.tensor([0.3]).to("cuda")

    def _sft_loss(wrapper):
        return float(
            wrapper.compute_loss(
                {
                    "observation": vmodel.Observation.from_dict(copy.deepcopy(raw)),
                    "actions": actions.clone(),
                    "noise": loss_noise.clone(),
                    "time": loss_time.clone(),
                }
            )
        )

    sft_diff = abs(_sft_loss(pre) - _sft_loss(post))
    assert sft_diff <= _FORWARD_TOL, f"wrapper SFT loss parity {sft_diff}"

    # Eval action parity through predict_action_batch + the injected-noise hook.
    env_obs = _fixed_env_obs(torch)
    eval_noise = torch.randn(
        1, 32, 32, generator=torch.Generator().manual_seed(123)
    ).to("cuda")
    pre_actions, pre_res = pre.predict_action_batch(
        env_obs, mode="eval", noise=eval_noise.clone()
    )
    post_actions, post_res = post.predict_action_batch(
        env_obs, mode="eval", noise=eval_noise.clone()
    )
    env_diff = float((pre_actions - post_actions).abs().max())
    model_action_diff = float(
        (
            pre_res["forward_inputs"]["model_action"].float()
            - post_res["forward_inputs"]["model_action"].float()
        )
        .abs()
        .max()
    )
    assert env_diff <= _FORWARD_TOL, f"wrapper denormalized action parity {env_diff}"
    assert model_action_diff <= _FORWARD_TOL, (
        f"wrapper normalized model_action parity {model_action_diff}"
    )
