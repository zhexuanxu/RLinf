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

"""Generate the SFT->new-format converter value-parity evidence.

Records, for the committed converted checkpoint: the weight-identity facts (every
converted tensor equals the consolidated full-weights tensor cast to bf16), the
structural match against the reference new-format eval checkpoint, and -- when a
GPU is available -- the fixed-input forward parity (loss + normalized +
denormalized action chunk) between the model built from the consolidated
checkpoint and the model built from the converted checkpoint.

The forward helpers are reused from the test module so the evidence and the gate
share one implementation. Run from the repo root::

    python tests/unit_tests/_converter_parity_dump.py \
        --out docs/evidence/phase4_converter_parity.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib

import torch
from safetensors import safe_open

from rlinf.models.embodiment.openpi_pytorch.utils.export_sft_checkpoint import (
    _WRAPPER_PREFIXES,
    _as_state_dict,
)

_T = importlib.util.spec_from_file_location(
    "_converter_parity_test",
    pathlib.Path(__file__).parent / "test_openpi_pytorch_converter_parity.py",
)
t = importlib.util.module_from_spec(_T)
_T.loader.exec_module(t)


def weight_identity() -> dict:
    raw = _as_state_dict(
        torch.load(
            str(t._FULL_WEIGHTS), map_location="cpu", weights_only=False, mmap=True
        )
    )
    bare_to_raw = {t._strip_all_prefixes(k, _WRAPPER_PREFIXES): k for k in raw}
    with safe_open(str(t._CONVERTED_WEIGHTS), framework="pt", device="cpu") as conv:
        conv_keys = set(conv.keys())
        max_abs = 0.0
        all_bf16 = True
        for bare_key in conv_keys:
            got = conv.get_tensor(bare_key)
            want = raw[bare_to_raw[bare_key]].to(torch.bfloat16)
            all_bf16 = all_bf16 and got.dtype == torch.bfloat16
            max_abs = max(max_abs, float((got.float() - want.float()).abs().max()))
    return {
        "n_tensors": len(conv_keys),
        "key_map_is_prefix_strip_bijection": len(bare_to_raw) == len(raw)
        and conv_keys == set(bare_to_raw),
        "no_wrapper_prefix_survives": not any(
            k.startswith(_WRAPPER_PREFIXES) for k in conv_keys
        ),
        "all_bf16": all_bf16,
        "max_abs_diff_vs_bf16_cast": max_abs,
        "bitwise_lossless": max_abs == 0.0,
    }


def structural_match() -> dict:
    with (
        safe_open(str(t._CONVERTED_WEIGHTS), framework="pt", device="cpu") as conv,
        safe_open(str(t._REF_EVAL_CKPT), framework="pt", device="cpu") as ref,
    ):
        conv_keys, ref_keys = set(conv.keys()), set(ref.keys())
        shape_ok = all(
            list(conv.get_slice(k).get_shape()) == list(ref.get_slice(k).get_shape())
            for k in conv_keys & ref_keys
        )
        dtype_ok = all(
            conv.get_slice(k).get_dtype() == ref.get_slice(k).get_dtype()
            for k in conv_keys & ref_keys
        )
        return {
            "reference_checkpoint": str(t._REF_EVAL_CKPT),
            "key_set_equal": conv_keys == ref_keys,
            "n_keys": len(conv_keys),
            "shapes_equal": shape_ok,
            "dtypes_equal": dtype_ok,
        }


def forward_parity() -> dict:
    if not torch.cuda.is_available():
        return {"skipped": "CUDA not available"}
    import copy

    import safetensors.torch

    from rlinf.models.embodiment.openpi_pytorch.pi0_model import model as vmodel
    from rlinf.models.embodiment.openpi_pytorch.utils.export_sft_checkpoint import (
        _strip_wrapper_prefix,
    )

    pre = t._build_pi0(
        _strip_wrapper_prefix(
            _as_state_dict(
                torch.load(
                    str(t._FULL_WEIGHTS),
                    map_location="cpu",
                    weights_only=False,
                    mmap=True,
                )
            )
        ),
        torch,
    )
    post = t._build_pi0(
        safetensors.torch.load_file(str(t._CONVERTED_WEIGHTS), device="cpu"), torch
    )
    raw = t._fixed_observation(torch)
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
            num_steps=t._NUM_STEPS,
            noise=sample_noise.clone().to("cuda"),
        ).float()

    with torch.no_grad():
        loss_diff = float((_loss(pre) - _loss(post)).abs().max())
        a_pre, a_post = _act(pre), _act(post)
    proc = t._make_processor()
    result = {
        "num_steps": t._NUM_STEPS,
        "tolerance": t._FORWARD_TOL,
        "loss_max_abs_diff": loss_diff,
        "normalized_action_max_abs_diff": float((a_pre - a_post).abs().max()),
        "denormalized_action_max_abs_diff": float(
            (proc.postprocess_actions(a_pre) - proc.postprocess_actions(a_post))
            .abs()
            .max()
        ),
    }

    # Wrapper-boundary parity (OpenPiPytorchActionModel + predict_action_batch).
    pre_w, post_w = t._wrap(pre, proc), t._wrap(post, proc)

    def _sft(wrapper):
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

    env_obs = t._fixed_env_obs(torch)
    eval_noise = torch.randn(
        1, 32, 32, generator=torch.Generator().manual_seed(123)
    ).to("cuda")
    pa, pr = pre_w.predict_action_batch(env_obs, mode="eval", noise=eval_noise.clone())
    qa, qr = post_w.predict_action_batch(env_obs, mode="eval", noise=eval_noise.clone())
    result["wrapper"] = {
        "sft_loss_max_abs_diff": abs(_sft(pre_w) - _sft(post_w)),
        "eval_denormalized_action_max_abs_diff": float((pa - qa).abs().max()),
        "eval_normalized_model_action_max_abs_diff": float(
            (
                pr["forward_inputs"]["model_action"].float()
                - qr["forward_inputs"]["model_action"].float()
            )
            .abs()
            .max()
        ),
    }
    return result


def build_evidence() -> dict:
    return {
        "description": (
            "Value-parity of sft_to_new_pytorch.py: the converter's only value "
            "transform is strip-wrapper-prefix + bf16 cast. Weight identity proves "
            "value-losslessness up to that cast; the structural and forward checks "
            "confirm the converted checkpoint loads and behaves identically."
        ),
        "consolidated_checkpoint": str(t._FULL_WEIGHTS),
        "converted_checkpoint": str(t._CONVERTED_WEIGHTS),
        "weight_identity": weight_identity(),
        "structural_match": structural_match(),
        "forward_parity": forward_parity(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="docs/evidence/phase4_converter_parity.json")
    args = parser.parse_args()
    evidence = build_evidence()
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out}")
    print("weight bitwise_lossless =", evidence["weight_identity"]["bitwise_lossless"])
    print("forward_parity =", evidence["forward_parity"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
