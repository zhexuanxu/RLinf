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

"""Convert a NEW-format PyTorch OpenPI 0.5 checkpoint to the OLD layout.

The inverse of :mod:`old_to_new`: ``new_to_old_state_dict`` maps the bare
``Pi0`` keys back to the ``paligemma_with_expert.*`` layout (QKV split, MLP
unstack+transpose, pos-embedding squeeze). ``convert_trained_ckpt`` additionally
validates the converted keys/shapes against a reference old-format model and
fixes the (untrained) narrow action-expert ``lm_head``.

Usage (four-parameter interface; the two norm-stats paths are copied across):

    python -m rlinf.models.embodiment.openpi_pytorch.utils.new_to_old \\
        --input-model       /path/to/new_format_model.safetensors \\
        --input-norm-stats  /path/to/norm_stats.json \\
        --output-model      /path/to/old_format_out \\
        --output-norm-stats /path/to/old_format_out/physical-intelligence/behavior/norm_stats.json
"""

from __future__ import annotations

import argparse
import pathlib

import torch

from rlinf.models.embodiment.openpi_pytorch.utils.old_to_new import (
    _resolve_model_safetensors,
    copy_norm_stats,
)


def new_to_old_state_dict(new_sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert a new-format flat state dict back to the old ``paligemma_with_expert`` layout."""
    old_sd: dict[str, torch.Tensor] = {}

    _SIGLIP_OLD = "paligemma_with_expert.paligemma.model.vision_tower.vision_model."

    for suf in (".weight", ".bias"):
        nk = "img.stem" + suf
        if nk in new_sd:
            old_sd[_SIGLIP_OLD + "embeddings.patch_embedding" + suf] = new_sd[nk]

    if "img.pos_embedding" in new_sd:
        t = new_sd["img.pos_embedding"]
        if t.dim() == 3 and t.shape[0] == 1:
            t = t.squeeze(0)
        old_sd[_SIGLIP_OLD + "embeddings.position_embedding.weight"] = t

    for i in range(27):
        op = f"{_SIGLIP_OLD}encoder.layers.{i}."
        np = f"img.encoder.layers.{i}."

        for old_n, new_n in [("layer_norm1", "norm1"), ("layer_norm2", "norm2")]:
            for suf in (".weight", ".bias"):
                nk = f"{np}{new_n}{suf}"
                if nk in new_sd:
                    old_sd[f"{op}{old_n}{suf}"] = new_sd[nk]

        wk = f"{np}attn.in_proj_weight"
        if wk in new_sd:
            q_w, k_w, v_w = torch.chunk(new_sd[wk], 3, dim=0)
            old_sd[f"{op}self_attn.q_proj.weight"] = q_w.contiguous()
            old_sd[f"{op}self_attn.k_proj.weight"] = k_w.contiguous()
            old_sd[f"{op}self_attn.v_proj.weight"] = v_w.contiguous()
        bk = f"{np}attn.in_proj_bias"
        if bk in new_sd:
            q_b, k_b, v_b = torch.chunk(new_sd[bk], 3, dim=0)
            old_sd[f"{op}self_attn.q_proj.bias"] = q_b.contiguous()
            old_sd[f"{op}self_attn.k_proj.bias"] = k_b.contiguous()
            old_sd[f"{op}self_attn.v_proj.bias"] = v_b.contiguous()

        for suf in (".weight", ".bias"):
            nk = f"{np}attn.out_proj{suf}"
            if nk in new_sd:
                old_sd[f"{op}self_attn.out_proj{suf}"] = new_sd[nk]

        for name in ("fc1", "fc2"):
            for suf in (".weight", ".bias"):
                nk = f"{np}mlp.{name}{suf}"
                if nk in new_sd:
                    old_sd[f"{op}mlp.{name}{suf}"] = new_sd[nk]

    for suf in (".weight", ".bias"):
        nk = "img.encoder.norm" + suf
        if nk in new_sd:
            old_sd[_SIGLIP_OLD + "post_layernorm" + suf] = new_sd[nk]

    for suf in (".weight", ".bias"):
        nk = "img.head" + suf
        if nk in new_sd:
            old_sd[
                "paligemma_with_expert.paligemma.model.multi_modal_projector.linear"
                + suf
            ] = new_sd[nk]

    # --- PaliGemma LLM ---
    _PALI_LLM = "paligemma_with_expert.paligemma.model.language_model."
    for i in range(18):
        op = f"{_PALI_LLM}layers.{i}."
        np = f"llm.layers.{i}."

        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            nk = f"{np}attn.{proj}.0.weight"
            if nk in new_sd:
                old_sd[f"{op}self_attn.{proj}.weight"] = new_sd[nk]

        gk = f"{np}mlps.0.w_gating"
        if gk in new_sd:
            w = new_sd[gk]  # (2, features, hidden_dim)
            old_sd[f"{op}mlp.gate_proj.weight"] = w[0].T.contiguous()
            old_sd[f"{op}mlp.up_proj.weight"] = w[1].T.contiguous()

        nk = f"{np}mlps.0.w_linear"
        if nk in new_sd:
            old_sd[f"{op}mlp.down_proj.weight"] = new_sd[nk].T.contiguous()

        for old_n, new_n in [
            ("input_layernorm", "pre_attention_norms"),
            ("post_attention_layernorm", "pre_ffw_norms"),
        ]:
            nk = f"{np}{new_n}.0.scale"
            if nk in new_sd:
                old_sd[f"{op}{old_n}.weight"] = new_sd[nk]

    nk = "llm.final_norms.0.scale"
    if nk in new_sd:
        old_sd[_PALI_LLM + "norm.weight"] = new_sd[nk]

    # --- Gemma Expert ---
    _GEMMA_EXP = "paligemma_with_expert.gemma_expert.model."
    for i in range(18):
        op = f"{_GEMMA_EXP}layers.{i}."
        np = f"llm.layers.{i}."

        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            nk = f"{np}attn.{proj}.1.weight"
            if nk in new_sd:
                old_sd[f"{op}self_attn.{proj}.weight"] = new_sd[nk]

        gk = f"{np}mlps.1.w_gating"
        if gk in new_sd:
            w = new_sd[gk]  # (2, features, hidden_dim)
            old_sd[f"{op}mlp.gate_proj.weight"] = w[0].T.contiguous()
            old_sd[f"{op}mlp.up_proj.weight"] = w[1].T.contiguous()

        nk = f"{np}mlps.1.w_linear"
        if nk in new_sd:
            old_sd[f"{op}mlp.down_proj.weight"] = new_sd[nk].T.contiguous()

        for old_n, new_n in [
            ("input_layernorm", "pre_attention_norms"),
            ("post_attention_layernorm", "pre_ffw_norms"),
        ]:
            for suf in (".weight", ".bias"):
                nk = f"{np}{new_n}.1.ada_modulation{suf}"
                if nk in new_sd:
                    old_sd[f"{op}{old_n}.dense{suf}"] = new_sd[nk]

    for suf in (".weight", ".bias"):
        nk = "llm.final_norms.1.ada_modulation" + suf
        if nk in new_sd:
            old_sd[_GEMMA_EXP + "norm.dense" + suf] = new_sd[nk]

    # --- embedder -> PaliGemma lm_head ---
    # The new format carries a SINGLE shared embedder (PaliGemma's, width 2048),
    # tied to ``paligemma.lm_head``. The old format ALSO has a separate 1024-wide
    # action-expert head (``gemma_expert.lm_head``) that ``old_to_new`` drops and
    # the new format does NOT carry, so it cannot be reconstructed here — emitting
    # the 2048-wide embedder for it would be a malformed (wrong-shape) tensor. The
    # reference-backed ``convert_trained_ckpt`` sources the correct 1024-wide head
    # from a reference model; the four-parameter ``convert_new_to_old`` legitimately
    # omits this one old-only tensor.
    if "llm.embedder.embedding.weight" in new_sd:
        old_sd["paligemma_with_expert.paligemma.lm_head.weight"] = new_sd[
            "llm.embedder.embedding.weight"
        ]

    # --- Action head (pass through) ---
    for k in new_sd:
        if k.startswith(
            (
                "action_in_proj",
                "action_out_proj",
                "time_mlp_",
                "state_proj",
                "action_time_mlp_",
                "pointnet.",
            )
        ):
            old_sd[k] = new_sd[k]

    return old_sd


def convert_trained_ckpt(
    input_ckpt: str,
    output_dir: str,
    reference_model: str,
    norm_stats: str | None = None,
) -> None:
    """Convert a new-format trained checkpoint to old format aligned with a reference model.

    Args:
        input_ckpt: Path to model.pt (new-format, possibly with _orig_mod. prefix).
        output_dir: Output directory for the converted checkpoint.
        reference_model: Path to the reference model directory (old format) containing
            model.safetensors and config.json.
        norm_stats: Optional path to norm_stats.json to copy into the output.
    """
    import os
    import shutil

    import safetensors.torch

    sd = torch.load(input_ckpt, map_location="cpu", weights_only=True)

    # Strip _orig_mod. prefix (added by torch.compile)
    stripped = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    del sd

    old_sd = new_to_old_state_dict(stripped)
    del stripped

    ref_safetensors = os.path.join(reference_model, "model.safetensors")
    ref_sd = safetensors.torch.load_file(ref_safetensors)

    # Source gemma_expert.lm_head.weight — the new model only has a single 2048-dim
    # embedding (PaliGemma's), but the old format needs a separate 1024-dim one for
    # the action expert. ``new_to_old_state_dict`` does NOT emit it (it cannot be
    # reconstructed from the new format), so copy the correct 1024-wide weight from
    # the reference when it is absent or shape-mismatched. It is not trained.
    expert_lm_head_key = "paligemma_with_expert.gemma_expert.lm_head.weight"
    if expert_lm_head_key in ref_sd and (
        expert_lm_head_key not in old_sd
        or old_sd[expert_lm_head_key].shape != ref_sd[expert_lm_head_key].shape
    ):
        old_sd[expert_lm_head_key] = ref_sd[expert_lm_head_key].clone()

    for k in old_sd:
        if old_sd[k].dtype != torch.bfloat16:
            old_sd[k] = old_sd[k].to(torch.bfloat16)

    # Validate against reference (keys + shapes must match exactly).
    ref_keys = set(ref_sd.keys())
    conv_keys = set(old_sd.keys())
    missing = ref_keys - conv_keys
    extra = conv_keys - ref_keys
    shape_mismatches = [
        (k, tuple(ref_sd[k].shape), tuple(old_sd[k].shape))
        for k in sorted(ref_keys & conv_keys)
        if old_sd[k].shape != ref_sd[k].shape
    ]
    if missing or extra or shape_mismatches:
        raise RuntimeError(
            "Validation failed — keys/shapes do not match reference model: "
            f"missing={sorted(missing)} extra={sorted(extra)} "
            f"shape_mismatches={shape_mismatches}"
        )
    del ref_sd

    os.makedirs(output_dir, exist_ok=True)
    safetensors.torch.save_file(old_sd, os.path.join(output_dir, "model.safetensors"))
    del old_sd

    ref_config = os.path.join(reference_model, "config.json")
    if os.path.exists(ref_config):
        shutil.copy2(ref_config, os.path.join(output_dir, "config.json"))

    if norm_stats and os.path.exists(norm_stats):
        norm_dst_dir = os.path.join(output_dir, "physical-intelligence", "behavior")
        os.makedirs(norm_dst_dir, exist_ok=True)
        shutil.copy2(norm_stats, os.path.join(norm_dst_dir, "norm_stats.json"))


def convert_new_to_old(
    input_model: str | pathlib.Path,
    input_norm_stats: str | pathlib.Path,
    output_model: str | pathlib.Path,
    output_norm_stats: str | pathlib.Path,
) -> pathlib.Path:
    """Convert a new-format checkpoint to the old layout (four-parameter interface).

    Loads the new ``model.safetensors`` from ``input_model`` (a directory or a
    file), converts it via :func:`new_to_old_state_dict`, writes
    ``output_model/model.safetensors``, and copies ``input_norm_stats`` verbatim
    to ``output_norm_stats``.
    """
    import safetensors.torch

    input_model = pathlib.Path(input_model)
    output_model = pathlib.Path(output_model)
    new_path = _resolve_model_safetensors(input_model)
    if not new_path.exists():
        raise FileNotFoundError(f"new checkpoint not found: {new_path}")

    new_sd = safetensors.torch.load_file(str(new_path), device="cpu")
    old_sd = new_to_old_state_dict(new_sd)

    output_model.mkdir(parents=True, exist_ok=True)
    safetensors.torch.save_file(old_sd, str(output_model / "model.safetensors"))

    copy_norm_stats(input_norm_stats, output_norm_stats)
    return output_model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-model", required=True, help="new checkpoint dir or model.safetensors"
    )
    parser.add_argument(
        "--input-norm-stats", required=True, help="norm_stats.json to copy across"
    )
    parser.add_argument(
        "--output-model", required=True, help="output (old-format) checkpoint dir"
    )
    parser.add_argument(
        "--output-norm-stats", required=True, help="destination norm_stats.json path"
    )
    args = parser.parse_args()
    convert_new_to_old(
        args.input_model,
        args.input_norm_stats,
        args.output_model,
        args.output_norm_stats,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
