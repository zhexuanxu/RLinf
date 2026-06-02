"""Checkpoint format conversion — module-level patching.

Each module class (RMSNorm, Attention, FeedForward, etc.) owns its
old-format conversion logic.  This module provides:

- ``set_old_ckpt_format(enable)`` — global toggle
- ``_is_old_format(state_dict)`` — detect old-format keys
- ``pi0_old_state_dict(model)`` — build old-format state dict from a Pi0
- ``pi0_load_old_state_dict(model, old_sd)`` — load old-format params into Pi0

Pi0.<load_state_dict/state_dict> overrides call these helpers so that
``safetensors.torch.{load_model,save_model}`` work transparently.
export PYTHONPATH=.../src                                                                                                          
  python -m openpi.models_pytorch_new.checkpoint_format \                                                                            
      --input_ckpt .../model.pt \                                                                                                    
      --output_dir .../output \                             
      --reference_model /mnt/public/xzxuan/models/pi05_base_pytorch \                                                                
      --norm_stats .../norm_stats.json 

"""

from __future__ import annotations

import torch


def old_to_new_state_dict(old_sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert old-format flat state dict to new-format.

    Handles key renaming and weight transformations:
      - SigLIP Q/K/V concat → in_proj_weight/bias
      - LLM MLP gate/up transpose+stack → w_gating (2, features, hidden_dim)
      - LLM MLP down transpose → w_linear
    """
    new_sd: dict[str, torch.Tensor] = {}

    # --- Helper: build a SigLIP encoder prefix pair ---
    _SIGLIP_OLD = "paligemma_with_expert.paligemma.model.vision_tower.vision_model."

    # Stem
    for suf in (".weight", ".bias"):
        ok = _SIGLIP_OLD + "embeddings.patch_embedding" + suf
        if ok in old_sd:
            new_sd["img.stem" + suf] = old_sd[ok]

    # Pos embedding
    ok = _SIGLIP_OLD + "embeddings.position_embedding.weight"
    if ok in old_sd:
        new_sd["img.pos_embedding"] = old_sd[ok]

    # Encoder layers (0..26)
    for i in range(27):
        op = f"{_SIGLIP_OLD}encoder.layers.{i}."
        np = f"img.encoder.layers.{i}."

        # Norms
        for old_n, new_n in [("layer_norm1", "norm1"), ("layer_norm2", "norm2")]:
            for suf in (".weight", ".bias"):
                ok = f"{op}{old_n}{suf}"
                if ok in old_sd:
                    new_sd[f"{np}{new_n}{suf}"] = old_sd[ok]

        # Attention: QKV concat
        qkv_w = []
        qkv_b = []
        for proj in ("q_proj", "k_proj", "v_proj"):
            wk = f"{op}self_attn.{proj}.weight"
            bk = f"{op}self_attn.{proj}.bias"
            if wk in old_sd:
                qkv_w.append(old_sd[wk])
            if bk in old_sd:
                qkv_b.append(old_sd[bk])
        if qkv_w:
            new_sd[f"{np}attn.in_proj_weight"] = torch.cat(qkv_w, dim=0)
        if qkv_b:
            new_sd[f"{np}attn.in_proj_bias"] = torch.cat(qkv_b, dim=0)

        # Out proj
        for suf in (".weight", ".bias"):
            ok = f"{op}self_attn.out_proj{suf}"
            if ok in old_sd:
                new_sd[f"{np}attn.out_proj{suf}"] = old_sd[ok]

        # MLP
        for name in ("fc1", "fc2"):
            for suf in (".weight", ".bias"):
                ok = f"{op}mlp.{name}{suf}"
                if ok in old_sd:
                    new_sd[f"{np}mlp.{name}{suf}"] = old_sd[ok]

    # Post layernorm
    for suf in (".weight", ".bias"):
        ok = _SIGLIP_OLD + "post_layernorm" + suf
        if ok in old_sd:
            new_sd["img.encoder.norm" + suf] = old_sd[ok]

    # Multi-modal projector
    for suf in (".weight", ".bias"):
        ok = "paligemma_with_expert.paligemma.model.multi_modal_projector.linear" + suf
        if ok in old_sd:
            new_sd["img.head" + suf] = old_sd[ok]

    # --- PaliGemma LLM (expert 0) ---
    _PALI_LLM = "paligemma_with_expert.paligemma.model.language_model."
    for i in range(18):
        op = f"{_PALI_LLM}layers.{i}."
        np = f"llm.layers.{i}."

        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            ok = f"{op}self_attn.{proj}.weight"
            if ok in old_sd:
                new_sd[f"{np}attn.{proj}.0.weight"] = old_sd[ok]

        # MLP: gate + up → w_gating (transpose + stack)
        gk = f"{op}mlp.gate_proj.weight"
        uk = f"{op}mlp.up_proj.weight"
        if gk in old_sd and uk in old_sd:
            gate_t = old_sd[gk].T.contiguous()  # (hidden_dim, features) → (features, hidden_dim)
            up_t = old_sd[uk].T.contiguous()
            new_sd[f"{np}mlps.0.w_gating"] = torch.stack([gate_t, up_t], dim=0)

        dk = f"{op}mlp.down_proj.weight"
        if dk in old_sd:
            new_sd[f"{np}mlps.0.w_linear"] = old_sd[dk].T.contiguous()

        # Norms
        for old_n, new_n in [("input_layernorm", "pre_attention_norms"), ("post_attention_layernorm", "pre_ffw_norms")]:
            ok = f"{op}{old_n}.weight"
            if ok in old_sd:
                new_sd[f"{np}{new_n}.0.scale"] = old_sd[ok]

    # Final norm expert 0
    ok = _PALI_LLM + "norm.weight"
    if ok in old_sd:
        new_sd["llm.final_norms.0.scale"] = old_sd[ok]

    # --- Gemma Action Expert (expert 1) ---
    _GEMMA_EXP = "paligemma_with_expert.gemma_expert.model."
    for i in range(18):
        op = f"{_GEMMA_EXP}layers.{i}."
        np = f"llm.layers.{i}."

        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            ok = f"{op}self_attn.{proj}.weight"
            if ok in old_sd:
                new_sd[f"{np}attn.{proj}.1.weight"] = old_sd[ok]

        gk = f"{op}mlp.gate_proj.weight"
        uk = f"{op}mlp.up_proj.weight"
        if gk in old_sd and uk in old_sd:
            gate_t = old_sd[gk].T.contiguous()
            up_t = old_sd[uk].T.contiguous()
            new_sd[f"{np}mlps.1.w_gating"] = torch.stack([gate_t, up_t], dim=0)

        dk = f"{op}mlp.down_proj.weight"
        if dk in old_sd:
            new_sd[f"{np}mlps.1.w_linear"] = old_sd[dk].T.contiguous()

        # Norms (ada_modulation)
        for old_n, new_n in [("input_layernorm", "pre_attention_norms"), ("post_attention_layernorm", "pre_ffw_norms")]:
            for suf in (".weight", ".bias"):
                ok = f"{op}{old_n}.dense{suf}"
                if ok in old_sd:
                    new_sd[f"{np}{new_n}.1.ada_modulation{suf}"] = old_sd[ok]

    # Final norm expert 1
    for suf in (".weight", ".bias"):
        ok = _GEMMA_EXP + "norm.dense" + suf
        if ok in old_sd:
            new_sd["llm.final_norms.1.ada_modulation" + suf] = old_sd[ok]

    # --- lm_head → embedder (tied weights — either key works) ---
    lm_head_key = None
    if "paligemma_with_expert.gemma_expert.lm_head.weight" in old_sd:
        lm_head_key = "paligemma_with_expert.gemma_expert.lm_head.weight"
    elif "paligemma_with_expert.paligemma.lm_head.weight" in old_sd:
        lm_head_key = "paligemma_with_expert.paligemma.lm_head.weight"
    if lm_head_key is not None:
        new_sd["llm.embedder.embedding.weight"] = old_sd[lm_head_key]

    # --- Action head (same names in both formats) ---
    for k in old_sd:
        if k.startswith(
            ("action_in_proj", "action_out_proj", "time_mlp_", "state_proj", "action_time_mlp_", "pointnet.")
        ):
            new_sd[k] = old_sd[k]

    return new_sd


def new_to_old_state_dict(new_sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert new-format flat state dict to old-format."""
    old_sd: dict[str, torch.Tensor] = {}

    _SIGLIP_OLD = "paligemma_with_expert.paligemma.model.vision_tower.vision_model."

    # Stem
    for suf in (".weight", ".bias"):
        nk = "img.stem" + suf
        if nk in new_sd:
            old_sd[_SIGLIP_OLD + "embeddings.patch_embedding" + suf] = new_sd[nk]

    if "img.pos_embedding" in new_sd:
        t = new_sd["img.pos_embedding"]
        if t.dim() == 3 and t.shape[0] == 1:
            t = t.squeeze(0)
        old_sd[_SIGLIP_OLD + "embeddings.position_embedding.weight"] = t

    # Encoder layers
    for i in range(27):
        op = f"{_SIGLIP_OLD}encoder.layers.{i}."
        np = f"img.encoder.layers.{i}."

        for old_n, new_n in [("layer_norm1", "norm1"), ("layer_norm2", "norm2")]:
            for suf in (".weight", ".bias"):
                nk = f"{np}{new_n}{suf}"
                if nk in new_sd:
                    old_sd[f"{op}{old_n}{suf}"] = new_sd[nk]

        # QKV split
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
            old_sd["paligemma_with_expert.paligemma.model.multi_modal_projector.linear" + suf] = new_sd[nk]

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

        for old_n, new_n in [("input_layernorm", "pre_attention_norms"), ("post_attention_layernorm", "pre_ffw_norms")]:
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

        for old_n, new_n in [("input_layernorm", "pre_attention_norms"), ("post_attention_layernorm", "pre_ffw_norms")]:
            for suf in (".weight", ".bias"):
                nk = f"{np}{new_n}.1.ada_modulation{suf}"
                if nk in new_sd:
                    old_sd[f"{op}{old_n}.dense{suf}"] = new_sd[nk]

    for suf in (".weight", ".bias"):
        nk = "llm.final_norms.1.ada_modulation" + suf
        if nk in new_sd:
            old_sd[_GEMMA_EXP + "norm.dense" + suf] = new_sd[nk]

    # --- embedder → lm_head ---
    if "llm.embedder.embedding.weight" in new_sd:
        old_sd["paligemma_with_expert.paligemma.lm_head.weight"] = new_sd["llm.embedder.embedding.weight"]
        old_sd["paligemma_with_expert.gemma_expert.lm_head.weight"] = new_sd["llm.embedder.embedding.weight"]

    # --- Action head (pass through) ---
    for k in new_sd:
        if k.startswith(
            ("action_in_proj", "action_out_proj", "time_mlp_", "state_proj", "action_time_mlp_", "pointnet.")
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
    import json
    import os
    import shutil

    import safetensors.torch

    print(f"Loading trained checkpoint: {input_ckpt}")
    sd = torch.load(input_ckpt, map_location="cpu", weights_only=True)
    print(f"  {len(sd)} keys loaded")

    # Strip _orig_mod. prefix (added by torch.compile)
    stripped = {}
    for k, v in sd.items():
        new_k = k.removeprefix("_orig_mod.")
        stripped[new_k] = v
    print(f"  Stripped _orig_mod. prefix from {sum(1 for k in sd if k.startswith('_orig_mod.'))} keys")
    del sd

    # Convert new → old format
    print("Converting new format → old format...")
    old_sd = new_to_old_state_dict(stripped)
    print(f"  {len(old_sd)} keys after conversion")
    del stripped

    # Load reference model for validation and fixing gemma_expert.lm_head.weight
    ref_safetensors = os.path.join(reference_model, "model.safetensors")
    print(f"Loading reference model: {ref_safetensors}")
    ref_sd = safetensors.torch.load_file(ref_safetensors)
    print(f"  {len(ref_sd)} keys in reference")

    # Fix gemma_expert.lm_head.weight — the new model only has a single 2048-dim
    # embedding (PaliGemma's), but the old format needs a separate 1024-dim one for
    # the action expert. This weight is not trained, so we copy from reference.
    expert_lm_head_key = "paligemma_with_expert.gemma_expert.lm_head.weight"
    if expert_lm_head_key in ref_sd and expert_lm_head_key in old_sd:
        if old_sd[expert_lm_head_key].shape != ref_sd[expert_lm_head_key].shape:
            print(f"  Fixing {expert_lm_head_key}: {old_sd[expert_lm_head_key].shape} → {ref_sd[expert_lm_head_key].shape}")
            old_sd[expert_lm_head_key] = ref_sd[expert_lm_head_key].clone()

    # Cast to bfloat16
    print("Casting all tensors to bfloat16...")
    for k in old_sd:
        if old_sd[k].dtype != torch.bfloat16:
            old_sd[k] = old_sd[k].to(torch.bfloat16)

    # Validate against reference
    ref_keys = set(ref_sd.keys())
    conv_keys = set(old_sd.keys())
    missing = ref_keys - conv_keys
    extra = conv_keys - ref_keys
    shape_mismatches = []
    for k in sorted(ref_keys & conv_keys):
        if old_sd[k].shape != ref_sd[k].shape:
            shape_mismatches.append((k, ref_sd[k].shape, old_sd[k].shape))

    print(f"\nValidation:")
    print(f"  Keys: {len(conv_keys)} converted, {len(ref_keys)} reference")
    print(f"  Missing: {len(missing)}")
    for k in sorted(missing):
        print(f"    {k}")
    print(f"  Extra: {len(extra)}")
    for k in sorted(extra):
        print(f"    {k}")
    print(f"  Shape mismatches: {len(shape_mismatches)}")
    for k, ref_s, conv_s in shape_mismatches:
        print(f"    {k}: ref={ref_s} conv={conv_s}")

    if missing or extra or shape_mismatches:
        raise RuntimeError("Validation failed — keys or shapes do not match reference model.")

    del ref_sd

    # Save
    os.makedirs(output_dir, exist_ok=True)

    out_safetensors = os.path.join(output_dir, "model.safetensors")
    print(f"\nSaving model to {out_safetensors} ...")
    safetensors.torch.save_file(old_sd, out_safetensors)
    print(f"  Done ({os.path.getsize(out_safetensors) / 1e9:.2f} GB)")
    del old_sd

    # Copy config.json
    ref_config = os.path.join(reference_model, "config.json")
    out_config = os.path.join(output_dir, "config.json")
    if os.path.exists(ref_config):
        shutil.copy2(ref_config, out_config)
        print(f"Copied config.json")

    # Copy norm_stats.json
    if norm_stats and os.path.exists(norm_stats):
        norm_dst_dir = os.path.join(output_dir, "physical-intelligence", "behavior")
        os.makedirs(norm_dst_dir, exist_ok=True)
        shutil.copy2(norm_stats, os.path.join(norm_dst_dir, "norm_stats.json"))
        print(f"Copied norm_stats.json → {norm_dst_dir}/norm_stats.json")

    print(f"\nConversion complete: {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert a new-format trained checkpoint to old format.",
    )
    parser.add_argument("--input_ckpt", required=True, help="Path to model.pt (new format)")
    parser.add_argument("--output_dir", required=True, help="Output directory for converted checkpoint")
    parser.add_argument("--reference_model", required=True, help="Path to reference model directory (old format)")
    parser.add_argument("--norm_stats", default=None, help="Path to norm_stats.json to include")
    args = parser.parse_args()

    convert_trained_ckpt(
        input_ckpt=args.input_ckpt,
        output_dir=args.output_dir,
        reference_model=args.reference_model,
        norm_stats=args.norm_stats,
    )

