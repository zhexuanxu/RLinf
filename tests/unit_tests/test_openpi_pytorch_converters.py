# Copyright (c) 2026, RLinf contributors.
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

"""Value-level verification for the three directional checkpoint converters (M4/AC-7).

The converters (``utils/old_to_new.py``, ``utils/new_to_old.py``,
``utils/jax_to_new_pytorch.py``) are deterministic tensor reshuffles — key
renames plus concat/split, stack/unstack, and transpose — with NO floating-point
arithmetic, so within a dtype they are *exact* and fully reversible. This module
verifies:

CPU-hard (always run, synthetic):
  - the exact transforms (q/k/v concat, gate/up transpose+stack, down transpose,
    pos (un)squeeze, shared-embedder source) on small representative dicts;
  - round-trip identity old->new->old and new->old->new;
  - a wrong-transpose injection fails value parity;
  - the documented four-parameter CLI marks all four params required;
  - ``copy_norm_stats`` copies the stats file byte-for-byte.

Reference parity (skip-gated on the multi-GB reference base models being present):
  - run the REAL converter on representative keys loaded from the on-disk
    reference (old ``pi05_base_pytorch`` <-> new ``pi05_base_pytorch_new``) and
    assert the produced keys match the reference target within the tolerance
    PINNED by task10 (old is bf16, new is fp32 -> compare in float within the
    bf16 band: measured max rel-diff ~3.9e-3, pinned rtol=6e-3 / atol=3e-2).

The jax->new conversion loads a ~14GB float32 JAX pytree; per DEC-5 that is
heavy/evidence rather than a CI gate, so its reference value test is additionally
gated behind ``RLINF_RUN_JAX_CONVERT=1`` (the assertion remains the literal AC-7
reference comparison when enabled).
"""

from __future__ import annotations

import json
import os
import pathlib

import pytest
import torch

from rlinf.models.embodiment.openpi_pytorch.utils.new_to_old import (
    ACTION_EXPERT_LM_HEAD,
    new_to_old_state_dict,
)
from rlinf.models.embodiment.openpi_pytorch.utils.old_to_new import (
    convert_old_to_new,
    copy_norm_stats,
    old_to_new_state_dict,
)

# --------------------------------------------------------------------------- #
# task10 pinned spec: reference models + representative keys + tolerances.
# --------------------------------------------------------------------------- #
_OLD_REF = pathlib.Path("/mnt/public/xzxuan/models/pi05_base_pytorch/model.safetensors")
_NEW_REF = pathlib.Path(
    "/mnt/public/xzxuan/models/pi05_base_pytorch_new/model.safetensors"
)
_JAX_REF = pathlib.Path("/mnt/public/xzxuan/models/pi05_base")
# bf16 band measured against the references (max rel-diff ~3.9e-3); pinned w/ margin.
_RTOL, _ATOL = 6e-3, 3e-2

_SIG_OLD = "paligemma_with_expert.paligemma.model.vision_tower.vision_model."
_PALI_LLM = "paligemma_with_expert.paligemma.model.language_model."


# --------------------------------------------------------------------------- #
# Synthetic dicts exercising the trickiest transforms (non-square shapes so a
# missing/dropped transpose is also caught by the resulting shape).
# --------------------------------------------------------------------------- #
def _synthetic_old_sd():
    torch.manual_seed(0)
    sd = {}
    L0 = f"{_SIG_OLD}encoder.layers.0."
    for proj in ("q_proj", "k_proj", "v_proj"):
        sd[f"{L0}self_attn.{proj}.weight"] = torch.randn(4, 4)
        sd[f"{L0}self_attn.{proj}.bias"] = torch.randn(4)
    sd[f"{_SIG_OLD}embeddings.position_embedding.weight"] = torch.randn(3, 4)
    P0 = f"{_PALI_LLM}layers.0."
    # gate/up are (out=6, in=4); down is (out=4, in=6) -> transposes change shape.
    sd[f"{P0}mlp.gate_proj.weight"] = torch.randn(6, 4)
    sd[f"{P0}mlp.up_proj.weight"] = torch.randn(6, 4)
    sd[f"{P0}mlp.down_proj.weight"] = torch.randn(4, 6)
    sd["paligemma_with_expert.paligemma.lm_head.weight"] = torch.randn(10, 4)
    return sd


def test_old_to_new_applies_exact_transforms():
    old = _synthetic_old_sd()
    new = old_to_new_state_dict(old)

    L0 = f"{_SIG_OLD}encoder.layers.0."
    q = old[f"{L0}self_attn.q_proj.weight"]
    k = old[f"{L0}self_attn.k_proj.weight"]
    v = old[f"{L0}self_attn.v_proj.weight"]
    # q/k/v concat along dim 0, in that order.
    assert tuple(new["img.encoder.layers.0.attn.in_proj_weight"].shape) == (12, 4)
    assert torch.equal(
        new["img.encoder.layers.0.attn.in_proj_weight"], torch.cat([q, k, v], dim=0)
    )
    qb = old[f"{L0}self_attn.q_proj.bias"]
    kb = old[f"{L0}self_attn.k_proj.bias"]
    vb = old[f"{L0}self_attn.v_proj.bias"]
    assert torch.equal(
        new["img.encoder.layers.0.attn.in_proj_bias"], torch.cat([qb, kb, vb], dim=0)
    )
    # pos embedding gains a leading broadcast dim.
    pos = old[f"{_SIG_OLD}embeddings.position_embedding.weight"]
    assert tuple(new["img.pos_embedding"].shape) == (1, 3, 4)
    assert torch.equal(new["img.pos_embedding"], pos.unsqueeze(0))
    # gate/up -> stack([gate.T, up.T]) shape (2, in, out); down -> down.T.
    gate = old[f"{_PALI_LLM}layers.0.mlp.gate_proj.weight"]
    up = old[f"{_PALI_LLM}layers.0.mlp.up_proj.weight"]
    down = old[f"{_PALI_LLM}layers.0.mlp.down_proj.weight"]
    assert tuple(new["llm.layers.0.mlps.0.w_gating"].shape) == (2, 4, 6)
    assert torch.equal(
        new["llm.layers.0.mlps.0.w_gating"],
        torch.stack([gate.T.contiguous(), up.T.contiguous()], dim=0),
    )
    assert torch.equal(new["llm.layers.0.mlps.0.w_linear"], down.T.contiguous())
    # shared embedder comes from the PaliGemma lm_head.
    assert torch.equal(
        new["llm.embedder.embedding.weight"],
        old["paligemma_with_expert.paligemma.lm_head.weight"],
    )


def test_new_to_old_inverts_the_transforms():
    old = _synthetic_old_sd()
    new = old_to_new_state_dict(old)
    back = new_to_old_state_dict(new)

    L0 = f"{_SIG_OLD}encoder.layers.0."
    for proj in ("q_proj", "k_proj", "v_proj"):
        assert torch.equal(
            back[f"{L0}self_attn.{proj}.weight"], old[f"{L0}self_attn.{proj}.weight"]
        )
        assert torch.equal(
            back[f"{L0}self_attn.{proj}.bias"], old[f"{L0}self_attn.{proj}.bias"]
        )
    P0 = f"{_PALI_LLM}layers.0."
    for proj in ("gate_proj", "up_proj", "down_proj"):
        assert torch.equal(
            back[f"{P0}mlp.{proj}.weight"], old[f"{P0}mlp.{proj}.weight"]
        )
    assert torch.equal(
        back[f"{_SIG_OLD}embeddings.position_embedding.weight"],
        old[f"{_SIG_OLD}embeddings.position_embedding.weight"],
    )


def test_old_new_round_trip_is_identity():
    """old -> new -> old returns every original key bit-for-bit."""
    old = _synthetic_old_sd()
    back = new_to_old_state_dict(old_to_new_state_dict(old))
    for key, value in old.items():
        assert key in back, f"round-trip dropped {key}"
        assert torch.equal(back[key], value), f"round-trip changed {key}"


def test_new_old_round_trip_is_identity():
    """new -> old -> new returns every original new-format key bit-for-bit."""
    new = old_to_new_state_dict(_synthetic_old_sd())
    again = old_to_new_state_dict(new_to_old_state_dict(new))
    for key, value in new.items():
        assert key in again, f"round-trip dropped {key}"
        assert torch.equal(again[key], value), f"round-trip changed {key}"


def test_wrong_transpose_fails_value_parity():
    """An injected wrong transpose must change the converter output (AC-7 negative).

    The down-projection is stacked-transposed into ``w_linear``; comparing the real
    output against a NON-transposed expectation must fail (and the wrong-transpose
    shape does not even match), proving the transpose is load-bearing.
    """
    old = _synthetic_old_sd()
    new = old_to_new_state_dict(old)
    down = old[f"{_PALI_LLM}layers.0.mlp.down_proj.weight"]  # (4, 6)
    wrong = down.clone()  # missing the .T
    assert new["llm.layers.0.mlps.0.w_linear"].shape != wrong.shape
    assert not torch.equal(new["llm.layers.0.mlps.0.w_linear"], wrong)
    # And a wrong q/k/v order (v,k,q) must not match the real (q,k,v) concat.
    L0 = f"{_SIG_OLD}encoder.layers.0."
    q = old[f"{L0}self_attn.q_proj.weight"]
    k = old[f"{L0}self_attn.k_proj.weight"]
    v = old[f"{L0}self_attn.v_proj.weight"]
    assert not torch.equal(
        new["img.encoder.layers.0.attn.in_proj_weight"], torch.cat([v, k, q], dim=0)
    )


@pytest.mark.parametrize(
    "module_name",
    [
        "rlinf.models.embodiment.openpi_pytorch.utils.old_to_new",
        "rlinf.models.embodiment.openpi_pytorch.utils.new_to_old",
        "rlinf.models.embodiment.openpi_pytorch.utils.jax_to_new_pytorch",
    ],
)
def test_converter_cli_requires_four_params(module_name, monkeypatch):
    """Each converter CLI marks the four path params required; a missing one errors."""
    import importlib

    module = importlib.import_module(module_name)
    # All four present -> argparse succeeds (then convert_* would run; we only need
    # to confirm the parser accepts the 4 params, so stop before the heavy convert).
    full = [
        "prog",
        "--input-model",
        "i",
        "--input-norm-stats",
        "ins",
        "--output-model",
        "o",
        "--output-norm-stats",
        "ons",
    ]
    for drop in (
        "--input-model",
        "--input-norm-stats",
        "--output-model",
        "--output-norm-stats",
    ):
        # Drop the flag and its value (each flag is immediately followed by its value).
        idx = full.index(drop)
        argv = full[:idx] + full[idx + 2 :]
        monkeypatch.setattr("sys.argv", argv)
        with pytest.raises(SystemExit):
            module.main()


def test_copy_norm_stats_is_byte_identical(tmp_path):
    src = tmp_path / "src" / "norm_stats.json"
    src.parent.mkdir(parents=True)
    payload = {"norm_stats": {"state": {"mean": [0.5] * 4, "std": [1.0] * 4}}}
    src.write_text(json.dumps(payload))
    dst = tmp_path / "nested" / "out" / "norm_stats.json"
    copy_norm_stats(src, dst)
    assert dst.is_file()
    assert dst.read_bytes() == src.read_bytes()


def test_convert_old_to_new_writes_model_and_copies_norm_stats_bytewise(tmp_path):
    """The 4-param convert writes the new model and copies norm stats byte-for-byte."""
    import safetensors.torch

    in_dir = tmp_path / "old"
    in_dir.mkdir()
    safetensors.torch.save_file(_synthetic_old_sd(), str(in_dir / "model.safetensors"))
    src_stats = tmp_path / "src_norm_stats.json"
    src_stats.write_text(json.dumps({"norm_stats": {"actions": {"mean": [0.0] * 4}}}))
    out_dir = tmp_path / "new"
    out_stats = tmp_path / "out" / "norm_stats.json"

    convert_old_to_new(in_dir, src_stats, out_dir, out_stats)

    assert (out_dir / "model.safetensors").is_file()
    written = safetensors.torch.load_file(str(out_dir / "model.safetensors"))
    assert "llm.layers.0.mlps.0.w_gating" in written  # transform applied on write
    assert out_stats.read_bytes() == src_stats.read_bytes()


# --------------------------------------------------------------------------- #
# Reference-model value parity (skip-gated on the on-disk references).
# --------------------------------------------------------------------------- #
def _load_keys(path, keys):
    from safetensors import safe_open

    out = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        present = set(f.keys())
        for k in keys:
            if k in present:
                out[k] = f.get_tensor(k)
    return out


def _ref_key(path, key):
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as f:
        return f.get_tensor(key)


# Full structural parity uses the safetensors HEADER only (shapes + dtypes, no
# tensor data) plus `meta` tensors, so the converter runs over the FULL key set
# without loading the 7-13GB models. The converter ops (cat/stack/transpose/
# chunk/unsqueeze) are all shape/dtype-deterministic and work on meta tensors.
_ST_DTYPE = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}


def _safetensors_header(path):
    """Return ``{key: (shape_tuple, torch_dtype)}`` from the safetensors header."""
    import struct

    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return {k: (tuple(v["shape"]), _ST_DTYPE[v["dtype"]]) for k, v in header.items()}


def _meta_state_dict(header):
    return {
        k: torch.empty(shape, dtype=dtype, device="meta")
        for k, (shape, dtype) in header.items()
    }


def _struct_diff(produced, ref_header):
    """Return (missing, extra, shape_mismatches) comparing produced vs a header."""
    produced_keys, ref_keys = set(produced), set(ref_header)
    missing = sorted(ref_keys - produced_keys)
    extra = sorted(produced_keys - ref_keys)
    shape_mismatches = [
        (k, tuple(produced[k].shape), ref_header[k][0])
        for k in sorted(produced_keys & ref_keys)
        if tuple(produced[k].shape) != ref_header[k][0]
    ]
    return missing, extra, shape_mismatches


# ACTION_EXPERT_LM_HEAD is the one old-format tensor the new format does not carry:
# old_to_new keeps only PaliGemma's 2048-wide embedder and drops the 1024-wide
# action-expert head, so new_to_old_state_dict cannot reconstruct it (sourced from a
# reference by convert_trained_ckpt instead). It is excluded from the
# new_to_old_state_dict helper's representable-key structural coverage.


def test_new_to_old_omits_unreconstructible_action_expert_lm_head():
    """new->old must NOT emit a malformed gemma_expert.lm_head (AC-7 regression).

    Feeding the single 2048-wide shared embedder must produce the correct 2048-wide
    paligemma.lm_head and NO gemma_expert.lm_head (the old format's 1024-wide
    action-expert head is not represented in the new format).
    """
    new_sd = {"llm.embedder.embedding.weight": torch.zeros(20, 8)}
    old_sd = new_to_old_state_dict(new_sd)
    assert ACTION_EXPERT_LM_HEAD not in old_sd
    pali = old_sd["paligemma_with_expert.paligemma.lm_head.weight"]
    assert tuple(pali.shape) == (20, 8)
    assert torch.equal(pali, new_sd["llm.embedder.embedding.weight"])


@pytest.mark.skipif(
    not (_OLD_REF.is_file() and _NEW_REF.is_file()),
    reason="reference base models pi05_base_pytorch{,_new} not available",
)
def test_old_to_new_full_structural_parity_vs_reference():
    """old->new produces EXACTLY the new reference key set, shapes, and the
    dtype-preserving (bf16 source) policy — full key/shape/dtype gate (AC-7)."""
    old_header = _safetensors_header(_OLD_REF)
    new_header = _safetensors_header(_NEW_REF)
    produced = old_to_new_state_dict(_meta_state_dict(old_header))

    missing, extra, shape_mismatches = _struct_diff(produced, new_header)
    assert not missing, f"old->new missing new keys: {missing[:10]}"
    assert not extra, f"old->new produced unexpected keys: {extra[:10]}"
    assert not shape_mismatches, f"old->new shape mismatches: {shape_mismatches[:10]}"
    # The old reference is bf16 and the converter is dtype-preserving.
    assert {t.dtype for t in produced.values()} == {torch.bfloat16}


@pytest.mark.skipif(
    not (_OLD_REF.is_file() and _NEW_REF.is_file()),
    reason="reference base models pi05_base_pytorch{,_new} not available",
)
def test_new_to_old_state_dict_representable_keys_vs_reference():
    """Helper-level coverage: the `new_to_old_state_dict` transform maps EVERY
    representable new-format tensor to the old reference key set (minus the one
    old-only action-expert lm-head it cannot reconstruct), with matching shapes and
    no malformed key. This verifies the pure state-dict helper, NOT that the
    four-parameter `convert_new_to_old` writes a complete old checkpoint — that path
    fails loudly (see test_convert_new_to_old_fails_loudly_*)."""
    old_header = _safetensors_header(_OLD_REF)
    new_header = _safetensors_header(_NEW_REF)
    produced = new_to_old_state_dict(_meta_state_dict(new_header))

    # The action-expert head is intentionally not reconstructed; exclude it.
    expected = {k: v for k, v in old_header.items() if k != ACTION_EXPERT_LM_HEAD}
    missing, extra, shape_mismatches = _struct_diff(produced, expected)
    assert not missing, f"new->old missing old keys: {missing[:10]}"
    assert not extra, f"new->old produced unexpected keys: {extra[:10]}"
    assert not shape_mismatches, f"new->old shape mismatches: {shape_mismatches[:10]}"
    # The previously-malformed key must be absent (not a 2048-wide duplicate).
    assert ACTION_EXPERT_LM_HEAD not in produced
    # The new reference is fp32 and the converter is dtype-preserving.
    assert {t.dtype for t in produced.values()} == {torch.float32}


@pytest.mark.skipif(
    not (_OLD_REF.is_file() and _NEW_REF.is_file()),
    reason="reference base models pi05_base_pytorch{,_new} not available",
)
def test_reference_round_trip_is_structurally_faithful():
    """Helper-level round-trip over the FULL reference key universe (meta tensors):
    the `old_to_new_state_dict`/`new_to_old_state_dict` transforms compose to
    reproduce every new key/shape (new->old->new) and every old key/shape except the
    documented action-expert lm-head (old->new->old) — AC-7 round-trip at the
    state-dict helper level."""
    old_header = _safetensors_header(_OLD_REF)
    new_header = _safetensors_header(_NEW_REF)

    # new -> old -> new must reproduce the new key universe exactly.
    new_again = old_to_new_state_dict(
        new_to_old_state_dict(_meta_state_dict(new_header))
    )
    missing, extra, shape_mismatches = _struct_diff(new_again, new_header)
    assert not (missing or extra or shape_mismatches), (
        f"new->old->new drift: missing={missing[:5]} extra={extra[:5]} "
        f"shape={shape_mismatches[:5]}"
    )

    # old -> new -> old reproduces the old key universe except the action-expert head.
    old_again = new_to_old_state_dict(
        old_to_new_state_dict(_meta_state_dict(old_header))
    )
    expected = {k: v for k, v in old_header.items() if k != ACTION_EXPERT_LM_HEAD}
    missing, extra, shape_mismatches = _struct_diff(old_again, expected)
    assert not (missing or extra or shape_mismatches), (
        f"old->new->old drift: missing={missing[:5]} extra={extra[:5]} "
        f"shape={shape_mismatches[:5]}"
    )


@pytest.mark.skipif(
    not _OLD_REF.is_file(),
    reason="reference base model pi05_base_pytorch not available",
)
def test_convert_trained_ckpt_sources_action_expert_lm_head_from_reference(tmp_path):
    """convert_trained_ckpt produces a COMPLETE valid old checkpoint: the 1024-wide
    bf16 action-expert lm-head is sourced from the reference (AC-7 reference-backed
    path), and the reference key/shape validation passes."""
    import safetensors.torch

    from rlinf.models.embodiment.openpi_pytorch.utils.new_to_old import (
        convert_trained_ckpt,
    )

    # A tiny synthetic reference whose old key set is {paligemma.lm_head (2048),
    # gemma_expert.lm_head (1024), one passthrough}, all bf16.
    ref = {
        "paligemma_with_expert.paligemma.lm_head.weight": torch.zeros(
            20, 8, dtype=torch.bfloat16
        ),
        ACTION_EXPERT_LM_HEAD: torch.arange(20 * 4, dtype=torch.bfloat16).reshape(
            20, 4
        ),
        "action_in_proj.weight": torch.zeros(4, 8, dtype=torch.bfloat16),
    }
    ref_dir = tmp_path / "ref"
    ref_dir.mkdir()
    safetensors.torch.save_file(ref, str(ref_dir / "model.safetensors"))

    # A new-format trained checkpoint that maps to {paligemma.lm_head, action_in_proj}
    # but CANNOT produce the 1024-wide action-expert head.
    new_ckpt = tmp_path / "trained.pt"
    torch.save(
        {
            "_orig_mod.llm.embedder.embedding.weight": torch.ones(20, 8),
            "_orig_mod.action_in_proj.weight": torch.ones(4, 8),
        },
        str(new_ckpt),
    )
    out_dir = tmp_path / "out"
    convert_trained_ckpt(str(new_ckpt), str(out_dir), str(ref_dir))

    produced = safetensors.torch.load_file(str(out_dir / "model.safetensors"))
    assert set(produced) == set(ref)  # complete: matches reference key set
    head = produced[ACTION_EXPERT_LM_HEAD]
    assert tuple(head.shape) == (20, 4) and head.dtype == torch.bfloat16
    assert torch.equal(head, ref[ACTION_EXPERT_LM_HEAD])  # sourced from reference
    assert {t.dtype for t in produced.values()} == {torch.bfloat16}  # bf16-cast


def test_convert_new_to_old_fails_loudly_without_action_expert_head(tmp_path):
    """The four-parameter convert_new_to_old must NOT write an incomplete old
    checkpoint: it raises RuntimeError naming the unreconstructible action-expert
    lm-head and writes nothing (AC-7 fail-loud). The complete-output path is the
    reference-backed convert_trained_ckpt."""
    import safetensors.torch

    from rlinf.models.embodiment.openpi_pytorch.utils.new_to_old import (
        convert_new_to_old,
    )

    in_dir = tmp_path / "new"
    in_dir.mkdir()
    # A new-format checkpoint (only the shared embedder + a passthrough) — no way to
    # produce the old 1024-wide action-expert head.
    safetensors.torch.save_file(
        {
            "llm.embedder.embedding.weight": torch.zeros(20, 8),
            "action_in_proj.weight": torch.zeros(4, 8),
        },
        str(in_dir / "model.safetensors"),
    )
    src_stats = tmp_path / "norm_stats.json"
    src_stats.write_text("{}")
    out_dir = tmp_path / "old_out"
    out_stats = tmp_path / "out" / "norm_stats.json"

    with pytest.raises(RuntimeError, match="gemma_expert.lm_head"):
        convert_new_to_old(in_dir, src_stats, out_dir, out_stats)

    # Nothing misleading was written: no output model and no copied norm stats.
    assert not (out_dir / "model.safetensors").exists()
    assert not out_stats.exists()


@pytest.mark.skipif(
    not (_OLD_REF.is_file() and _NEW_REF.is_file()),
    reason="reference base models pi05_base_pytorch{,_new} not available",
)
def test_old_to_new_matches_reference_representative_keys():
    """Run the REAL old->new converter on representative keys from the on-disk old
    reference and assert they match the new reference within the pinned tolerance.
    """
    old_in = [
        _SIG_OLD + "encoder.layers.0.self_attn.q_proj.weight",
        _SIG_OLD + "encoder.layers.0.self_attn.k_proj.weight",
        _SIG_OLD + "encoder.layers.0.self_attn.v_proj.weight",
        _SIG_OLD + "encoder.layers.0.self_attn.q_proj.bias",
        _SIG_OLD + "encoder.layers.0.self_attn.k_proj.bias",
        _SIG_OLD + "encoder.layers.0.self_attn.v_proj.bias",
        _SIG_OLD + "embeddings.position_embedding.weight",
        _PALI_LLM + "layers.0.mlp.gate_proj.weight",
        _PALI_LLM + "layers.0.mlp.up_proj.weight",
        _PALI_LLM + "layers.0.mlp.down_proj.weight",
    ]
    produced = old_to_new_state_dict(_load_keys(_OLD_REF, old_in))
    # Transform-sensitive keys: compare values within the bf16 band. The converter
    # is dtype-preserving, so the produced dtype matches the old (bf16) source even
    # though the new reference is stored as fp32 (hence the float-cast comparison).
    for key in (
        "img.encoder.layers.0.attn.in_proj_weight",
        "img.encoder.layers.0.attn.in_proj_bias",
        "img.pos_embedding",
        "llm.layers.0.mlps.0.w_gating",
        "llm.layers.0.mlps.0.w_linear",
    ):
        ref = _ref_key(_NEW_REF, key)
        got = produced[key]
        assert got.shape == ref.shape, f"{key}: shape {got.shape} vs {ref.shape}"
        assert got.dtype == torch.bfloat16, (
            f"{key}: dtype {got.dtype} (expected bf16 source)"
        )
        torch.testing.assert_close(
            got.float(),
            ref.float(),
            rtol=_RTOL,
            atol=_ATOL,
            msg=f"value drift on {key}",
        )


@pytest.mark.skipif(
    not (_OLD_REF.is_file() and _NEW_REF.is_file()),
    reason="reference base models pi05_base_pytorch{,_new} not available",
)
def test_new_to_old_matches_reference_representative_keys():
    """Run the REAL new->old converter on representative keys from the on-disk new
    reference and assert they match the old reference within the pinned tolerance.
    """
    new_in = [
        "img.encoder.layers.0.attn.in_proj_weight",
        "img.encoder.layers.0.attn.in_proj_bias",
        "img.pos_embedding",
        "llm.layers.0.mlps.0.w_gating",
        "llm.layers.0.mlps.0.w_linear",
    ]
    produced = new_to_old_state_dict(_load_keys(_NEW_REF, new_in))
    # dtype-preserving: produced matches the new (fp32) source; the old reference is
    # stored as bf16, hence the float-cast value comparison.
    for key in (
        _SIG_OLD + "encoder.layers.0.self_attn.q_proj.weight",
        _SIG_OLD + "encoder.layers.0.self_attn.v_proj.weight",
        _SIG_OLD + "embeddings.position_embedding.weight",
        _PALI_LLM + "layers.0.mlp.gate_proj.weight",
        _PALI_LLM + "layers.0.mlp.up_proj.weight",
        _PALI_LLM + "layers.0.mlp.down_proj.weight",
    ):
        ref = _ref_key(_OLD_REF, key)
        got = produced[key]
        assert got.shape == ref.shape, f"{key}: shape {got.shape} vs {ref.shape}"
        assert got.dtype == torch.float32, (
            f"{key}: dtype {got.dtype} (expected fp32 source)"
        )
        torch.testing.assert_close(
            got.float(),
            ref.float(),
            rtol=_RTOL,
            atol=_ATOL,
            msg=f"value drift on {key}",
        )


@pytest.mark.skipif(
    os.environ.get("RLINF_RUN_JAX_CONVERT") != "1"
    or not _JAX_REF.is_dir()
    or not _NEW_REF.is_file(),
    reason="jax->new value check is heavy (loads ~14GB JAX pytree); set "
    "RLINF_RUN_JAX_CONVERT=1 with the JAX reference present to run it (DEC-5 evidence)",
)
def test_jax_to_new_matches_reference_representative_keys():
    """Run the REAL jax->new SigLIP/LLM conversion and assert the most jax-specific
    transforms (NCHW patch-embed permute, q_proj head/dim/width transpose+reshape)
    match the new reference within the pinned tolerance.
    """
    pytest.importorskip("jax")
    pytest.importorskip("orbax")
    from rlinf.models.embodiment.openpi_pytorch.utils.jax_to_new_pytorch import (
        _load_jax_params,
        convert_llm,
        convert_siglip,
    )

    params = _load_jax_params(_JAX_REF)
    produced = {**convert_siglip(params), **convert_llm(params, pi05=True)}
    for key in (
        "img.stem.weight",
        "img.encoder.layers.0.attn.in_proj_weight",
        "llm.layers.0.attn.q_proj.0.weight",
    ):
        ref = _ref_key(_NEW_REF, key)
        got = produced[key]
        assert got.shape == ref.shape, f"{key}: shape {got.shape} vs {ref.shape}"
        torch.testing.assert_close(
            got.float(),
            ref.float(),
            rtol=_RTOL,
            atol=_ATOL,
            msg=f"value drift on {key}",
        )
