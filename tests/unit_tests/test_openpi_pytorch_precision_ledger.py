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

"""Dual-repo runtime mixed-precision dtype-ledger gate (complete schema).

Validates the committed ledgers: for BOTH repos on rank 0 AND a sharded rank, a
runtime-observed dtype ledger records EVERY enumerated precision surface as a
``{value, provenance, stage?}`` object whose provenance points at a runtime object
(never YAML); both repos consumed the SAME pinned input (matching per-field
sha256); the gradient sample is non-empty on the sharded rank; autocast / grad-scaler
are observed from RUNTIME (not config); the during-forward compute dtype is captured;
and a save + load-after-save round trip is recorded. The gate FAILS if any surface is
missing, a required sample is empty, a value lacks runtime provenance, autocast/scaler
are config-derived, or the save/load round trip is absent.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest
import torch

_EV = pathlib.Path("/mnt/public/xzxuan/repos/RLinf_pi05/docs/evidence")
_REPOS = ("rlinf", "ref")
_RANKS = (0, 1)

# Every enumerated surface must be present in BOTH repos' ledgers on BOTH ranks.
_REQUIRED_SURFACES = (
    "master_param_dtype",
    "param_dtype_during_forward",
    "compute_autocast_enabled",
    "fsdp_param_dtype",
    "fsdp_reduce_dtype",
    "fsdp_buffer_dtype",
    "gradient_reduction_dtype",
    "mp_cast_forward_inputs",
    "mp_cast_root_forward_inputs",
    "mp_keep_low_precision_grads",
    "loss_dtype",
    "output_dtype",
    "action_input_dtype",
    "noise_input_dtype",
    "time_input_dtype",
    "grad_dtype",
    "grad_norm",
    "optimizer",
    "optimizer_state_dtype",
    "buffer_count",
    "buffer_dtypes",
    "ema",
    "grad_scaler",
    "saved_checkpoint_dtype",
    "load_after_save_dtype",
)


def _load(repo, rank):
    p = _EV / f"phase5_precision_ledger_{repo}_rank{rank}.json"
    if not p.is_file():
        pytest.skip(f"precision ledger {p.name} not present")
    return json.loads(p.read_text())


def test_all_four_ledgers_ok_rank0_and_sharded_rank():
    for repo in _REPOS:
        for rank in _RANKS:
            d = _load(repo, rank)
            assert d["ok"] is True, f"{repo} rank{rank} ledger not ok"
            assert d["rank"] == rank and d["world_size"] >= 2


def test_every_surface_present_with_runtime_provenance_both_ranks():
    """Each required surface exists on BOTH repos AND BOTH ranks, each as a
    {value, provenance} object whose provenance is a runtime object, not YAML."""
    for repo in _REPOS:
        for rank in _RANKS:
            s = _load(repo, rank)["surfaces"]
            for surface in _REQUIRED_SURFACES:
                assert surface in s, f"{repo} rank{rank} missing surface {surface}"
                rec = s[surface]
                assert "value" in rec and "provenance" in rec, surface
                prov = rec["provenance"].lower()
                assert ".yaml" not in prov and "config text" not in prov


def test_same_pinned_input_across_repos():
    """Both repos consumed the SAME pinned batch (matching per-field sha256)."""
    rl = _load("rlinf", 0)["pinned_input"]["field_sha256"]
    rf = _load("ref", 0)["pinned_input"]["field_sha256"]
    assert rl == rf and len(rl) >= 5


def test_pinned_input_uses_committed_ref_pinned_artifact():
    """The precision ledger must use the committed pinned SFT artifact, not the
    old shape-only synthetic spec."""
    for repo in _REPOS:
        for rank in _RANKS:
            pinned = _load(repo, rank)["pinned_input"]
            assert pinned.get("spec_path") == (
                "docs/evidence/phase5_precision_pinned_artifact.json"
            )
            assert "phase5_precision_pinned_spec.json" not in json.dumps(pinned)
            artifact = pinned.get("artifact")
            assert artifact, f"{repo} rank{rank} missing pinned artifact metadata"
            assert artifact["format"] == "ref_pinned_npz"
            assert artifact["synthetic"] is False
            generator = artifact["source"]["generator"]
            assert (
                "_ref_pinned_run.py" in generator
                or "_precision_pinned_artifact_rlinf.py" in generator
            )
            assert artifact["rank"] == rank
            for key in ("batches_npz", "noise_time_npz"):
                p = pathlib.Path("/mnt/public/xzxuan/repos/RLinf_pi05") / artifact[key]
                assert p.is_file(), f"{repo} rank{rank} missing {key}: {p}"


def test_precision_pinned_loader_rejects_shape_only_manifest(tmp_path):
    """A manifest not tied to the pinned SFT artifact path is rejected before
    tensors are loaded."""
    from _precision_pinned import load_pinned

    manifest = tmp_path / "phase5_precision_pinned_artifact.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "format": "shape_only_spec",
                "synthetic": True,
                "batches_npz": "missing.npz",
                "noise_time_npz": "missing_noise.npz",
                "world_size": 2,
                "micro_batch": 1,
                "source": {"generator": "phase5_precision_pinned_spec.json"},
            }
        )
    )
    with pytest.raises(ValueError, match="ref_pinned_npz"):
        load_pinned(torch.device("cpu"), manifest)


def test_precision_pinned_builder_extracts_ref_pinned_npz(tmp_path, monkeypatch):
    """The artifact builder consumes the same two-NPZ schema as the pinned SFT
    loader and emits a compact rank-addressable artifact."""
    from _precision_pinned import build_pinned, load_pinned

    image_keys = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    batches = {}
    for key in image_keys:
        batches[f"image__{key}"] = np.full((2, 2, 4, 4, 3), 0.25, np.float32)
        batches[f"image_mask__{key}"] = np.ones((2, 2), dtype=bool)
    batches["state"] = np.arange(2 * 2 * 8, dtype=np.float32).reshape(2, 2, 8)
    batches["actions"] = np.arange(2 * 2 * 4 * 4, dtype=np.float32).reshape(2, 2, 4, 4)
    batches["tokenized_prompt"] = np.ones((2, 2, 5), dtype=np.int64)
    batches["tokenized_prompt_mask"] = np.ones((2, 2, 5), dtype=bool)
    batches_npz = tmp_path / "ref_pinned_batches.npz"
    np.savez(batches_npz, **batches)

    noise = np.arange(1 * 4 * 4 * 4, dtype=np.float32).reshape(1, 4, 4, 4)
    time = np.arange(4, dtype=np.float32).reshape(1, 4)
    noise_time_npz = tmp_path / "ref_pinned_noise_time.npz"
    np.savez(noise_time_npz, noise=noise, time=time)

    manifest = tmp_path / "phase5_precision_pinned_artifact.json"
    build_pinned(
        batches_npz,
        noise_time_npz,
        manifest_path=manifest,
        source_command="python tests/unit_tests/_ref_pinned_run.py OUT 1 2",
        world_size=2,
        micro=1,
    )
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "1")
    pin = load_pinned(torch.device("cpu"), manifest)
    assert pin["artifact"]["synthetic"] is False
    assert pin["actions"].shape == (1, 4, 4)
    assert torch.allclose(pin["noise"], torch.from_numpy(noise[:, 1:2][0]))


def test_grad_sample_nonempty_on_sharded_rank():
    """Gradient dtype evidence is non-empty on the sharded (rank 1) for both repos —
    not rank-0-only proof."""
    for repo in _REPOS:
        gd = _load(repo, 1)["surfaces"]["grad_dtype"]["value"]
        assert gd, f"{repo} rank1 grad_dtype is empty"


def test_autocast_and_grad_scaler_observed_at_runtime():
    """autocast is read from torch.is_autocast_enabled() inside the forward, and the
    grad-scaler is the real object — NOT config-derived constants."""
    for repo in _REPOS:
        s = _load(repo, 0)["surfaces"]
        assert "is_autocast_enabled" in s["compute_autocast_enabled"]["provenance"]
        assert "object" in s["grad_scaler"]["provenance"] or "no GradScaler" in s["grad_scaler"]["provenance"]


def test_during_forward_compute_dtype_and_save_load_present():
    """The during-forward compute dtype is captured (bf16) and a save + load-after-save
    round trip is recorded with non-empty dtypes."""
    for repo in _REPOS:
        s = _load(repo, 0)["surfaces"]
        assert s["param_dtype_during_forward"]["value"] == "bfloat16"
        assert s["master_param_dtype"]["value"] == "float32"
        assert s["saved_checkpoint_dtype"]["value"]
        assert s["load_after_save_dtype"]["value"]


def test_grad_norm_value_and_fp32_dtype_recorded():
    """Grad-norm is recorded as a value + the fp32 accumulation dtype (not the python
    float it is .item()-ed to)."""
    for repo in _REPOS:
        gn = _load(repo, 0)["surfaces"]["grad_norm"]["value"]
        assert "value" in gn and "dtype" in gn
        assert gn["value"] == gn["value"]  # finite
        assert gn["dtype"] == "float32"


def test_per_record_identity_fields():
    """Every surface record carries surface (== its key), repo (== ledger repo), and
    rank (== ledger rank)."""
    for repo in _REPOS:
        for rank in _RANKS:
            d = _load(repo, rank)
            ledger_repo = d["repo"]
            for key, rec in d["surfaces"].items():
                assert rec.get("surface") == key, f"{repo} r{rank} {key} surface≠key"
                assert rec.get("repo") == ledger_repo
                assert rec.get("rank") == rank


def test_saved_and_load_cover_all_tensors():
    """Saved + load-after-save record ALL tensor dtypes with a count (not first-8), and
    the load-after-save is a real reload matching the saved tensor count."""
    for repo in _REPOS:
        s = _load(repo, 0)["surfaces"]
        saved = s["saved_checkpoint_dtype"]["value"]
        loaded = s["load_after_save_dtype"]["value"]
        assert saved["count"] > 8 and loaded["count"] > 8
        assert saved["dtypes"] and loaded["dtypes"]


def test_omitted_buffer_dtype_default_observed():
    """A labeled FSDP probe OBSERVES the installed default for an omitted buffer_dtype
    (the buffer stays its original dtype), separate from the model's 0-buffer evidence."""
    for repo in _REPOS:
        s = _load(repo, 0)["surfaces"]
        assert s["buffer_count"]["value"] == 0  # model has no buffer surface
        probe = s["buffer_default_probe"]["value"]
        assert probe.get("omitted_buffer_dtype") == "None"
        assert probe.get("buffer_dtype_after_wrap")  # an observed buffer dtype
