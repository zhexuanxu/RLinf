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

"""Gate for the confirmed-fixed instrumented step-0 baseline.

Two layers:

* Pure-logic tests of ``summarize_step0_frames`` prove the rank-disjoint / effective-
  batch-256 invariant detector (no GPU): a rank-disjoint 8x32 capture is accepted
  (fix exercised, 256 distinct frames); a rank-replicated capture (every rank the same
  32 frames) is rejected (union 32, not exercised).
* Committed-artifact tests validate the real captured step-0 provenance + capture
  artifacts when present (skip-gated until the instrumented run has committed them).
"""

from __future__ import annotations

import json
import pathlib

import pytest

import sys

_REPO = pathlib.Path("/mnt/public/xzxuan/repos/RLinf_pi05")
sys.path.insert(0, str(_REPO))

from tools.sft_step0_instrument import frame_hashes, summarize_step0_frames  # noqa: E402

_EV = _REPO / "docs/evidence"
_CAPTURE = _EV / "phase6_sft_step0_capture.json"
_PROVENANCE = _EV / "phase6_sft_step0_provenance.json"


def test_summary_accepts_rank_disjoint_256():
    gathered = [[f"r{r}f{i}" for i in range(32)] for r in range(8)]
    s = summarize_step0_frames(gathered, global_batch_size=256)
    assert s["global_unique_frames"] == 256
    assert s["rank_disjoint"] is True
    assert s["fix_exercised"] is True
    assert s["per_rank_frame_count"] == [32] * 8


def test_summary_rejects_rank_replicated():
    shared = [f"f{i}" for i in range(32)]
    gathered = [list(shared) for _ in range(8)]  # every rank identical (pre-fix bug)
    s = summarize_step0_frames(gathered, global_batch_size=256)
    assert s["global_unique_frames"] == 32
    assert s["rank_disjoint"] is False
    assert s["fix_exercised"] is False


def test_summary_rejects_partial_overlap():
    # rank-disjoint count but union below the configured global batch -> not exercised
    gathered = [[f"r{r}f{i}" for i in range(16)] for r in range(8)]  # 128 distinct
    s = summarize_step0_frames(gathered, global_batch_size=256)
    assert s["rank_disjoint"] is True
    assert s["global_unique_frames"] == 128
    assert s["fix_exercised"] is False  # 128 != 256


def test_frame_hash_byte_identical_to_reference_helper_with_int_tokens():
    """The instrumentation frame hash must be byte-identical to the reference fanout
    helper, INCLUDING integer token ids (a float cast would change the bytes)."""
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    sys.path.insert(0, str(_REPO / "tests/unit_tests"))
    from _ref_fanout_dump import _frame_hashes as ref_frame_hashes

    class _Obs:
        def __init__(self, state, tokenized_prompt):
            self.state = state
            self.tokenized_prompt = tokenized_prompt

    gen = torch.Generator().manual_seed(0)
    state = torch.rand(4, 8, generator=gen, dtype=torch.float32)
    actions = torch.rand(4, 5, 3, generator=gen, dtype=torch.float32)
    tokens = torch.randint(0, 257000, (4, 12), generator=gen, dtype=torch.int64)
    obs = _Obs(state, tokens)
    assert frame_hashes(obs, actions) == ref_frame_hashes(obs, actions)
    # And a float-cast of the int tokens must produce DIFFERENT hashes (proves the
    # test would catch a reintroduced cast).
    obs_float = _Obs(state, tokens.to(torch.float32))
    assert frame_hashes(obs_float, actions) != ref_frame_hashes(obs, actions)
    _ = np


def _load(p):
    if not p.is_file():
        pytest.skip(f"{p.name} not present (instrumented run not yet committed)")
    return json.loads(p.read_text())


def test_committed_capture_proves_fix_exercised():
    d = _load(_CAPTURE)
    assert d["fix_exercised"] is True
    assert d["rank_disjoint"] is True
    assert d["global_unique_frames"] == d["expected_global_batch"]
    assert d["world_size"] >= 2


def test_committed_capture_records_step0_numbers():
    d = _load(_CAPTURE)["step0"]
    assert d["loss"] is not None and d["loss"] == d["loss"]  # finite
    assert d["grad_norm"] is not None
    # step-0 lr equals the reference warmup init within 1e-12
    assert abs(float(d["lr"]) - 2.4975024975024977e-08) <= 1e-12


def test_committed_provenance_has_required_fields():
    d = _load(_PROVENANCE)
    assert d["git"]["rev"]
    assert "dirty" in d["git"]
    assert d["world_size"] == 8
    assert d["global_batch_size"] and d["micro_batch_size"]
    assert d["base_checkpoint"]["fingerprint"]["rollup_sha256"]
    assert d["norm_stats"]["fingerprint"]["rollup_sha256"]
    assert d["norm_stats"]["norm_stats_sha256"]
    for pkg in ("torch", "ray", "numpy"):
        assert d["package_versions"].get(pkg), f"missing package version {pkg}"


def test_committed_provenance_has_resolved_config_and_grad_accum():
    """The plan/contract require a fully-resolved config + explicit gradient
    accumulation; the gate must REJECT null/missing values."""
    d = _load(_PROVENANCE)
    rc = d.get("resolved_config")
    assert rc is not None and isinstance(rc, dict) and rc, "resolved_config null/empty"
    # the resolved config must actually be resolved (no ${...} interpolations left)
    assert "${" not in json.dumps(rc)
    assert d.get("gradient_accumulation") is not None


def test_committed_provenance_has_openpi_reference_provenance():
    """OpenPI/reference provenance must be recorded (reference repo git rev + vendored
    import path), per the round contract."""
    d = _load(_PROVENANCE)
    rp = d.get("reference_provenance")
    assert rp, "missing reference_provenance"
    assert rp.get("reference_git_rev"), "missing reference repo git rev"
    assert rp.get("openpi_vendored_import_path"), "missing openpi import path"
