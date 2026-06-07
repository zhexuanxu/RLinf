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

"""Exact-gate test for BEHAVIOR norm-stats alignment (AC-1).

The gate is FILE-CONTENT identity (sha256) across every touchpoint of the RLinf
SFT-train -> convert -> eval pipeline and the reference, resolved through the
PRODUCTION :func:`resolve_norm_stats_dir` -- not config-string equality. The
external asset files live outside the repo, so each test skip-gates when they
are absent (e.g. CI) while keeping the assertion the AC's literal requirement.
"""

from __future__ import annotations

import hashlib
import pathlib

import pytest

from rlinf.models.embodiment.openpi_pytorch.pi0_model.normalize import (
    resolve_norm_stats_dir,
)

# Touchpoint sources (documented in docs/phase4-normstats-alignment-evidence.md).
_REF_ASSETS_DIR = pathlib.Path(
    "/mnt/public/xzxuan/repos/openpi-comet-pytorch-mixed/outputs/assets/train/"
    "pi05_b1k-task0000_sft_pytorch_mixed"
)
_REF_REPO_ID = "behavior-1k/2025-challenge-demos"
_CONVERTED = pathlib.Path(
    "/mnt/public/xzxuan/repos/RLinf_pi05/logs/20260605-12:39:44-behavior_pi05_vla/"
    "pi05_sft_pytorch_new"
)
_BEHAVIOR_ASSET_ID = "physical-intelligence/behavior"
_REF_CKPT = pathlib.Path("/mnt/public/xzxuan/models/ckpt/jax_task0000_sft_29999_ptnew")
_OLD_ASSETS = pathlib.Path("/mnt/public/xzxuan/models/pi05-b1kpt50-cs32/assets")

# The four touchpoints of the CURRENT pipeline, as (assets_dir, asset_id) pairs
# fed through the production resolver. (a) SFT-train and (d) reference share the
# openpi base/{name} + repo_id shape; (b) converter-out and (c) eval-load share
# the converted-checkpoint layout.
_CURRENT_TOUCHPOINTS = {
    "sft_train": (_REF_ASSETS_DIR, _REF_REPO_ID),
    "reference": (_REF_ASSETS_DIR, _REF_REPO_ID),
    "converter_out": (_CONVERTED, _BEHAVIOR_ASSET_ID),
    "rlinf_eval": (_CONVERTED, _BEHAVIOR_ASSET_ID),
}


def _resolved_file(assets_dir: pathlib.Path, asset_id: str) -> pathlib.Path:
    return resolve_norm_stats_dir(assets_dir, asset_id) / "norm_stats.json"


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assets_present(*pairs: tuple[pathlib.Path, str]) -> bool:
    return all(
        (assets_dir / asset_id / "norm_stats.json").is_file()
        for assets_dir, asset_id in pairs
    )


_HAVE_CURRENT = _assets_present(*_CURRENT_TOUCHPOINTS.values())
_HAVE_REF_CKPT = _assets_present((_REF_CKPT, _BEHAVIOR_ASSET_ID))
_HAVE_OLD = _assets_present((_OLD_ASSETS, _REF_REPO_ID))


@pytest.mark.skipif(
    not _HAVE_CURRENT, reason="BEHAVIOR norm-stats asset files not present"
)
def test_all_current_touchpoints_resolve_one_canonical_file():
    """(a) SFT-train, (b) converter-out, (c) eval-load, (d) reference all
    resolve a byte-identical norm_stats.json (AC-1 positive gate: content, not
    config string)."""
    canonical = _sha256(_resolved_file(_REF_ASSETS_DIR, _REF_REPO_ID))
    digests = {
        name: _sha256(_resolved_file(assets_dir, asset_id))
        for name, (assets_dir, asset_id) in _CURRENT_TOUCHPOINTS.items()
    }
    assert set(digests.values()) == {canonical}, (
        f"norm-stats touchpoints disagree: {digests}"
    )


@pytest.mark.skipif(
    not (_HAVE_CURRENT and _HAVE_REF_CKPT),
    reason="BEHAVIOR norm-stats / reference-ckpt asset files not present",
)
def test_eval2_control_uses_reference_norm_stats():
    """The eval2 control loads the reference MODEL but RLinf-converted norm-stats;
    AC-1 flags this confound. Verify by content that the reference checkpoint's
    own norm-stats equal the canonical file (so the control is not skewed)."""
    canonical = _sha256(_resolved_file(_REF_ASSETS_DIR, _REF_REPO_ID))
    ref_ckpt = _sha256(_resolved_file(_REF_CKPT, _BEHAVIOR_ASSET_ID))
    assert ref_ckpt == canonical


@pytest.mark.skipif(
    not (_HAVE_CURRENT and _HAVE_OLD),
    reason="BEHAVIOR norm-stats / OLD asset files not present",
)
def test_old_pre_switch_file_is_a_different_file():
    """The gate is file content, not the config string: the OLD pre-switch file
    is a genuinely DIFFERENT norm_stats.json (different sha256), which the
    content gate detects. A string-only check could not."""
    canonical = _sha256(_resolved_file(_REF_ASSETS_DIR, _REF_REPO_ID))
    old = _sha256(_resolved_file(_OLD_ASSETS, _REF_REPO_ID))
    assert old != canonical
