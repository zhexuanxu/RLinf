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

"""Exact-gate test for BEHAVIOR norm-stats alignment.

The gate is FILE-CONTENT identity (sha256) across every touchpoint of the RLinf
SFT-train -> convert -> eval pipeline and the reference. Each touchpoint is
DERIVED from the source that drove it (the run's dumped config + log, the eval
YAML, the reference repo's own ``get_config``) via the shared derivation helpers
in ``_normstats_alignment_dump`` -- so this test fails if any of those sources
drifts, not merely if a copied path changes. The external assets / reference
venv live outside the repo, so each test skip-gates when its source is absent
while keeping the assertion the literal file-content requirement.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

# Load the sibling derivation module by path (it is not an importable package).
_DUMP_PATH = pathlib.Path(__file__).parent / "_normstats_alignment_dump.py"
_spec = importlib.util.spec_from_file_location("_normstats_alignment_dump", _DUMP_PATH)
dump = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dump)


def _have_sft() -> bool:
    return dump._SFT_DUMPED_CONFIG.is_file() and dump._SFT_RUN_LOG.is_file()


def _have_eval() -> bool:
    return dump._EVAL_YAML.is_file()


def _have_ref_venv() -> bool:
    return dump._REF_VENV_PY.is_file() and dump._REF_REPO.is_dir()


def _have_converted() -> bool:
    return (dump._CONVERTED / dump._BEHAVIOR_ASSET_ID / "norm_stats.json").is_file()


def _have_ref_ckpt() -> bool:
    return (dump._REF_CKPT / dump._BEHAVIOR_ASSET_ID / "norm_stats.json").is_file()


def _have_old() -> bool:
    return (dump._OLD_ASSETS / dump._OLD_REPO_ID / "norm_stats.json").is_file()


@pytest.mark.skipif(
    not (_have_sft() and _have_eval() and _have_converted() and _have_ref_venv()),
    reason="SFT run / eval YAML / converted ckpt / reference venv not all present",
)
def test_all_current_touchpoints_resolve_one_canonical_file():
    """Training, converter output, eval-load, and the reference path -- each
    DERIVED from its driving source -- resolve a byte-identical norm_stats.json
    (the positive gate: content, not config string)."""
    reference = dump.derive_reference()
    conv = dump._CONVERTED / dump._BEHAVIOR_ASSET_ID / "norm_stats.json"
    digests = {
        "sft_train": dump.derive_sft_train()["sha256"],
        "rlinf_eval": dump.derive_rlinf_eval()["sha256"],
        "converter_out": dump._sha256(conv),
        "reference": reference["sha256"],
    }
    assert set(digests.values()) == {reference["sha256"]}, (
        f"norm-stats touchpoints disagree: {digests}"
    )


_EXPECTED_SFT_WORKERS = 8  # this run's FSDP world size (single node, 8 GPUs)


@pytest.mark.skipif(not _have_sft(), reason="SFT run dump/log not present")
def test_sft_runlog_confirms_every_worker_loaded_the_resolved_file():
    """The run log must show every FSDP worker loading the SAME directory that
    the dumped config resolves to, and the number of norm-stats loads must equal
    the FSDP world size derived independently from the base-model load ranks (no
    worker silently loaded a different file or skipped the load)."""
    sft = dump.derive_sft_train()
    assert sft["runlog_matches_resolved"], (
        f"run log dirs {sft['runlog_loaded_dirs']} != resolved {sft['resolved_file']}"
    )
    # Loads must match the independently-derived world size, and that world size
    # must be the expected 8 for this run.
    assert sft["runlog_worker_count_matches_expected"], (
        f"norm-stats loads {sft['runlog_worker_loads']} != expected workers "
        f"{sft['runlog_expected_workers']}"
    )
    assert sft["runlog_worker_loads"] == _EXPECTED_SFT_WORKERS
    assert sft["runlog_expected_workers"] == _EXPECTED_SFT_WORKERS


@pytest.mark.skipif(
    not (_have_ref_venv() and _have_ref_ckpt()),
    reason="reference venv / reference checkpoint not present",
)
def test_control_eval_uses_reference_norm_stats():
    """The control eval loads the reference MODEL but converted norm-stats; verify
    by content that the reference checkpoint's own norm-stats equal the canonical
    file, so the control is not skewed by a different normalization."""
    canonical = dump.derive_reference()["sha256"]
    ref_ckpt = dump._hashed(dump._REF_CKPT, dump._BEHAVIOR_ASSET_ID)["sha256"]
    assert ref_ckpt == canonical


@pytest.mark.skipif(
    not (_have_ref_venv() and _have_old()),
    reason="reference venv / pre-switch assets not present",
)
def test_old_pre_switch_file_is_a_different_file():
    """The gate is file content, not the config string: the pre-switch file is a
    genuinely DIFFERENT norm_stats.json (different sha256), which the content gate
    detects and a string-only check could not."""
    canonical = dump.derive_reference()["sha256"]
    old = dump._hashed(dump._OLD_ASSETS, dump._OLD_REPO_ID)["sha256"]
    assert old != canonical
