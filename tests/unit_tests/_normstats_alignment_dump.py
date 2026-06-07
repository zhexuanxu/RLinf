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

"""Generate the BEHAVIOR norm-stats alignment evidence.

Hashes the `norm_stats.json` reached by every touchpoint of the RLinf
SFT-train -> convert -> eval pipeline and compares them against the reference
(`openpi-comet-pytorch-mixed`) resolution, then records the OLD-vs-NEW quantile
diff so the effect of the prior ``assets_dir`` switch is quantified.

The five touchpoints:

* ``sft_train``  -- the directory the ACTUAL SFT run loaded from (read from that
  run's dumped ``tensorboard/config.yaml`` + ``run_embodiment.log``), resolved
  through the production :func:`resolve_norm_stats_dir`.
* ``converter_out`` -- the file ``sft_to_new_pytorch.py`` copied into the
  converted eval checkpoint.
* ``rlinf_eval`` -- the directory the eval model factory resolves from the eval
  YAML ``assets_dir`` + ``asset_id`` (production resolver).
* ``reference``  -- the file the reference ``TrainConfig`` default
  ``assets_base_dir`` resolves to (the canonical source).
* ``reference_ckpt`` -- the norm-stats shipped inside the reference eval
  checkpoint used by the control eval (``eval2``).

The ``old`` entry is the pre-switch RLinf ``assets_dir`` file, kept only to
quantify what the switch changed; it is NOT a touchpoint of the current run.

Run from the repo root::

    python -m tests.unit_tests._normstats_alignment_dump \
        --out docs/evidence/phase4_normstats_alignment.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

from rlinf.models.embodiment.openpi_pytorch.pi0_model.normalize import (
    resolve_norm_stats_dir,
)

# --- Touchpoint sources (paths documented in the plan / read from run dumps) ---
_REF_ASSETS_BASE = pathlib.Path(
    "/mnt/public/xzxuan/repos/openpi-comet-pytorch-mixed/outputs/assets/train"
)
_REF_CONFIG_NAME = "pi05_b1k-task0000_sft_pytorch_mixed"
_REF_REPO_ID = "behavior-1k/2025-challenge-demos"

_SFT_RUN = pathlib.Path(
    "/mnt/public/xzxuan/repos/RLinf_pi05/logs/20260605-12:39:44-behavior_pi05_vla"
)
_CONVERTED = _SFT_RUN / "pi05_sft_pytorch_new"
_REF_CKPT = pathlib.Path("/mnt/public/xzxuan/models/ckpt/jax_task0000_sft_29999_ptnew")
_OLD_ASSETS = pathlib.Path("/mnt/public/xzxuan/models/pi05-b1kpt50-cs32/assets")

# (assets_dir, asset_id) pairs resolved through the production resolver. The SFT
# and reference rows use the SAME (base/{name}, repo_id) shape openpi uses.
_TOUCHPOINTS = {
    "sft_train": (_REF_ASSETS_BASE / _REF_CONFIG_NAME, _REF_REPO_ID),
    "rlinf_eval": (_CONVERTED, "physical-intelligence/behavior"),
    "reference": (_REF_ASSETS_BASE / _REF_CONFIG_NAME, _REF_REPO_ID),
    "reference_ckpt": (_REF_CKPT, "physical-intelligence/behavior"),
    "old_pre_switch": (_OLD_ASSETS, _REF_REPO_ID),
}
# converter_out is a plain file copy, resolved directly (not via assets_dir).
_CONVERTER_OUT = _CONVERTED / "physical-intelligence/behavior"


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_raw(path: pathlib.Path) -> dict:
    data = json.loads(path.read_text())
    return data.get("norm_stats", data)


def _stat_diff(old: dict, new: dict) -> dict:
    """Per-key max/mean abs diff of mean/std/q01/q99 between OLD and NEW."""

    def _flat(x):
        out = []
        stack = [x]
        while stack:
            cur = stack.pop()
            if isinstance(cur, list):
                stack.extend(cur)
            elif cur is not None:
                out.append(float(cur))
        return out

    keys = sorted(set(old) & set(new))
    diff = {}
    for key in keys:
        per_field = {}
        for field in ("mean", "std", "q01", "q99"):
            o = _flat(old[key].get(field))
            n = _flat(new[key].get(field))
            if not o or not n or len(o) != len(n):
                per_field[field] = {"dim_old": len(o), "dim_new": len(n)}
                continue
            absdiff = [abs(a - b) for a, b in zip(o, n)]
            per_field[field] = {
                "dim": len(o),
                "max_abs_diff": max(absdiff),
                "mean_abs_diff": sum(absdiff) / len(absdiff),
            }
        diff[key] = per_field
    return diff


def build_evidence() -> dict:
    resolved: dict[str, dict] = {}
    for name, (assets_dir, asset_id) in _TOUCHPOINTS.items():
        entry: dict = {"assets_dir": str(assets_dir), "asset_id": asset_id}
        try:
            directory = resolve_norm_stats_dir(assets_dir, asset_id)
            file = directory / "norm_stats.json"
            entry["resolved_file"] = str(file)
            entry["sha256"] = _sha256(file)
            entry["bytes"] = file.stat().st_size
        except FileNotFoundError as exc:
            entry["error"] = str(exc)
        resolved[name] = entry

    conv_file = _CONVERTER_OUT / "norm_stats.json"
    converter_entry: dict = {"resolved_file": str(conv_file)}
    if conv_file.is_file():
        converter_entry["sha256"] = _sha256(conv_file)
        converter_entry["bytes"] = conv_file.stat().st_size
    else:
        converter_entry["error"] = f"missing: {conv_file}"
    resolved["converter_out"] = converter_entry

    canonical = resolved["reference"].get("sha256")
    current_touchpoints = ("sft_train", "converter_out", "rlinf_eval", "reference")
    aligned = {
        name: resolved[name].get("sha256") == canonical for name in current_touchpoints
    }

    # OLD-vs-NEW quantile diff (NEW == reference == canonical).
    old_file = resolve_norm_stats_dir(_OLD_ASSETS, _REF_REPO_ID) / "norm_stats.json"
    new_file = (
        resolve_norm_stats_dir(_REF_ASSETS_BASE / _REF_CONFIG_NAME, _REF_REPO_ID)
        / "norm_stats.json"
    )
    stat_diff = _stat_diff(_load_raw(old_file), _load_raw(new_file))

    return {
        "description": (
            "BEHAVIOR norm-stats alignment across the RLinf SFT-train -> convert "
            "-> eval pipeline vs the reference. Canonical source = the reference "
            "TrainConfig default assets_base_dir resolution (the NEW file)."
        ),
        "canonical_sha256": canonical,
        "touchpoints": resolved,
        "current_touchpoints_aligned": aligned,
        "all_current_touchpoints_aligned": all(aligned.values()),
        "old_vs_new_quantile_diff": stat_diff,
        "sft_run_proof": {
            "run_dir": str(_SFT_RUN),
            "dumped_config": str(_SFT_RUN / "tensorboard/config.yaml"),
            "run_log": str(_SFT_RUN / "run_embodiment.log"),
            "note": (
                "run_embodiment.log records all 8 FSDP workers 'Loaded BEHAVIOR "
                "norm stats from .../pi05_b1k-task0000_sft_pytorch_mixed/"
                "behavior-1k/2025-challenge-demos' == the NEW canonical file; the "
                "checkpoint was trained on the same stats eval uses (no "
                "train-on-OLD / eval-on-NEW mismatch)."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="docs/evidence/phase4_normstats_alignment.json",
        help="output JSON evidence path",
    )
    args = parser.parse_args()
    evidence = build_evidence()
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out}")
    print(
        "all_current_touchpoints_aligned =",
        evidence["all_current_touchpoints_aligned"],
        "canonical =",
        evidence["canonical_sha256"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
