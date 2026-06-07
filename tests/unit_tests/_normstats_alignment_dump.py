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

Each touchpoint of the RLinf SFT-train -> convert -> eval pipeline is DERIVED
from the source that actually drove it -- never from a duplicated literal -- so
the evidence (and the companion test) fail if that source drifts:

* ``sft_train``  -- parsed from the ACTUAL run's dumped ``tensorboard/config.yaml``
  (``model.openpi.assets_dir`` / ``asset_id``) and cross-checked against
  ``run_embodiment.log`` (every FSDP worker must log the same resolved dir).
* ``rlinf_eval`` -- parsed from the eval YAML's ``actor.model.openpi.assets_dir``
  / ``asset_id``.
* ``reference``  -- resolved by executing the reference repo's own
  ``get_config(<name>).assets_dirs`` + ``data.repo_id`` in the reference venv, so
  the canonical source is the reference's code, not a copied path.
* ``converter_out`` -- the file ``sft_to_new_pytorch.py`` copied into the
  converted checkpoint; additionally checked to equal the reference (its input).
* ``reference_ckpt`` -- the norm-stats inside the reference eval checkpoint used
  by the control eval; ``old_pre_switch`` is the pre-switch file, kept only to
  quantify what the ``assets_dir`` switch changed.

Every (assets_dir, asset_id) pair is resolved through the production
:func:`resolve_norm_stats_dir`, the same resolver the SFT loader and eval factory
use, so the hashed file is the one the pipeline really reads.

Run from the repo root::

    python tests/unit_tests/_normstats_alignment_dump.py \
        --out docs/evidence/phase4_normstats_alignment.json
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import pathlib
import subprocess

import yaml

from rlinf.models.embodiment.openpi_pytorch.pi0_model.normalize import (
    resolve_norm_stats_dir,
)

_REPO = pathlib.Path("/mnt/public/xzxuan/repos/RLinf_pi05")

# --- Source locations (the inputs that are parsed, not the answers) ---
_SFT_RUN = _REPO / "logs/20260605-12:39:44-behavior_pi05_vla"
_SFT_DUMPED_CONFIG = _SFT_RUN / "tensorboard/config.yaml"
_SFT_RUN_LOG = _SFT_RUN / "run_embodiment.log"
_EVAL_YAML = (
    _REPO / "examples/embodiment/config/behavior_ppo_openpi_pi05_pytorch_eval.yaml"
)
_CONVERTED = _SFT_RUN / "pi05_sft_pytorch_new"

# Reference resolution is delegated to the reference repo's own code.
_REF_VENV_PY = pathlib.Path("/mnt/public/xzxuan/repos/openpi-comet/.venv/bin/python")
_REF_REPO = pathlib.Path("/mnt/public/xzxuan/repos/openpi-comet-pytorch-mixed")
_REF_CONFIG_NAME = "pi05_b1k-task0000_sft_pytorch_mixed"

# Comparators (not current touchpoints): the reference eval checkpoint's own
# stats and the pre-switch file used by earlier work.
_REF_CKPT = pathlib.Path("/mnt/public/xzxuan/models/ckpt/jax_task0000_sft_29999_ptnew")
_BEHAVIOR_ASSET_ID = "physical-intelligence/behavior"
_OLD_ASSETS = pathlib.Path("/mnt/public/xzxuan/models/pi05-b1kpt50-cs32/assets")
_OLD_REPO_ID = "behavior-1k/2025-challenge-demos"

_RUN_LOG_MARKER = "Loaded BEHAVIOR norm stats from "


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hashed(assets_dir, asset_id) -> dict:
    """Resolve (assets_dir, asset_id) via the production resolver and hash it."""
    directory = resolve_norm_stats_dir(assets_dir, asset_id)
    file = directory / "norm_stats.json"
    return {
        "assets_dir": str(assets_dir),
        "asset_id": asset_id,
        "resolved_file": str(file),
        "sha256": _sha256(file),
        "bytes": file.stat().st_size,
    }


def _dig(tree, *keys):
    for key in keys:
        if not isinstance(tree, dict) or key not in tree:
            raise KeyError(f"missing {'.'.join(keys)} in parsed config")
        tree = tree[key]
    return tree


def derive_sft_train() -> dict:
    """Parse the run's dumped config + cross-check the run log (all workers)."""
    cfg = yaml.safe_load(_SFT_DUMPED_CONFIG.read_text())
    assets_dir = _dig(cfg, "actor", "model", "openpi", "assets_dir")
    asset_id = _dig(cfg, "actor", "model", "openpi", "asset_id")
    entry = _hashed(assets_dir, asset_id)
    resolved_dir = str(pathlib.Path(entry["resolved_file"]).parent)
    log_lines = _SFT_RUN_LOG.read_text().splitlines()
    loaded = [
        line.split(_RUN_LOG_MARKER, 1)[1].strip()
        for line in log_lines
        if _RUN_LOG_MARKER in line
    ]
    entry["runlog_loaded_dirs"] = sorted(set(loaded))
    entry["runlog_worker_loads"] = len(loaded)
    entry["runlog_matches_resolved"] = sorted(set(loaded)) == [resolved_dir]
    return entry


def derive_rlinf_eval() -> dict:
    """Parse the eval YAML's actor.model.openpi.{assets_dir,asset_id}."""
    cfg = yaml.safe_load(_EVAL_YAML.read_text())
    assets_dir = _dig(cfg, "actor", "model", "openpi", "assets_dir")
    asset_id = _dig(cfg, "actor", "model", "openpi", "asset_id")
    return _hashed(assets_dir, asset_id)


@functools.lru_cache(maxsize=1)
def derive_reference() -> dict:
    """Resolve the canonical file via the reference repo's own get_config()."""
    code = (
        "import sys; sys.path.insert(0, 'src')\n"
        "from openpi.training.config import get_config\n"
        f"c = get_config({_REF_CONFIG_NAME!r})\n"
        "print('ASSETS_DIRS', c.assets_dirs)\n"
        "print('REPO_ID', getattr(c.data, 'repo_id', None))\n"
        "print('ASSET_ID', getattr(getattr(c, 'assets', None), 'asset_id', None))\n"
    )
    proc = subprocess.run(
        [str(_REF_VENV_PY), "-c", code],
        cwd=str(_REF_REPO),
        capture_output=True,
        text=True,
        timeout=600,
    )
    fields = {}
    for line in proc.stdout.splitlines():
        for key in ("ASSETS_DIRS", "REPO_ID", "ASSET_ID"):
            if line.startswith(key + " "):
                fields[key] = line.split(" ", 1)[1].strip()
    if "ASSETS_DIRS" not in fields:
        raise RuntimeError(f"reference get_config failed: {proc.stderr[-400:]}")
    # asset_id falls back to repo_id when AssetsConfig.asset_id is unset (None).
    asset_id = (
        fields.get("ASSET_ID")
        if fields.get("ASSET_ID") not in (None, "None")
        else fields["REPO_ID"]
    )
    entry = _hashed(fields["ASSETS_DIRS"], asset_id)
    entry["reference_config_name"] = _REF_CONFIG_NAME
    entry["reference_repo"] = str(_REF_REPO)
    return entry


def _flatten(value) -> list[float]:
    out: list[float] = []
    stack = [value]
    while stack:
        cur = stack.pop()
        if isinstance(cur, list):
            stack.extend(cur)
        elif cur is not None:
            out.append(float(cur))
    return out


def _quantile_diff(old_file: pathlib.Path, new_file: pathlib.Path) -> dict:
    def _raw(path):
        data = json.loads(path.read_text())
        return data.get("norm_stats", data)

    old, new = _raw(old_file), _raw(new_file)
    diff = {}
    for key in sorted(set(old) & set(new)):
        per_field = {}
        for field in ("mean", "std", "q01", "q99"):
            o, n = _flatten(old[key].get(field)), _flatten(new[key].get(field))
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
    sft_train = derive_sft_train()
    rlinf_eval = derive_rlinf_eval()
    reference = derive_reference()

    conv_file = _CONVERTED / _BEHAVIOR_ASSET_ID / "norm_stats.json"
    conv_sha = _sha256(conv_file)
    converter_out = {
        "resolved_file": str(conv_file),
        "sha256": conv_sha,
        "bytes": conv_file.stat().st_size,
        "equals_reference_input": conv_sha == reference["sha256"],
    }
    reference_ckpt = _hashed(_REF_CKPT, _BEHAVIOR_ASSET_ID)
    old_pre_switch = _hashed(_OLD_ASSETS, _OLD_REPO_ID)

    canonical = reference["sha256"]
    current = {
        "sft_train": sft_train["sha256"],
        "converter_out": converter_out["sha256"],
        "rlinf_eval": rlinf_eval["sha256"],
        "reference": reference["sha256"],
    }
    aligned = {name: sha == canonical for name, sha in current.items()}

    old_file = pathlib.Path(old_pre_switch["resolved_file"])
    new_file = pathlib.Path(reference["resolved_file"])

    return {
        "description": (
            "BEHAVIOR norm-stats alignment across the RLinf SFT-train -> convert "
            "-> eval pipeline vs the reference. Every touchpoint is derived from "
            "its driving source (run dump, eval YAML, reference get_config), then "
            "resolved through the production resolve_norm_stats_dir and hashed."
        ),
        "canonical_sha256": canonical,
        "touchpoints": {
            "sft_train": sft_train,
            "converter_out": converter_out,
            "rlinf_eval": rlinf_eval,
            "reference": reference,
            "reference_ckpt": reference_ckpt,
            "old_pre_switch": old_pre_switch,
        },
        "current_touchpoints_aligned": aligned,
        "all_current_touchpoints_aligned": all(aligned.values()),
        "control_eval_uses_canonical": reference_ckpt["sha256"] == canonical,
        "old_is_a_different_file": old_pre_switch["sha256"] != canonical,
        "old_vs_new_quantile_diff": _quantile_diff(old_file, new_file),
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
        "| canonical =",
        evidence["canonical_sha256"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
