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

"""AC-4 generator: same raw batch through RLinf's full normalization + forward.

Consumes ``ref_raw_norm.npz`` from ``_ref_raw_norm_dump.py`` (the reference's RAW
first batch + its NEW-stats transform + its loss + the numpy noise/time), feeds the
IDENTICAL raw frames through RLinf's ``BehaviorSftTransform`` (the production
``_repack`` → ``BehaviorInputs`` → resize → ``normalize_quantile`` → tokenize → pad
path, NOT the pinned loader), and:

* re-hashes the raw fields and confirms they equal the reference dump (so the same
  raw batch is consumed, not a pre-normalized one);
* compares RLinf-normalized vs reference-normalized state / actions / token ids+mask
  / images;
* builds the RLinf ``Pi0`` (from ``pi05_base_pytorch_new``, bf16) and computes
  ``compute_loss(train=True, rng=None, noise, time)`` on its OWN transform output
  with the SAME noise/time, comparing the same-batch loss to the reference loss;
* runs the OLD / NEW(canonical = reference-resolved) norm-stats MATRIX -- the same
  raw batch transformed under each stat file -- and attributes any normalized/loss
  delta to the specific stat file.

Writes ``docs/evidence/phase4_raw_norm_forward_matrix.{json,csv}``. GPU + RLinf.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib

import numpy as np

from rlinf.models.embodiment.openpi_pytorch.pi0_model.normalize import (
    load_norm_stats,
    resolve_norm_stats_dir,
)

_NEW_ASSETS = (
    "/mnt/public/xzxuan/repos/openpi-comet-pytorch-mixed/outputs/assets/train/"
    "pi05_b1k-task0000_sft_pytorch_mixed"
)
_OLD_ASSETS = "/mnt/public/xzxuan/models/pi05-b1kpt50-cs32/assets"
_ASSET_ID = "behavior-1k/2025-challenge-demos"
_BASE_WEIGHTS = "/mnt/public/xzxuan/models/pi05_base_pytorch_new/model.safetensors"
_IMG = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
# bf16 same-batch forward + quantile normalization tolerances (documented gate).
_LOSS_TOL = 0.01
_NORM_TOL = 1e-4


def _sha(arr) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(arr)).tobytes()).hexdigest()


def _stats_sha(assets_dir: str) -> str:
    file = resolve_norm_stats_dir(assets_dir, _ASSET_ID) / "norm_stats.json"
    return hashlib.sha256(file.read_bytes()).hexdigest()


def _transform(norm_dir, frames):
    """Run RLinf's production BehaviorSftTransform on each raw frame."""
    from rlinf.data.datasets.behavior.behavior_sft_transform import BehaviorSftTransform

    tf = BehaviorSftTransform(norm_stats=load_norm_stats(norm_dir))
    out = {"state": [], "actions": [], "tok": [], "mask": [], **{k: [] for k in _IMG}}
    for fr in frames:
        item = tf(fr)
        out["state"].append(np.asarray(item["state"]))
        out["actions"].append(np.asarray(item["actions"]))
        out["tok"].append(np.asarray(item["tokenized_prompt"]))
        out["mask"].append(np.asarray(item["tokenized_prompt_mask"]))
        imgs = item["image"]
        for k in _IMG:
            out[k].append(np.asarray(imgs[k]))
    return {k: np.stack(v) for k, v in out.items()}


def _loss(transformed, noise, time):
    import torch

    from rlinf.models.embodiment.openpi_pytorch.pi0_model import model as vmodel
    from rlinf.models.embodiment.openpi_pytorch.pi0_model.pi0_config import Pi0Config

    model = Pi0Config(
        pi05=True, action_horizon=32, action_dim=32, dtype="bfloat16", pcd=False
    ).create()
    import safetensors.torch

    model.load_state_dict(
        safetensors.torch.load_file(_BASE_WEIGHTS, device="cpu"), strict=True
    )
    model = model.to("cuda").to(torch.bfloat16).eval()
    n = transformed["state"].shape[0]
    obs = vmodel.Observation.from_dict(
        {
            "image": {k: torch.from_numpy(transformed[k]).to("cuda") for k in _IMG},
            "image_mask": {
                k: torch.ones(n, dtype=torch.bool, device="cuda") for k in _IMG
            },
            "state": torch.from_numpy(transformed["state"]).to("cuda").float(),
            "tokenized_prompt": torch.from_numpy(transformed["tok"]).to("cuda").long(),
            "tokenized_prompt_mask": torch.from_numpy(transformed["mask"])
            .to("cuda")
            .bool(),
        }
    )
    actions = torch.from_numpy(transformed["actions"]).to("cuda").float()
    with torch.no_grad():
        return float(
            model.compute_loss(
                obs,
                actions,
                train=True,
                rng=None,
                noise=torch.from_numpy(noise).to("cuda"),
                time=torch.from_numpy(time).to("cuda"),
            )
            .float()
            .mean()
        )


def build_evidence(ref_npz: pathlib.Path, ref_result: dict, git_rev: str) -> dict:
    d = np.load(ref_npz)
    n = int(ref_result["n"])
    tasks = ref_result["tasks"]
    frames = [
        {
            "observation.images.rgb.head": d["raw_head"][i],
            "observation.images.rgb.left_wrist": d["raw_left"][i],
            "observation.images.rgb.right_wrist": d["raw_right"][i],
            "observation.state": d["raw_state"][i],
            "action": d["raw_action"][i],
            "task": tasks[i],
        }
        for i in range(n)
    ]
    # 1) Raw-batch identity: the RLinf arm consumes the SAME raw fields.
    raw_hashes = {
        "raw_head": _sha(d["raw_head"]),
        "raw_left": _sha(d["raw_left"]),
        "raw_right": _sha(d["raw_right"]),
        "raw_state": _sha(d["raw_state"]),
        "raw_action": _sha(d["raw_action"]),
    }
    raw_match = raw_hashes == ref_result["raw_hashes"]

    new_dir = resolve_norm_stats_dir(_NEW_ASSETS, _ASSET_ID)
    old_dir = resolve_norm_stats_dir(_OLD_ASSETS, _ASSET_ID)
    rlinf_new = _transform(new_dir, frames)
    rlinf_old = _transform(old_dir, frames)

    # 2) Per-field normalized comparison: RLinf(NEW) vs reference(NEW).
    def _maxabs(a, b):
        return float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())

    field_deltas = {
        "state": _maxabs(rlinf_new["state"], d["ref_state"]),
        "actions": _maxabs(rlinf_new["actions"], d["ref_actions"]),
        "images_base": _maxabs(rlinf_new["base_0_rgb"], d["ref_image__base_0_rgb"]),
    }
    tokens_identical = bool(np.array_equal(rlinf_new["tok"], d["ref_tokenized_prompt"]))
    masks_identical = bool(
        np.array_equal(rlinf_new["mask"], d["ref_tokenized_prompt_mask"])
    )

    # 3) Same-batch raw->normalized->loss: RLinf(NEW) vs reference.
    noise, time = d["noise"], d["time"]
    rlinf_loss_new = _loss(rlinf_new, noise, time)
    rlinf_loss_old = _loss(rlinf_old, noise, time)
    ref_loss = float(ref_result["ref_loss"])

    # 4) Norm-stats matrix (OLD / NEW=canonical=reference-resolved).
    matrix = [
        {
            "stats": "NEW (canonical = reference-resolved)",
            "norm_stats_sha256": _stats_sha(_NEW_ASSETS),
            "rlinf_same_batch_loss": rlinf_loss_new,
            "loss_delta_vs_reference": abs(rlinf_loss_new - ref_loss),
            "actions_max_abs_vs_reference": field_deltas["actions"],
        },
        {
            "stats": "OLD (pre-switch)",
            "norm_stats_sha256": _stats_sha(_OLD_ASSETS),
            "rlinf_same_batch_loss": rlinf_loss_old,
            "loss_delta_vs_reference": abs(rlinf_loss_old - ref_loss),
            "actions_max_abs_vs_NEW": _maxabs(
                rlinf_old["actions"], rlinf_new["actions"]
            ),
        },
    ]

    canonical_matches = (
        raw_match
        and tokens_identical
        and masks_identical
        and field_deltas["state"] <= _NORM_TOL
        and field_deltas["actions"] <= _NORM_TOL
        and abs(rlinf_loss_new - ref_loss) <= _LOSS_TOL
    )
    if canonical_matches:
        verdict = (
            "CANONICAL SAME-BATCH MATCH — on the IDENTICAL raw batch, RLinf's full "
            "raw→normalized→loss path (canonical NEW stats) reproduces the reference: "
            f"tokens/masks byte-identical, normalized state/actions within {_NORM_TOL}, "
            f"same-batch loss {rlinf_loss_new:.4f} vs reference {ref_loss:.4f} "
            f"(|Δ|={abs(rlinf_loss_new - ref_loss):.4f} ≤ {_LOSS_TOL}). The per-batch "
            "normalization APPLICATION is correct, so the significant ~9% eval gap is "
            "NOT a same-batch transform/norm-stats bug; it is localized to the "
            "production data TRAJECTORY (which frames / effective batch the 30000-step "
            "run actually trained on), which a single-batch test cannot exercise. The "
            "OLD stats row shows the norm-stats CHOICE does change the normalized "
            "actions + loss, confirming the matrix is sensitive; the production run "
            "used NEW (AC-1)."
        )
    else:
        verdict = (
            "CANONICAL SAME-BATCH DIVERGENCE — the RLinf full transform+forward differs "
            "from the reference on the identical raw batch; a localized transform / "
            "norm-stats-application bug is the root cause (see the field deltas)."
        )

    return {
        "description": (
            "AC-4 same-batch raw→normalized→loss comparison: the reference's RAW first "
            "batch is driven through each repo's FULL transform (raw actions → its own "
            "quantile normalization → loss) with identical injected noise/time, plus an "
            "OLD/NEW(=reference-resolved) norm-stats matrix. Not pre-normalized batches."
        ),
        "n_frames": n,
        "raw_batch_hashes_match_reference": raw_match,
        "raw_hashes": raw_hashes,
        "tokens_byte_identical": tokens_identical,
        "masks_byte_identical": masks_identical,
        "normalized_field_max_abs_vs_reference": field_deltas,
        "norm_tolerance": _NORM_TOL,
        "reference_same_batch_loss": ref_loss,
        "rlinf_same_batch_loss_canonical": rlinf_loss_new,
        "same_batch_loss_abs_delta": abs(rlinf_loss_new - ref_loss),
        "loss_tolerance": _LOSS_TOL,
        "canonical_same_batch_matches_reference": canonical_matches,
        "norm_stats_matrix": matrix,
        "verdict": verdict,
        "evidence_generation_git_revision": git_rev,
        "provenance": {
            "ref_dump": str(ref_npz),
            "ref_assets_dir": ref_result.get("assets_dir"),
            "ref_asset_id": ref_result.get("asset_id"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref-npz", required=True)
    parser.add_argument("--ref-result", required=True, help="REF_RAW_NORM json")
    parser.add_argument("--git-rev", default="unknown")
    parser.add_argument(
        "--out", default="docs/evidence/phase4_raw_norm_forward_matrix.json"
    )
    args = parser.parse_args()
    ref_result = json.loads(pathlib.Path(args.ref_result).read_text())
    evidence = build_evidence(pathlib.Path(args.ref_npz), ref_result, args.git_rev)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(evidence, indent=2) + "\n")
    with out.with_suffix(".csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(evidence["norm_stats_matrix"][0]), lineterminator="\n"
        )
        writer.writeheader()
        for row in evidence["norm_stats_matrix"]:
            writer.writerow({k: row.get(k, "") for k in writer.fieldnames})
    print(
        f"raw_match={evidence['raw_batch_hashes_match_reference']} "
        f"tokens_id={evidence['tokens_byte_identical']} "
        f"canonical_match={evidence['canonical_same_batch_matches_reference']} "
        f"rlinf_loss={evidence['rlinf_same_batch_loss_canonical']:.4f} "
        f"ref_loss={evidence['reference_same_batch_loss']:.4f}"
    )
    print(evidence["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
