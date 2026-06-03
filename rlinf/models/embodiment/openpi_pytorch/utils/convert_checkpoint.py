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

"""Pre-convert an old-format PyTorch OpenPI 0.5 checkpoint to the new layout.

Reads an old ``model.safetensors`` (the ``paligemma_with_expert.*`` layout used
by the previous PyTorch model and the BEHAVIOR eval checkpoint), converts the
state dict to the new self-contained ``Pi0`` layout via
``utils.checkpoint_format.old_to_new_state_dict``, and writes a new checkpoint
directory containing ``model.safetensors`` plus the original ``config.json`` and
the norm-stats asset tree (so the new directory is self-sufficient for eval).

Usage:
    python -m rlinf.models.embodiment.openpi_pytorch.utils.convert_checkpoint \
        --input  /path/to/old_ckpt_dir \
        --output /path/to/new_ckpt_dir
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil

import safetensors.torch
import torch

from rlinf.models.embodiment.openpi_pytorch.utils import checkpoint_format as cf


def _state_dict_digest(state_dict: dict[str, torch.Tensor]) -> str:
    """A stable digest over (sorted key, dtype, shape) — for a reproducible report."""
    hasher = hashlib.sha256()
    for key in sorted(state_dict):
        tensor = state_dict[key]
        hasher.update(key.encode("utf-8"))
        hasher.update(str(tensor.dtype).encode("utf-8"))
        hasher.update(str(tuple(tensor.shape)).encode("utf-8"))
    return hasher.hexdigest()[:16]


def convert_checkpoint(
    input_dir: str | pathlib.Path, output_dir: str | pathlib.Path
) -> pathlib.Path:
    """Convert ``input_dir/model.safetensors`` (old layout) to ``output_dir`` (new)."""
    input_dir = pathlib.Path(input_dir)
    output_dir = pathlib.Path(output_dir)
    old_path = input_dir / "model.safetensors"
    if not old_path.exists():
        raise FileNotFoundError(f"old checkpoint not found: {old_path}")

    print(f"[convert] loading old checkpoint: {old_path}")
    old_sd = safetensors.torch.load_file(str(old_path), device="cpu")
    print(f"[convert] old tensors: {len(old_sd)}  digest={_state_dict_digest(old_sd)}")

    new_sd = cf.old_to_new_state_dict(old_sd)
    n_params = sum(t.numel() for t in new_sd.values())
    print(
        f"[convert] new tensors: {len(new_sd)}  params={n_params:,}  "
        f"digest={_state_dict_digest(new_sd)}"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "model.safetensors"
    safetensors.torch.save_file(new_sd, str(out_path))
    print(f"[convert] wrote: {out_path} ({out_path.stat().st_size / 1e9:.2f} GB)")

    # Copy config.json (if present) and the norm-stats asset tree so the new dir
    # is self-sufficient for eval.
    config_src = input_dir / "config.json"
    if config_src.exists():
        shutil.copy2(config_src, output_dir / "config.json")
        print(f"[convert] copied config.json: {json.loads(config_src.read_text())}")
    for asset_dir in input_dir.glob("*"):
        if asset_dir.is_dir():
            if asset_dir.resolve() == output_dir.resolve():
                continue
            dst = output_dir / asset_dir.name
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(asset_dir, dst)
            print(f"[convert] copied asset tree: {asset_dir.name}/")

    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="old checkpoint directory")
    parser.add_argument("--output", required=True, help="new checkpoint directory")
    args = parser.parse_args()
    convert_checkpoint(args.input, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
